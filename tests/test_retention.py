"""Archival must preserve replay and reject loss, corruption and capacity overruns."""
import asyncio
import json
import hashlib

import httpx
import pytest

from timepredict.retention import archive_history, read_history
from timepredict.service import create_app
from timepredict.store import RedisStore
from timepredict.client import PredictionClient, WindowConsumer


def test_archive_trim_replay_and_append_are_lossless(tmp_path):
    fakeredis = pytest.importorskip("fakeredis.aioredis")
    async def run():
        redis = fakeredis.FakeRedis(decode_responses=True)
        for i in range(1, 5):
            await redis.xadd("p:history", {"data": json.dumps({"event_id": str(i)})}, id=f"{i}-0")
        original = await read_history(redis, "p")
        result = await archive_history(redis, "p", tmp_path, trim_history=True)
        assert result["archived_events"] == 4 and await redis.xlen("p:history") == 0
        with pytest.raises(ValueError, match="archive required"):
            await read_history(redis, "p")
        assert await read_history(redis, "p", archive_dir=tmp_path) == original
        await redis.xadd("p:history", {"data": '{"event_id":"5"}'}, id="5-0")
        assert await read_history(redis, "p", archive_dir=tmp_path) == original+[{"event_id": "5"}]
        await archive_history(redis, "p", tmp_path, trim_history=True)
        assert len(await read_history(redis, "p", archive_dir=tmp_path)) == 5
        with pytest.raises(ValueError, match="history_capacity_exceeded"):
            await read_history(redis, "p", archive_dir=tmp_path, max_events=4)
        with pytest.raises(ValueError, match="history_capacity_exceeded"):
            await read_history(redis, "p", archive_dir=tmp_path, max_bytes=1)
        marker = json.loads(await redis.get("p:history-archive"))
        index = json.loads((tmp_path/marker["index_file"]).read_text())
        (tmp_path/index["segments"][0]["file"]).write_text("corrupt\n")
        with pytest.raises(ValueError, match="checksum"):
            await read_history(redis, "p", archive_dir=tmp_path)
        await redis.aclose()
    asyncio.run(run())


def test_raw_event_trim_preserves_pending_and_undelivered_entries(tmp_path):
    fakeredis = pytest.importorskip("fakeredis.aioredis")
    async def run():
        redis = fakeredis.FakeRedis(decode_responses=True)
        for i in range(1, 5):
            await redis.xadd("p:events", {"data": "{}"}, id=f"{i}-0")
        await redis.xgroup_create("p:events", "predictor", id="0")
        await redis.xreadgroup("predictor", "worker", {"p:events": ">"}, count=3)
        await redis.xack("p:events", "predictor", "1-0", "2-0")
        result = await archive_history(redis, "p", tmp_path, trim_events=True)
        assert result["trimmed_raw_events"] == 2
        assert [ident for ident, _ in await redis.xrange("p:events")] == ["3-0", "4-0"]
        await redis.aclose()
    asyncio.run(run())


def test_archive_io_failure_cannot_trim_history(tmp_path, monkeypatch):
    fakeredis = pytest.importorskip("fakeredis.aioredis")
    from timepredict import retention
    async def run():
        redis = fakeredis.FakeRedis(decode_responses=True)
        await redis.xadd("p:history", {"data": "{}"}, id="1-0")
        def failed(*args):
            raise OSError("fixture disk failure")
        monkeypatch.setattr(retention.os, "fsync", failed)
        with pytest.raises(OSError, match="disk failure"):
            await archive_history(redis, "p", tmp_path, trim_history=True)
        assert await redis.xlen("p:history") == 1
        assert not await redis.exists("p:history-archive")
        await redis.aclose()
    asyncio.run(run())


def test_storage_budget_and_notice_resync(tmp_path):
    fakeredis = pytest.importorskip("fakeredis.aioredis")
    async def run():
        redis = fakeredis.FakeRedis(decode_responses=True)
        store = RedisStore(redis, "p", notice_limit=2)
        def notify(state):
            return state, [{"action": "remove", "tenant": "t", "key": "k", "generation": "g", "revision": 1}]
        await store.mutate(notify)
        old = (await redis.xrange(store.notices))[0][0]
        for _ in range(3):
            await store.mutate(notify)
        assert await redis.xlen(store.notices) == 2
        app = create_app(store, {"secret": "t"})
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://fixture") as client:
            response = await client.get("/internal/notices", params={"after": old}, headers={"Authorization": "Bearer secret"})
            assert response.json()["resync_required"] is True and response.json()["events"] == []
        limited = RedisStore(redis, "limited", max_state_bytes=1)
        with pytest.raises(RuntimeError, match="state_capacity_exceeded"):
            await limited.mutate(lambda state: (state, []))
        assert not await redis.exists(limited.state_key)
        await redis.aclose()
    asyncio.run(run())


def test_notification_resync_blocks_notices_until_snapshot():
    consumer = WindowConsumer(lambda: 0.)
    consumer.notice_page({"resync_required": True, "events": []})
    consumer.notice({"action": "candidate_update"})  # Ignored until authoritative resync.
    assert consumer.resync_required and consumer.candidates() == []
    consumer.snapshot({"revision": 1, "snapshot_utc": 1., "H_seconds": 2., "records": []})
    assert not consumer.resync_required


def test_prediction_client_fetches_snapshot_when_notice_cursor_is_trimmed():
    async def run():
        requests = []
        def handler(request):
            requests.append(request.url.path)
            if request.url.path.endswith("notices"):
                return httpx.Response(200, json={"resync_required": True, "events": [], "cursor": "1-0"})
            return httpx.Response(200, json={"revision": len(requests), "snapshot_utc": float(len(requests)),
                                            "H_seconds": 2., "records": [], "notice_cursor": "10-0"})
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            client = PredictionClient(http, "http://fixture", "secret", monotonic=lambda: 0.)
            await client.refresh(2.)
            await client.poll_notices()
            assert requests == ["/internal/window", "/internal/notices", "/internal/window"]
            assert not client.consumer.resync_required and client.notice_cursor == "10-0"
    asyncio.run(run())


def test_archive_lost_lease_cannot_trim_history(tmp_path, monkeypatch):
    fakeredis = pytest.importorskip("fakeredis.aioredis")
    async def run():
        redis = fakeredis.FakeRedis(decode_responses=True)
        await redis.xadd("p:history", {"data": "{}"}, id="1-0")
        pipeline = redis.pipeline
        def replace_at_commit(*args, **kwargs):
            pipe = pipeline(*args, **kwargs)
            execute = pipe.execute
            async def takeover(*a, **kw):
                await redis.set("p:archive-lease", "replacement")
                return await execute(*a, **kw)
            monkeypatch.setattr(pipe, "execute", takeover)
            return pipe
        monkeypatch.setattr(redis, "pipeline", replace_at_commit)
        with pytest.raises(Exception):
            await archive_history(redis, "p", tmp_path, trim_history=True)
        assert await redis.xlen("p:history") == 1
        assert not await redis.exists("p:history-archive")
        await redis.aclose()
    asyncio.run(run())


def test_archive_cannot_reseal_a_tampered_catalogue(tmp_path):
    fakeredis = pytest.importorskip("fakeredis.aioredis")
    async def run():
        redis = fakeredis.FakeRedis(decode_responses=True)
        await redis.xadd("p:history", {"data": '{"event_id":"1"}'}, id="1-0")
        await archive_history(redis, "p", tmp_path, trim_history=True)
        original = await redis.get("p:history-archive")
        marker = json.loads(original)
        path = tmp_path/marker.get("index_file", "archive-index.json")
        index = json.loads(path.read_text())
        index["segments"] = []  # Keep a valid prefix/identity but remove the only archived range.
        path.write_text(json.dumps(index))
        with pytest.raises(ValueError, match="checksum"):
            await archive_history(redis, "p", tmp_path, trim_history=True)
        assert await redis.get("p:history-archive") == original
        await redis.aclose()
    asyncio.run(run())


def test_stale_archive_writer_cannot_damage_replacement_catalogue(tmp_path, monkeypatch):
    fakeredis = pytest.importorskip("fakeredis.aioredis")
    from timepredict import retention
    async def run():
        redis = fakeredis.FakeRedis(decode_responses=True)
        await redis.xadd("p:history", {"data": '{"event_id":"1"}'}, id="1-0")
        await archive_history(redis, "p", tmp_path, trim_history=True)
        loop = asyncio.get_running_loop()
        verify = retention._verified_segment
        replaced = False
        async def takeover():
            await redis.delete("p:archive-lease")  # Simulate expiry while the old process is paused.
            await redis.xadd("p:history", {"data": '{"event_id":"2"}'}, id="2-0")
            await archive_history(redis, "p", tmp_path, trim_history=True)
        def delayed_verify(directory, segment):
            nonlocal replaced
            if not replaced:
                replaced = True
                asyncio.run_coroutine_threadsafe(takeover(), loop).result(timeout=5)
            return verify(directory, segment)
        monkeypatch.setattr(retention, "_verified_segment", delayed_verify)
        with pytest.raises(RuntimeError, match="lease lost"):
            await archive_history(redis, "p", tmp_path, trim_history=True)
        marker = json.loads(await redis.get("p:history-archive"))
        path = tmp_path/marker.get("index_file", "archive-index.json")
        assert hashlib.sha256(path.read_bytes()).hexdigest() == marker["index_sha256"]
        assert await read_history(redis, "p", archive_dir=tmp_path) == [{"event_id": "1"}, {"event_id": "2"}]
        await redis.aclose()
    asyncio.run(run())


def test_legacy_catalogue_migrates_to_immutable_index(tmp_path):
    fakeredis = pytest.importorskip("fakeredis.aioredis")
    async def run():
        redis = fakeredis.FakeRedis(decode_responses=True)
        await redis.xadd("p:history", {"data": '{"event_id":"1"}'}, id="1-0")
        await archive_history(redis, "p", tmp_path, trim_history=True)
        marker = json.loads(await redis.get("p:history-archive"))
        (tmp_path/"archive-index.json").write_bytes((tmp_path/marker.pop("index_file")).read_bytes())
        await redis.set("p:history-archive", json.dumps(marker))
        assert await read_history(redis, "p", archive_dir=tmp_path) == [{"event_id": "1"}]
        await redis.xadd("p:history", {"data": '{"event_id":"2"}'}, id="2-0")
        await archive_history(redis, "p", tmp_path, trim_history=True)
        assert json.loads(await redis.get("p:history-archive"))["index_file"] != "archive-index.json"
        assert len(await read_history(redis, "p", archive_dir=tmp_path)) == 2
        await redis.aclose()
    asyncio.run(run())


@pytest.mark.parametrize("raw", ["null", "false", "{}", "[]"])
@pytest.mark.parametrize("with_directory", [False, True])
def test_present_but_invalid_marker_cannot_enable_partial_replay(raw, with_directory, tmp_path):
    fakeredis = pytest.importorskip("fakeredis.aioredis")
    async def run():
        redis = fakeredis.FakeRedis(decode_responses=True)
        await redis.set("p:history-archive", raw)
        with pytest.raises(ValueError, match="archive required|invalid archive marker"):
            await read_history(redis, "p", archive_dir=tmp_path if with_directory else None)
        await redis.aclose()
    asyncio.run(run())
