"""Verified local history archives; Redis trimming follows durable archive publication."""
import asyncio
import hashlib
import json
import os
import time
import uuid
from pathlib import Path

from .model import _flat_name, _fsync_directory, _read_json_object


def stream_id(value):
    parts = value.split("-")
    if len(parts) != 2 or any(not p.isascii() or not p.isdecimal() for p in parts):
        raise ValueError("invalid stream ID")
    result = tuple(map(int, parts))
    if any(n > 2**64-1 for n in result):
        raise ValueError("stream ID overflow")
    return result


def _next_id(value):
    ms, seq = stream_id(value)
    return f"{ms}-{seq+1}" if seq < 2**64-1 else f"{ms+1}-0"


def _index(directory, prefix):
    path = Path(directory)/"archive-index.json"
    if not path.exists():
        return {"prefix": prefix, "archive_id": uuid.uuid4().hex, "segments": []}
    index = _read_json_object(path)
    if index.get("prefix") != prefix or not isinstance(index.get("segments"), list) or not _flat_name(index.get("archive_id")):
        raise ValueError("archive index does not match prefix")
    last = (0, 0)
    for segment in index["segments"]:
        if (not _flat_name(segment["file"]) or type(segment["count"]) is not int or segment["count"] <= 0
                or stream_id(segment["first"]) <= last or stream_id(segment["last"]) < stream_id(segment["first"])):
            raise ValueError("invalid archive segment ordering")
        last = stream_id(segment["last"])
    return index


def _verified_segment(directory, segment):
    root = Path(directory).resolve()
    path = (root/segment["file"]).resolve()
    if path.parent != root:
        raise ValueError("archive path escapes directory")
    digest = hashlib.sha256()
    count = 0
    with path.open("rb") as handle:
        for line in handle:
            digest.update(line)
            count += 1
    if digest.hexdigest() != segment["sha256"] or count != segment["count"]:
        raise ValueError("archive checksum/count mismatch")
    return path


async def read_history(redis, prefix, *, archive_dir=None, max_events=100000, max_bytes=128*1024*1024):
    """Bound total replay input. Never silently discard older workflow events."""
    if type(max_events) is not int or max_events <= 0 or type(max_bytes) is not int or max_bytes <= 0:
        raise ValueError("positive history event budget required")
    tail = await redis.xrevrange(prefix+":history", count=1)
    boundary = tail[0][0] if tail else "0-0"
    marker_raw = await redis.get(prefix+":history-archive")
    marker = json.loads(marker_raw) if marker_raw else None
    index = _index(archive_dir, prefix) if archive_dir else None
    if marker and (index is None or marker["archive_id"] != index["archive_id"]
                   or marker["index_sha256"] != hashlib.sha256((Path(archive_dir)/"archive-index.json").read_bytes()).hexdigest()):
        raise ValueError("verified history archive required before replay")
    events, last, used_bytes = [], "0-0", 0
    def append(raw):
        nonlocal used_bytes
        if len(events) >= max_events:
            raise ValueError("history_capacity_exceeded; increase budget or migrate replay storage")
        used_bytes += len(raw.encode("utf-8"))
        if used_bytes > max_bytes:
            raise ValueError("history_capacity_exceeded; replay byte budget exceeded")
        events.append(json.loads(raw))
    if index:
        if sum(s["count"] for s in index["segments"]) > max_events:
            raise ValueError("history_capacity_exceeded; archive exceeds replay budget")
        for segment in index["segments"]:
            path = await asyncio.to_thread(_verified_segment, archive_dir, segment)
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    append(line)
                    if len(events) % 1000 == 0:
                        await asyncio.sleep(0)
            last = segment["last"]
    cursor = "("+last
    while True:
        batch = await redis.xrange(prefix+":history", min=cursor, max=boundary, count=min(1000, max_events-len(events)+1))
        if not batch:
            if await redis.get(prefix+":history-archive") != marker_raw:
                raise ValueError("history archive changed during replay; retry with the current archive")
            return events
        for _, fields in batch:
            append(fields["data"])
        cursor = "("+batch[-1][0]


async def archive_history(redis, prefix, directory, *, older_than_days=7., trim_history=False, trim_events=False):
    """One writer per prefix. Raw events are trimmed only below pending/undelivered IDs."""
    if not isinstance(older_than_days, (float, int)) or not 0 < older_than_days < float("inf"):
        raise ValueError("positive finite retention days required")
    key, token = prefix+":archive-lease", uuid.uuid4().hex
    if not await redis.set(key, token, nx=True, ex=120):
        raise RuntimeError("archive writer already running")
    async def renew():
        while True:
            await asyncio.sleep(30)
            if not await redis.eval("if redis.call('get',KEYS[1]) == ARGV[1] then return redis.call('expire',KEYS[1],120) else return 0 end", 1, key, token):
                raise RuntimeError("archive lease lost")
    lease_task = asyncio.create_task(renew())
    try:
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        index = _index(root, prefix)
        raw = await redis.get(prefix+":history-archive")
        if raw and json.loads(raw)["archive_id"] != index["archive_id"]:
            raise ValueError("archive identity mismatch; do not start a second archive")
        for segment in index["segments"]:
            await asyncio.to_thread(_verified_segment, root, segment)
        last = index["segments"][-1]["last"] if index["segments"] else "0-0"
        boundary = f"{max(0, int((time.time()-older_than_days*86400)*1000))}-0"
        filename = "history-"+uuid.uuid4().hex+".jsonl"
        path = root/filename
        cursor, first, count, digest = "("+last, None, 0, hashlib.sha256()
        # No history is removed if writing, flushing, checksum verification or index publication fails.
        with path.open("xb") as handle:
            while True:
                if lease_task.done():
                    await lease_task
                batch = await redis.xrange(prefix+":history", min=cursor, max=boundary, count=1000)
                if not batch:
                    break
                for ident, fields in batch:
                    data = (fields["data"]+"\n").encode("utf-8")
                    json.loads(fields["data"])
                    handle.write(data)
                    digest.update(data)
                    first = first or ident
                    last, count = ident, count+1
                cursor = "("+batch[-1][0]
            handle.flush()
            os.fsync(handle.fileno())
        if count:
            segment = {"file": filename, "first": first, "last": last, "count": count, "sha256": digest.hexdigest()}
            await asyncio.to_thread(_verified_segment, root, segment)
            index["segments"].append(segment)
        else:
            path.unlink()  # The exclusive, empty file created by this invocation only.
        from .pipeline import atomic_json
        atomic_json(root/"archive-index.json", index)
        _fsync_directory(root)
        marker = json.dumps({"archive_id": index["archive_id"],
                             "index_sha256": hashlib.sha256((root/"archive-index.json").read_bytes()).hexdigest()})
        # WATCH the lease and stream types to avoid a partial SET + WRONGTYPE XTRIM.
        async with redis.pipeline(transaction=True) as pipe:
            await pipe.watch(key, prefix+":history", prefix+":history-archive")
            if await pipe.get(key) != token:
                raise RuntimeError("archive lease lost before trim")
            if await pipe.type(prefix+":history") not in ("none", "stream", b"none", b"stream"):
                raise ValueError("history must be a stream")
            pipe.multi()
            pipe.set(prefix+":history-archive", marker)
            if trim_history and last != "0-0":
                pipe.xtrim(prefix+":history", minid=_next_id(last), approximate=False)
            await pipe.execute()
        trimmed_events = 0
        if trim_events:
            # Trim no further than the oldest unacknowledged entry (or last delivered).
            async with redis.pipeline(transaction=True) as pipe:
                await pipe.watch(key, prefix+":events")
                if await pipe.get(key) != token:
                    raise RuntimeError("archive lease lost before event trim")
                groups = await pipe.xinfo_groups(prefix+":events")
                if len(groups) != 1 or groups[0]["name"] != "predictor":
                    raise ValueError("event trim requires only the predictor consumer group")
                pending = await pipe.xpending(prefix+":events", "predictor")
                safe = pending["min"] or groups[0]["last-delivered-id"]
                limit = min((safe, boundary), key=stream_id)
                # Conservatively retain the boundary entry; no pending message is removed.
                pipe.multi()
                pipe.xtrim(prefix+":events", minid=limit, approximate=False)
                trimmed_events = (await pipe.execute())[0]
        return {"archived_events": count, "through": last, "history_trimmed": trim_history,
                "trimmed_raw_events": trimmed_events, "archive_id": index["archive_id"]}
    finally:
        lease_task.cancel()
        await asyncio.gather(lease_task, return_exceptions=True)
        await redis.eval("if redis.call('get',KEYS[1]) == ARGV[1] then return redis.call('del',KEYS[1]) else return 0 end", 1, key, token)
