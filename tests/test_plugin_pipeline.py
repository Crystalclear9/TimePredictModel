"""Deterministic protocol/production-gate fixtures; these are not server acceptance evidence."""
import asyncio
import copy
import json
import math
import time
from types import SimpleNamespace

import numpy as np
import pytest

from timepredict.client import WindowConsumer
from timepredict.dataset import validate_label_provenance, validate_rows
from timepredict.features import FeatureSpec
from timepredict.model import Model
from timepredict.release import promote
from timepredict.service import Worker, create_app, window_rows
from timepredict.state import apply_prediction, empty_state, reduce_event
from timepredict.training import snapshot, train


def stamp(t):
    return {"utc": t, "mono": t, "source": "fixture", "boot": "b", "error": .001, "jumped": False}


def event(kind, t, request="r1", workflow="w", **fields):
    return {"event_id": f"{workflow}/{request}/{kind}/{t}", "kind": kind, "stamp": stamp(t), "tenant": "tenant",
            "workflow_id": workflow, "agent_id": "a", "request_id": request, "source_kind": "sglang",
            "serial_protocol": True, "call": {"id": "call1", "name": "search", "description": "search documents",
            "schema": {}, "arguments": {"max_results": 20, "query": "KV cache"}}, **fields}


def waiting():
    state, _ = reduce_event(empty_state(), event("prepared", 1))
    state, _ = reduce_event(state, event("out", 1))
    state["processor"] = stamp(2)
    return state, next(iter(state["records"].values()))


def test_prepared_round_is_predictable_and_closes_once():
    state, record = waiting()
    assert record["prediction_status"] == "pending_prediction"
    assert apply_prediction(state, record["generation"], {"mu": math.log(3), "sigma": .5, "model_version": "v",
                                                       "distribution_calibrated": True}, "ready")
    row = window_rows(state, stamp(3), 2)["records"][0]
    assert row["status"] == "ready" and row["in_window"]
    incoming = event("ingress", 4, request="r2", latest_tool_ids=["call1"])
    state, notices = reduce_event(state, incoming)
    state, duplicated = reduce_event(state, incoming)
    assert not duplicated and len(notices) == 1
    assert next(iter(state["records"].values()))["duration"] == 3
    assert not apply_prediction(state, record["generation"], {}, "ready")


@pytest.mark.parametrize("reason", ["delivery_failed", "unsupported_multi_tool", "incomplete_arguments"])
def test_prepared_failure_never_becomes_candidate(reason):
    state, _ = reduce_event(empty_state(), event("prepared", 1))
    state, _ = reduce_event(state, event("out", 1, reason=reason))
    record = next(iter(state["records"].values()))
    assert not record["valid"] and not state["pending"] and record["status"] == reason


def test_late_ingress_reconciles_prepared_out_and_concurrency():
    state, _ = reduce_event(empty_state(), event("prepared", 1))
    state, _ = reduce_event(state, event("ingress", 3, request="r2", latest_tool_ids=["call1"]))
    state, _ = reduce_event(state, event("out", 1))
    assert next(iter(state["records"].values()))["status"] == "returned"
    state, _ = waiting()
    state, _ = reduce_event(state, event("prepared", 2, request="other"))
    state, _ = reduce_event(state, event("out", 2, request="other"))
    assert not state["pending"]
    assert all(r["status"] == "concurrent_conflict" for r in state["records"].values())


@pytest.mark.parametrize("mode", ["serial_fallback", "tool_call_id"])
def test_snapshot_formal_association_and_tamper_detection(mode):
    ids = ["call1"] if mode == "tool_call_id" else []
    rows, _ = snapshot([event("prepared", 1), event("out", 1), event("ingress", 4, request="r2", latest_tool_ids=ids)], 5)
    validate_label_provenance(rows, data_kind="observed", production=True)
    assert rows[0]["association_mode"] == mode
    broken = copy.deepcopy(rows)
    broken[0]["lower"] = broken[0]["upper"] = 99
    with pytest.raises(ValueError, match="duration_mismatch"):
        validate_label_provenance(broken, data_kind="observed")


def test_censoring_keeps_observation_boundary_and_excludes_cancellation():
    events = [event("out", 1, seq=1), event("source_heartbeat", 5, seq=2), event("heartbeat", 5)]
    rows, _ = snapshot(events, 6, cutoff_stamp=stamp(6))
    assert rows[0]["lower"] == 5 and math.isinf(rows[0]["upper"]) and rows[0]["t_return"] is None
    validate_label_provenance(rows, data_kind="observed", production=True)
    events.append(event("cancel", 5.5))
    rows, excluded = snapshot(events, 6, cutoff_stamp=stamp(6))
    assert not rows and excluded["cancelled"] == 1


def test_structured_features_preserve_identity_missingness_and_unknown_tool():
    call = event("out", 1)["call"]
    spec = FeatureSpec.fit([call])
    assert spec.feature_mode == "structured" and spec.embedding_dim == 0
    modified = {**call, "arguments": {"timeout": 20, "query": "KV cache"}}
    a, b = spec.transform(call), spec.transform(modified)
    assert a.shape == b.shape and not np.array_equal(a, b)
    numeric = spec.fields.index('["search","/max_results","number"]')
    assert a[-len(spec.fields)+numeric] == 20
    assert math.isnan(b[-len(spec.fields)+numeric])
    unknown = spec.transform({**call, "name": "new_search"})
    assert unknown[len(spec.tools)] == 1


def test_shadow_and_uncalibrated_models_cannot_schedule():
    for prediction, expected in (({"mu": math.log(3), "sigma": .5, "model_version": "v"}, "unknown"),
                                 ({"mu": math.log(3), "sigma": .5, "model_version": "v", "shadow": True}, "ready")):
        state, record = waiting()
        apply_prediction(state, record["generation"], prediction, "ready")
        row = window_rows(state, stamp(2), 2)["records"][0]
        assert row["status"] == expected and not row["in_window"]
        if prediction.get("shadow"):
            assert 0 < row["prob_return_within_H"] < 1


def test_consumer_reconditions_cached_distribution_and_expires_without_network():
    state, record = waiting()
    apply_prediction(state, record["generation"], {"mu": math.log(3), "sigma": .5, "model_version": "v",
                                                "distribution_calibrated": True}, "ready")
    now = [0.]
    consumer = WindowConsumer(lambda: now[0])
    view = window_rows(state, stamp(3), 2)
    consumer.snapshot(view)
    before = consumer.candidates()[0]
    now[0] = .5
    after = consumer.candidates()[0]
    assert after["age_seconds"] == before["age_seconds"]+.5
    assert after["remaining_p50_seconds"] != before["remaining_p50_seconds"]
    now[0] = 3
    assert consumer.candidates() == []
    stale = copy.deepcopy(view)
    stale["revision"] -= 1
    consumer.snapshot(stale)
    assert consumer.candidates() == []


def test_end_to_end_training_publication_and_window(tmp_path):
    # Locally fabricated protocol/acceptance fixtures exercise the gate wiring only.
    events = []
    for i in range(24):
        start, duration, wf = i*20., 2.+i%5, f"w{i}"
        events.extend([event("prepared", start+1, workflow=wf), event("out", start+1, workflow=wf),
                       event("ingress", start+1+duration, request="r2", workflow=wf, latest_tool_ids=["call1"]),
                       event("terminate", start+2+duration, workflow=wf)])
    rows, _ = snapshot(events, 1000)
    version, report = train(rows, tmp_path/"candidates", rounds=3, threads=1, sigmas=(.5,), data_kind="observed")
    candidate = tmp_path/"candidates"/version
    assert Model(candidate).predict(rows[0]["call"])[0]["distribution_calibrated"] is False
    with pytest.raises(ValueError, match="not_approved"):
        Model(candidate, require_promoted=True)
    policy = {"min_supported_rows": 1, "min_effective_samples": .1, "horizons": [2.], "remaining_ages": [1.],
              "max_window_calibration_error": 1., "max_quantile_coverage_error": 1., "max_brier_regression": 1.,
              "max_entry_hook_p99_seconds": 1., "max_throughput_drop": 1., "max_prediction_available_p99_seconds": 1.,
              "evidence_max_age_seconds": 3600}
    evidence = {"kind": "server_acceptance", "fixture_only": True, "created_at": time.time(), "model_version": version,
                "checks": {k: True for k in ("sglang_nonstream", "sglang_stream", "cross_ingress_roundtrip", "clock_bounds", "worker_failover", "prediction_observed")},
                "metrics": {"throughput_drop_fraction": 0., "entry_hook_p99_seconds": .001,
                            "prediction_available_p99_seconds": .01, "request_errors": 0, "dropped_events": 0}}
    for name, obj in (("policy", policy), ("evidence", evidence)):
        (tmp_path/f"{name}.json").write_text(json.dumps(obj))
    def lost_lease():
        raise RuntimeError("fixture training lease lost")
    with pytest.raises(RuntimeError, match="lease lost"):
        promote(candidate, tmp_path/"models", tmp_path/"evidence.json", tmp_path/"policy.json", before_commit=lost_lease)
    assert not (tmp_path/"models"/"CURRENT").exists()
    promote(candidate, tmp_path/"models", tmp_path/"evidence.json", tmp_path/"policy.json")
    loaded = Model(tmp_path/"models", require_promoted=True)
    prediction, status = loaded.predict(rows[0]["call"])
    assert status == "ready" and prediction["distribution_calibrated"]
    state, record = waiting()
    apply_prediction(state, record["generation"], prediction, status)
    view = window_rows(state, stamp(2), 10)
    assert view["records"][0]["in_window"] and view["records"][0]["expected_capped_remaining_seconds"] > 0


def test_redis_transactions_and_async_worker(tmp_path):
    fakeredis = pytest.importorskip("fakeredis.aioredis")
    from timepredict.store import RedisStore
    async def run():
        redis = fakeredis.FakeRedis(decode_responses=True)
        store = RedisStore(redis, "test-plugin")
        await store.apply(event("prepared", 1))
        await store.apply(event("out", 1))
        await store.apply(event("out", 1))
        state = await store.read()
        record = next(iter(state["records"].values()))
        class ModelFixture:
            def predict(self, call):
                return {"mu": math.log(3), "sigma": .5, "model_version": "fixture"}, "ready"
        class ClockFixture:
            def stamp(self):
                return stamp(2)
        worker = Worker(store, clock=ClockFixture(), shadow=True)
        worker.model = ModelFixture()
        await worker.predict_one(record)
        updated = await store.read()
        assert updated["records"][record["generation"]]["prediction_status"] == "ready"
        assert await redis.xlen(store.history) == 2
        await store.apply(event("ingress", 4, request="r2", latest_tool_ids=["call1"]))
        assert not (await store.read())["pending"]
        await redis.aclose()
    asyncio.run(run())


def test_http_contract_authentication_and_tenant_isolation():
    import httpx
    fakeredis = pytest.importorskip("fakeredis.aioredis")
    from timepredict.store import RedisStore
    async def run():
        redis = fakeredis.FakeRedis(decode_responses=True)
        store = RedisStore(redis, "api-test")
        await store.apply(event("out", 1))
        class ClockFixture:
            def stamp(self): return stamp(2)
        app = create_app(store, {"secret": "tenant", "other": "different-tenant"}, clock=ClockFixture())
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            assert (await client.get("/internal/contract")).status_code == 401
            contract = await client.get("/internal/contract", headers={"Authorization": "Bearer secret"})
            assert contract.json()["api_version"] == 1
            response = await client.get("/internal/window", params={"H_seconds": 2}, headers={"Authorization": "Bearer other"})
            assert response.json()["records"] == []
            assert (await client.get("/internal/window", params={"H_seconds": -1}, headers={"Authorization": "Bearer secret"})).status_code == 422
            result = await client.post("/internal/lifecycle/heartbeat", json={"workflow_id": "w", "agent_id": "a"}, headers={"Authorization": "Bearer secret"})
            assert result.json()["accepted"]
            record = next(iter((await store.read())["records"].values()))
            await store.prediction(record["generation"], {"mu": math.log(3), "sigma": .5, "model_version": "fixture",
                                                        "distribution_calibrated": True}, "ready")
            def processor(state):
                state["processor"] = stamp(2)
                return state, []
            await store.mutate(processor)
            notice = {"action": "candidate_update", "tenant": "tenant", "key": record["key"], "generation": record["generation"], "revision": 10,
                      "snapshot_utc": 1., "snapshot_stamp": stamp(1), "H_seconds": 2., "candidate": {"status": "ready", "key": record["key"], "generation": record["generation"],
                      "revision": 10, "age_seconds": 1., "expires_at_utc": 3., "valid_for_seconds": 2., "in_window": False,
                      "distribution": {"mu": math.log(3), "sigma": .5, "distribution_calibrated": True}}}
            await redis.xadd(store.notices, {"data": json.dumps(notice)})
            response = await client.get("/internal/notices", headers={"Authorization": "Bearer secret"})
            candidate = next(e["candidate"] for e in response.json()["events"] if e["action"] == "candidate_update")
            assert candidate["age_seconds"] == 2.
            assert candidate["remaining_p50_seconds"] == pytest.approx(1.4244, abs=.001)
            assert candidate["valid_for_seconds"] < 1. and candidate["in_window"]
            unverified = copy.deepcopy(notice)
            unverified["snapshot_stamp"].update(source="different-process", error=1.)
            await redis.xadd(store.notices, {"data": json.dumps(unverified)})
            response = await client.get("/internal/notices", headers={"Authorization": "Bearer secret"})
            unsafe = [e["candidate"] for e in response.json()["events"] if e["action"] == "candidate_update"][-1]
            assert unsafe["status"] == "unknown" and unsafe["reason"] == "clock_uncertain" and not unsafe["in_window"]
        await redis.aclose()
    asyncio.run(run())


@pytest.mark.parametrize("stream,fail", [(False, False), (True, False), (False, True)])
def test_middleware_send_boundaries_feed_real_reducer(stream, fail):
    from timepredict.sglang import Capture, ObservationMiddleware
    captured = []
    class PublisherFixture:
        class ClockFixture:
            def __init__(self): self.t = 0
            def stamp(self):
                self.t += 1
                return stamp(float(self.t))
        clock = ClockFixture()
        def emit(self, kind, payload, stamp=None):
            captured.append({**payload, "event_id": str(len(captured)), "kind": kind, "stamp": stamp or self.clock.stamp()})
    async def app(scope, receive, send):
        identity = {"tenant": "tenant", "workflow_id": "w", "agent_id": "a", "request_id": "r1"}
        capture = Capture(identity, {"search": {"description": "search documents", "parameters": {}}})
        capture.observe({"choices": [{"message": {"tool_calls": [{"id": "call1", "function": {"name": "search", "arguments": "{}"}}]}}]}, False)
        scope["state"]["timepredict_capture"] = capture
        await send({"type": "http.response.start", "status": 200,
                    "headers": [(b"content-type", b"text/event-stream" if stream else b"application/json")]})
        await send({"type": "http.response.body", "body": b"data: [DONE]\n\n" if stream else b"{}", "more_body": False})
    async def send(message):
        if fail and message["type"] == "http.response.body":
            raise ConnectionError("fixture send failed")
    async def run():
        middleware = ObservationMiddleware(app, publisher=PublisherFixture())
        scope = {"type": "http", "path": "/v1/chat/completions"}
        if fail:
            with pytest.raises(ConnectionError):
                await middleware(scope, None, send)
        else:
            await middleware(scope, None, send)
    asyncio.run(run())
    state = empty_state()
    for e in captured:
        state, _ = reduce_event(state, e)
    record = next(iter(state["records"].values()))
    assert record["status"] == ("delivery_failed" if fail else "waiting")
    if not fail:
        assert record["prediction_status"] == "pending_prediction"


def test_joint_semantic_experiment_preserves_actual_values_and_canonical_order(monkeypatch):
    from timepredict.semantic import joint_call_text
    monkeypatch.setenv("TIMEPREDICT_TEST_ENCODER", "1")
    a = {"name": "search", "description": "search text", "arguments": {"query": "alpha", "n": 10}}
    b = {**a, "arguments": '{"n":10,"query":"alpha"}'}
    c = {**a, "arguments": {"query": "omega", "n": 100}}
    assert joint_call_text(a) == joint_call_text(b) != joint_call_text(c)
    spec = FeatureSpec.fit([a], feature_mode="semantic")
    assert np.array_equal(spec.transform(a), spec.transform(b))
    assert not np.array_equal(spec.transform(a), spec.transform(c))


def test_periodic_training_stable_cutoff_and_lease(tmp_path, monkeypatch):
    fakeredis = pytest.importorskip("fakeredis.aioredis")
    pytest.importorskip("lupa")
    from redis.asyncio import Redis
    from timepredict.pipeline import retrain_once
    async def run():
        redis = fakeredis.FakeRedis(decode_responses=True)
        # retrain_once owns/ closes its connection; fakeredis allows reconnecting.
        monkeypatch.setattr(Redis, "from_url", lambda *a, **kw: redis)
        for i in range(24):
            wf, t, duration = f"wf{i}", 20.*i, 2.+i%5
            for e in (event("out", t+1, workflow=wf), event("ingress", t+1+duration, request="r2", workflow=wf, latest_tool_ids=["call1"]), event("terminate", t+2+duration, workflow=wf)):
                await redis.xadd("periodic:history", {"data": json.dumps(e)})
        args = SimpleNamespace(redis_url="redis://fixture", prefix="periodic", output=tmp_path,
                               serving_dir=None, evidence=None, policy=None, rounds=3, threads=1,
                               feature_mode="structured", holdout_tool=[])
        first = await retrain_once(args)
        candidate = tmp_path/"candidates"/first["version"]
        candidate.rename(tmp_path/"saved-candidate")
        recovered = await retrain_once(args)
        assert recovered["status"] == "candidate_trained" and candidate.is_dir()
        (candidate/"features.json").write_text("damaged package", encoding="utf-8")
        repaired = await retrain_once(args)
        assert repaired["status"] == "candidate_trained"
        assert Model(candidate).predict({"name": "search", "arguments": "{}"})[1] == "ready"
        assert list((tmp_path/"candidates").glob(".invalid-*"))
        manifest_path = candidate/"manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["version"] = "different-package"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        assert (await retrain_once(args))["status"] == "candidate_trained"
        assert Model(candidate).version == first["version"]
        second = await retrain_once(args)
        assert first["status"] == "candidate_trained" and second["status"] == "unchanged"
        assert first["version"] == second["version"]
        await redis.set("periodic:training-lease", "other")
        assert (await retrain_once(args))["reason"] == "trainer_already_running"
        args.prefix = "empty-history"
        args.output = tmp_path/"empty"
        assert (await retrain_once(args))["status"] == "waiting_for_data"
    asyncio.run(run())


def test_incremental_consumer_cannot_revive_removed_or_newer_generation():
    state, record = waiting()
    apply_prediction(state, record["generation"], {"mu": math.log(3), "sigma": .5, "model_version": "v",
                                                "distribution_calibrated": True}, "ready")
    view = window_rows(state, stamp(3), 2)
    row = view["records"][0]
    consumer = WindowConsumer(lambda: 0.)
    consumer.snapshot(view)
    assert consumer.candidates()
    remove = {"action": "remove", "key": row["key"], "generation": row["generation"], "revision": view["revision"]+2}
    consumer.notice(remove)
    consumer.notice({"action": "candidate_update", "key": row["key"], "generation": row["generation"],
                     "revision": view["revision"]+1, "H_seconds": 2, "candidate": row})
    assert not consumer.candidates()


def test_ablation_has_same_budget_and_reports_both_encoding_costs(tmp_path, monkeypatch):
    from timepredict.pipeline import ablation
    monkeypatch.setenv("TIMEPREDICT_TEST_ENCODER", "1")
    rows = [{"generation": f"g{i}", "workflow": f"w{i}", "workflow_start": float(i),
             "workflow_end": float(i)+.1, "call": {"name": "search", "description": "search text", "arguments": {"n": i%3+1}},
             "lower": 2.+i%3, "upper": 2.+i%3} for i in range(12)]
    report = ablation(rows, tmp_path, data_kind="synthetic", rounds=2, threads=1)
    a, b = report["models"]["structured"], report["models"]["semantic"]
    assert a["split_counts"] == b["split_counts"]
    assert b["uncached_inference_mean_seconds"] > 0 and b["inference_mean_seconds"] > 0
    assert not report["production_benefit_verified"]
