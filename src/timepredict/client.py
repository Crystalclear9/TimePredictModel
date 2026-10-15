"""Scheduler-neutral consumer. No KV mapping, density policy or loading decisions."""
import math
import time

from .candidates import CandidateCache
from .distribution import LogNormal, UnreliableDistribution


class WindowConsumer:
    """Enforce freshness locally and expose distributions to an external scheduler."""
    def __init__(self, monotonic=time.perf_counter):
        self.clock = monotonic
        self.cache = CandidateCache(monotonic)
        self.received = {}

    def snapshot(self, snapshot, roundtrip_seconds=0.):
        now = self.clock()
        if not math.isfinite(roundtrip_seconds) or roundtrip_seconds < 0:
            raise ValueError("invalid_roundtrip")
        if not self.cache.snapshot(snapshot, roundtrip_seconds):
            return
        self.received = {(r["key"], r["generation"]): (now, roundtrip_seconds, snapshot["H_seconds"])
                         for r in snapshot["records"]}

    def notice(self, item, roundtrip_seconds=0.):
        now = self.clock()
        if not math.isfinite(roundtrip_seconds) or roundtrip_seconds < 0:
            raise ValueError("invalid_roundtrip")
        identity = item["key"], item["generation"]
        before = self.cache.revisions.get(identity, 0)
        self.cache.notice(item, roundtrip_seconds)
        current = self.cache.records.get(item["key"])
        if (self.cache.revisions.get(identity, 0) > before and item["action"] == "candidate_update"
                and current and current[0]["generation"] == item["generation"] and current[0]["revision"] == item["revision"]):
            self.received[identity] = (now, roundtrip_seconds, item["H_seconds"])

    def candidates(self):
        """Cached mu/sigma are reconditioned locally; expired leases never remain actionable."""
        result = []
        active = set()
        for cached in self.cache.candidates():
            identity = cached["key"], cached["generation"]
            active.add(identity)
            at, rtt, horizon = self.received[identity]
            passed = max(0., self.clock()-at)+rtt
            row = dict(cached)
            row["age_seconds"] += passed
            row["valid_for_seconds"] = max(0., row["valid_for_seconds"]-passed)
            prediction = row.get("distribution") or {}
            if prediction.get("distribution_calibrated") is not True or prediction.get("shadow"):
                continue
            try:
                summary = LogNormal(prediction["mu"], prediction["sigma"]).summarize(row["age_seconds"], horizon)
            except (KeyError, ValueError, UnreliableDistribution):
                continue
            if summary["remaining_p50_seconds"] <= horizon and row["valid_for_seconds"] > 0:
                row.update(summary)
                result.append(row)
        self.received = {k: v for k, v in self.received.items() if k in active}
        return result


class PredictionClient:
    """Use a caller-owned httpx.AsyncClient so connection pooling and timeouts remain explicit."""
    def __init__(self, http_client, url, token, monotonic=time.perf_counter):
        self.http, self.url = http_client, url.rstrip("/")
        self.headers = {"Authorization": "Bearer "+token}
        self.clock = monotonic
        self.consumer = WindowConsumer(monotonic)

    async def refresh(self, horizon):
        started = self.clock()
        response = await self.http.get(self.url+"/internal/window", params={"H_seconds": horizon}, headers=self.headers)
        response.raise_for_status()
        snapshot = response.json()
        self.consumer.snapshot(snapshot, self.clock()-started)
        return self.consumer.candidates()

    def candidates(self):
        # Safe even during a network outage: TTL is enforced without another request.
        return self.consumer.candidates()

    async def lifecycle(self, action, workflow_id, agent_id="*"):
        if action not in ("heartbeat", "cancel", "terminate"):
            raise ValueError("invalid_lifecycle_action")
        response = await self.http.post(self.url+"/internal/lifecycle/"+action,
                                       json={"workflow_id": workflow_id, "agent_id": agent_id}, headers=self.headers)
        response.raise_for_status()
        return response.json()
