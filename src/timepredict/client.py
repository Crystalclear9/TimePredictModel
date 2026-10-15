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
        self.resync_required = False

    def snapshot(self, snapshot, roundtrip_seconds=0.0):
        now = self.clock()
        if not math.isfinite(roundtrip_seconds) or roundtrip_seconds < 0:
            raise ValueError("invalid_roundtrip")
        if not self.cache.snapshot(snapshot, roundtrip_seconds):
            return False
        self.received = {
            key: (now, roundtrip_seconds, snapshot["H_seconds"]) for key in self.cache.records
        }
        self.resync_required = False
        return True

    def notice(self, item, roundtrip_seconds=0.0):
        if self.resync_required:
            return
        now = self.clock()
        if not math.isfinite(roundtrip_seconds) or roundtrip_seconds < 0:
            raise ValueError("invalid_roundtrip")
        if not self.cache.notice(item, roundtrip_seconds):
            return
        key = item["key"]
        if key in self.cache.records:
            self.received[key] = (now, roundtrip_seconds, item["H_seconds"])
        else:
            self.received.pop(key, None)

    def notice_page(self, page, roundtrip_seconds=0.0):
        """A trimmed cursor invalidates local candidates until a fresh snapshot."""
        if page.get("resync_required"):
            self.resync_required = True
            self.cache.records.clear()
            self.received.clear()
        else:
            for item in page["events"]:
                self.notice(item, roundtrip_seconds)

    def candidates(self):
        """Cached mu/sigma are reconditioned locally; expired leases never remain actionable."""
        result = []
        active = set()
        for cached in self.cache.candidates():
            key = cached["key"]
            active.add(key)
            at, rtt, horizon = self.received[key]
            passed = max(0.0, self.clock() - at) + rtt
            row = dict(cached)
            row["age_seconds"] += passed
            row["valid_for_seconds"] = max(0.0, row["valid_for_seconds"] - passed)
            prediction = row.get("distribution") or {}
            if prediction.get("distribution_calibrated") is not True or prediction.get("shadow"):
                continue
            try:
                summary = LogNormal(prediction["mu"], prediction["sigma"]).summarize(
                    row["age_seconds"], horizon
                )
            except (KeyError, ValueError, UnreliableDistribution):
                continue
            if summary["remaining_p50_seconds"] <= horizon and row["valid_for_seconds"] > 0:
                row.update(summary)
                result.append((row, at + cached["valid_for_seconds"] - rtt))
        self.received = {k: v for k, v in self.received.items() if k in active}
        # Computing this or later rows consumes the same lease. Check the
        # whole batch again at return, including rows computed earlier.
        completed_at = self.clock()
        current = []
        for row, expires_at in result:
            left = expires_at - completed_at
            if left > 0:
                row["valid_for_seconds"] = min(row["valid_for_seconds"], left)
                current.append(row)
        return current


class PredictionClient:
    """Use a caller-owned httpx.AsyncClient so connection pooling and timeouts remain explicit."""

    def __init__(self, http_client, url, token, monotonic=time.perf_counter):
        self.http, self.url = http_client, url.rstrip("/")
        self.headers = {"Authorization": "Bearer " + token}
        self.clock = monotonic
        self.consumer = WindowConsumer(monotonic)
        self.notice_cursor = None
        self.horizon = None

    async def refresh(self, horizon):
        started = self.clock()
        response = await self.http.get(
            self.url + "/internal/window", params={"H_seconds": horizon}, headers=self.headers
        )
        response.raise_for_status()
        snapshot = response.json()
        if self.consumer.snapshot(snapshot, self.clock() - started):
            self.notice_cursor = snapshot.get("notice_cursor")
            self.horizon = horizon
        return self.consumer.candidates()

    async def poll_notices(self):
        if self.notice_cursor is None:
            raise ValueError("refresh a snapshot before polling notices")
        started = self.clock()
        response = await self.http.get(
            self.url + "/internal/notices",
            params={"after": self.notice_cursor},
            headers=self.headers,
        )
        response.raise_for_status()
        page = response.json()
        self.consumer.notice_page(page, self.clock() - started)
        if self.consumer.resync_required:
            return await self.refresh(self.horizon)
        self.notice_cursor = page["cursor"]
        return self.consumer.candidates()

    def candidates(self):
        # Safe even during a network outage: TTL is enforced without another request.
        return self.consumer.candidates()

    async def lifecycle(self, action, workflow_id, agent_id="*"):
        if action not in ("heartbeat", "cancel", "terminate"):
            raise ValueError("invalid_lifecycle_action")
        response = await self.http.post(
            self.url + "/internal/lifecycle/" + action,
            json={"workflow_id": workflow_id, "agent_id": agent_id},
            headers=self.headers,
        )
        response.raise_for_status()
        return response.json()
