"""C2-side lease cache: expiry is enforced even when no new messages arrive."""
import time


class CandidateCache:
    def __init__(self, monotonic=time.monotonic):
        self.clock = monotonic
        self.records = {}
        self.floor = 0
        self.revisions = {}
        self.key_revisions = {}
        self.high_water = 0

    def _put(self, row, roundtrip_seconds):
        ttl = max(0., row.get("valid_for_seconds", 0.)-roundtrip_seconds)
        if row.get("status") != "ready" or not row.get("in_window") or ttl <= 0:
            self.records.pop(row["key"], None)
        else:
            self.records[row["key"]] = (dict(row), self.clock()+ttl)

    def snapshot(self, snapshot, roundtrip_seconds=0.):
        if snapshot["revision"] < self.high_water:
            return
        self.high_water = snapshot["revision"]
        self.records.clear()
        self.revisions.clear()
        self.key_revisions.clear()
        self.floor = snapshot["revision"]
        for row in snapshot["records"]:
            self.revisions[(row["key"], row["generation"])] = row["revision"]
            self.key_revisions[row["key"]] = row["revision"]
            self._put(row, roundtrip_seconds)

    def notice(self, item, roundtrip_seconds=0.):
        identity = (item["key"], item["generation"])
        if item["revision"] <= max(self.floor, self.revisions.get(identity, 0)):
            return
        self.revisions[identity] = item["revision"]
        self.high_water = max(self.high_water, item["revision"])
        if item["action"] == "remove":
            current = self.records.get(item["key"])
            if current and current[0]["generation"] == item["generation"]:
                del self.records[item["key"]]
        elif item["action"] == "candidate_update":
            if item["revision"] <= self.key_revisions.get(item["key"], 0):
                return
            self.key_revisions[item["key"]] = item["revision"]
            self._put(item["candidate"], roundtrip_seconds)

    def candidates(self):
        now = self.clock()
        for key, (_, expires) in list(self.records.items()):
            if expires <= now:
                del self.records[key]
        return [row for row, _ in self.records.values()]
