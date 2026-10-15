# 文件作用：候选消费端缓存：利用 generation/revision 拒绝旧通知，用单调时间执行本地过期。
"""C2-side lease cache: expiry is enforced even when no new messages arrive."""
import copy
import time


# 保存可执行候选、业务键版本及全量快照版本下界。
class CandidateCache:
    # 初始化版本索引；允许注入单调时钟以便确定性验证。
    def __init__(self, monotonic=time.perf_counter):
        self.clock = monotonic
        self.records = {}
        self.floor = 0
        self.key_revisions = {}
        self.high_water = 0
        self.snapshot_stamp = None

    # 从有效期扣除网络往返时间；仅缓存 ready 且已入窗的记录。
    def _put(self, row, roundtrip_seconds, received_at):
        ttl = max(0., row.get("valid_for_seconds", 0.)-roundtrip_seconds)
        if row.get("status") != "ready" or not row.get("in_window") or ttl <= 0:
            self.records.pop(row["key"], None)
        else:
            self.records[row["key"]] = (copy.deepcopy(row), received_at+ttl)

    # 接受足够新的全量快照，重建缓存并设置旧通知过滤下界。
    def snapshot(self, snapshot, roundtrip_seconds=0.):
        received_at = self.clock()
        if snapshot["revision"] < self.high_water:
            return False
        stamp = snapshot.get("snapshot_utc")
        if self.snapshot_stamp is not None and stamp is not None and snapshot["revision"] == self.floor:
            if stamp <= self.snapshot_stamp:
                return False
        if stamp is None and snapshot["revision"] == self.floor and self.snapshot_stamp is not None:
            return False
        self.snapshot_stamp = stamp if stamp is not None else float("-inf")
        self.high_water = snapshot["revision"]
        self.records.clear()
        self.key_revisions.clear()
        self.floor = snapshot["revision"]
        for row in snapshot["records"]:
            self.key_revisions[row["key"]] = row["revision"]
            self._put(row, roundtrip_seconds, received_at)
        return True

    # 业务键版本过滤更新；移除只作用于同轮且更旧的候选。
    def notice(self, item, roundtrip_seconds=0.):
        """Return whether cached candidate content changed, never renew a replayed lease."""
        received_at = self.clock()
        revision, key = item["revision"], item["key"]
        if revision <= self.floor:
            return False
        previous = self.key_revisions.get(key, 0)
        self.key_revisions[key] = max(previous, revision)
        self.high_water = max(self.high_water, revision)
        if item["action"] == "remove":
            # A newer round's tombstone also fences delayed updates of older rounds.
            # Still apply a matching removal below another round's fence: that
            # other tombstone deliberately did not remove this cached generation.
            current = self.records.get(key)
            if current and current[0]["generation"] == item["generation"] and revision > current[0]["revision"]:
                del self.records[key]
                return True
        elif item["action"] == "candidate_update" and revision > previous:
            self._put(item["candidate"], roundtrip_seconds, received_at)
            return True
        return False

    # 主动清除过期租约后返回当前候选，即使没有新通知也会失效。
    def candidates(self):
        now = self.clock()
        for key, (_, expires) in list(self.records.items()):
            if expires <= now:
                del self.records[key]
        return [copy.deepcopy(row) for row, _ in self.records.values()]
