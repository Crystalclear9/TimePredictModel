# 文件作用：读接口独立检查观察新鲜度；即使 worker 未发送移除通知，也不能继续使用过期候选。
"""Read-side liveness checks independent of the elected worker."""
import math
from .clock import elapsed

PROCESSOR_TTL = 2.0
OBSERVER_TTL = 30.0


# 用可信时间差从租约寿命中扣除已过去时间。
def remaining_ttl(stamp, now, ttl):
    return ttl - elapsed(stamp, now)


# 综合处理器 2 秒、来源/Agent 30 秒租约；返回降级原因及最短剩余有效期。
def observation_status(state, record, now):
    processor = state.get("processor")
    if not processor:
        return "processor_unavailable", 0.
    try:
        left = remaining_ttl(processor, now, PROCESSOR_TTL)
        if left <= 0:
            return "processor_stale", 0.
        source = state["sources"].get(record["out"]["source"] + "/" + record["out"]["boot"])
        if source and (source.get("bad") or source.get("lost")):
            return "observation_lost", 0.
        for source_state in state["sources"].values():
            if not source_state.get("retired") and not source_state.get("lost"):
                source_remaining = remaining_ttl(source_state["last"], now, OBSERVER_TTL)
                if source_remaining <= 0:
                    return "source_stale", 0.
                left = min(left, source_remaining)
        source_left = remaining_ttl(source["last"] if source else record["out"], now, OBSERVER_TTL)
        agent_stamp = state.get("heartbeat_stamps", {}).get(record["key"], record["out"])
        agent_left = remaining_ttl(agent_stamp, now, OBSERVER_TTL)
        left = min(left, source_left, agent_left)
        if left <= 0:
            return "observation_stale", 0.
        return None, max(0., left)
    except ValueError as exc:
        return str(exc), 0.
