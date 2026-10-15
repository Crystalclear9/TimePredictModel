# 时间预测插件接入契约（API v1）

## 责任边界

TimePredictModel 提供时间信息。外部模块持有 KV 节点、layer、内存预算、传输时间和收益，负责密度评分及预加载。两侧通过 `(tenant, workflow_id, agent_id)` 关联；`generation` 区分调用轮次，`revision` 拒绝迟到更新。业务身份不输入预测模型。

## HTTP 快照

请求 `GET /internal/window?H_seconds=2`，携带 `Authorization: Bearer INTERNAL_TOKEN`。所有时间为秒，`snapshot_utc` 是 Unix UTC。下面是格式示意，数值不是测量结果：

服务启动时验证令牌映射：键必须是非空 ASCII 字符串，租户值必须是非空字符串。null 或错误类型不会被解释为“全部租户”。服务保存已验证映射的副本，修改调用方原字典不会更改已有服务的绑定。非 ASCII 认证头返回 401。

```json
{
  "api_version": 1,
  "time_unit": "seconds",
  "snapshot_utc": 1000.0,
  "revision": 15,
  "H_seconds": 2.0,
  "notice_cursor": "1000-0",
  "records": [{
    "key": "[\"tenant\",\"workflow\",\"agent\"]",
    "tenant": "tenant",
    "workflow_id": "workflow",
    "agent_id": "agent",
    "generation": "generation-hash",
    "revision": 15,
    "status": "ready",
    "in_window": true,
    "shadow": false,
    "age_seconds": 2.0,
    "remaining_p50_seconds": 1.4244,
    "remaining_p90_seconds": 4.0744,
    "prob_return_within_H": 0.6430,
    "expected_remaining_seconds": 1.8882,
    "expected_capped_remaining_seconds": 1.2973,
    "valid_for_seconds": 1.0,
    "expires_at_utc": 1001.0,
    "model_version": "model-version",
    "distribution": {
      "mu": 1.0986122886681098,
      "sigma": 0.5,
      "distribution_kind": "lognormal",
      "time_unit": "seconds",
      "feature_version": 4,
      "feature_mode": "structured",
      "model_version": "model-version",
      "distribution_calibrated": true
    }
  }]
}
```

`mu` 是总回返时间 **log 秒** 的位置参数，不是剩余时间，也不是时间均值。初始中位数 `exp(mu)`，初始均值 `exp(mu+sigma²/2)`。更新只使用 μ、σ 和当前 age，无须再次推理。

入窗规则为剩余 P50 ≤ H；`prob_return_within_H` 是估计概率，不保证实际回返。`expected_capped_remaining_seconds` 为 `E[min(R,H) | T>age]`，适合有 H 秒驻留上限的成本模型；其他上限应从缓存分布重新计算。不能以 P50 替代期望成本。

消费方在响应抵达时扣除实测往返时间（保守扣减），以本地单调时钟倒计时。到期立即撤销候选，即使没有通知、网络断开或服务不可用。`expires_at_utc` 供诊断，不能在没有校时证据时仅依赖它。

SDK 在开始处理每个快照或通知时记录接收时间。建立整批缓存的耗时会继续消耗有效期，并计入 age；处理结束不会重新开始租约。若建缓存已耗尽有效期，候选不会输出。

## Python 接入

```python
import httpx
from timepredict.client import PredictionClient
from timepredict.distribution import LogNormal

async with httpx.AsyncClient(timeout=1) as http:
    predictor = PredictionClient(http, "http://predictor:8098", internal_token)
    candidates = await predictor.refresh(horizon=2)
    for candidate in candidates:
        parameters = candidate["distribution"]
        cost = LogNormal(parameters["mu"], parameters["sigma"]).summarize(
            candidate["age_seconds"], horizon=5
        )["expected_capped_remaining_seconds"]
        # existing_c2.update(candidate, expected_residence_seconds=cost)
```

这段示例省略了外部 KV 映射和密度计算。`PredictionClient.candidates()` 会更新 age 和条件分布，并执行本地过期检查。客户端收到异常时保留上次仍有效的候选，过期后返回空集合。

## 增量通知

先取得快照及 `notice_cursor`，再读取 `/internal/notices?after=CURSOR`。响应游标推进到最后读取事件，即使事件属于其他租户；只返回令牌所属租户内容。

游标使用服务返回的标准 ASCII 数字格式；非法或越界游标返回422，通知流或共享状态读取失败返回503。服务器在读取通知页后核对共享状态：轮次已返回、取消、被替换或清理时，历史候选降级为 `unknown/round_no_longer_pending`、`in_window=false`、有效期为零，即使对应 remove 尚在下一页。当前观察租约失效时也会降级。

- `candidate_update`：附带候选、`generation`、`revision`、`H_seconds`。
- `remove`：关闭、取消、观察失败等状态移除该 generation 的候选。
- `update`：内部状态变更提示；当前窗口是否可用仍由全量快照或后续 `candidate_update` 确认。

可用 `WindowConsumer.snapshot()`、`WindowConsumer.notice()`、`WindowConsumer.candidates()` 管理本地缓存。旧 revision、已结束 generation 和过期通知不能复活候选。周期全量快照用于恢复遗漏的通知。不要以时间戳或通知到达顺序替代版本检查。

SDK 对输入行和输出行保留独立副本；接入方修改返回的嵌套分布或附加调度信息不会污染内部缓存。服务器检查反映读取时的状态，响应之后仍可能发生回返，因此本地版本、通知和过期检查仍须执行。

较新revision的全量快照优先于不同API副本间的微小UTC偏差；同一revision则拒绝相同或更旧的快照时间，防止重复快照延长租约。采集端迟到的更早回返证据可以修正同一generation的训练标签，不创建额外轮次样本。

## unknown、影子和生命周期

`pending_prediction` 表示尚未计算；`unknown` 附带 `reason`，例如模型不存在、未校准、参数不支持、时钟不可靠、来源丢失或极端尾部。消费方均不主动加载。影子行可含预测统计，但 `in_window=false`，不能参与实际加载。

`observation_gap`、`observation_lost`、`source_restart` 也可能由其他采集入口的故障引起，因为续接不保证回到原入口。故障影响区间未排除时，有效期为零；入口恢复也不自动使区间内的旧调用变为可执行。接入方应遵守 unknown 和版本化移除，不能仅凭 out 所在入口仍存活而继续加载。

heartbeat/cancel 使用 JSON `{"workflow_id":"w","agent_id":"a"}`；terminate 可以省略 agent_id 结束整个工作流。`accepted=true` 只表示生命周期事件已入流，不是状态已经归约完成。

同一 `(tenant, workflow_id, agent_id)` 下必须遵守串行单工具调用。需要并发 Agent 分支时分配不同的 agent_id；并发多工具轮次不在 v1 预测范围内。

## 接入方验收

验证身份到 KV 对象的映射、候选移除与租约过期后的撤销、缓存预算及密度成本单位，并比较使用/不使用预测的真实加载命中率、GPU 等待、额外传输、byte-seconds 和总体吞吐。将编码、树预测和接口开销一起计入。插件单元测试或公开代理数据不能替代这些实验。
