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

条件分布计算同样消耗租约。SDK在返回整批结果前再次检查所有候选，包括先计算、随后在处理其他行时已过期的候选；返回的有效期已扣除这段处理耗时。age及分布统计对应本次计算时点，后续密度计算或排队耗时仍需由接入方扣除；实际加载前重新检查当前候选，不能长期保存返回列表而不执行过期和版本检查。客户端使用完整实测往返时间推进age并保守扣减TTL，网络延迟会引入时间估计偏差，需要在目标网络下评价。

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

`source_identity_conflict` 表示同一逻辑入口的不同 boot 无法按可信时钟区间排序，相关候选同样不可执行。需要排查重复的入口 ID 或校时状态，并让顺序可证明的替代入口恢复采集。

Redis 的 history/notices 键必须为 Stream。存储层在事务前检查类型，并同时 WATCH 两个流键，防止键类型异常时只写状态、漏写历史或通知。类型错误会阻止提交，事件保持未确认以便修复后重试。修复错误键前应核对既有数据及归档，不能把删除历史流当作通用恢复方法。旧版已经发生的部分写入需从保留的事件流/归档重建，升级不会自动补回缺失历史。此检查增加了Redis往返，需在目标环境重新测量P99和吞吐。

heartbeat/cancel 使用 JSON `{"workflow_id":"w","agent_id":"a"}`；terminate 可以省略 agent_id 结束整个工作流。`accepted=true` 只表示生命周期事件已入流，不是状态已经归约完成。

同一 `(tenant, workflow_id, agent_id)` 下必须遵守串行单工具调用。需要并发 Agent 分支时分配不同的 agent_id；并发多工具轮次不在 v1 预测范围内。

## 后续开发交接

当前交付范围是独立时间预测模块，可在以下边界上继续开发：

| 接入内容 | 使用入口 | 必须保持的约定 |
|---|---|---|
| Agent请求与工具执行 | 固定SGLang补丁、[Agent示例](../examples/agent_client.py) | 单工具串行；重试复用request_id；截断响应不执行工具；等待heartbeat、结束terminate |
| 外部C2/KV调度 | PredictionClient / WindowConsumer、[调度示例](../examples/scheduler_client.py) | 用tenant/workflow/agent/generation映射自有KV对象；处理移除和过期；密度公式、资源预算和加载由接入方实现 |
| 时间成本 | LogNormal(mu,sigma).summarize(age,horizon) | mu为log秒；按剩余分布计算驻留成本；保留模型版本与时间单位 |
| 数据与模型迭代 | snapshot / check-data / train / ablation / promote / retrain命令 | 正式事件证据、工作流时间隔离、独立验证/工具留出以及质量门禁 |

外部模块通过API v1或客户端接入；Redis内部状态及reducer由本插件维护。修改API字段、单位、候选判定或特征列/语义文本规则时，应显式升级对应契约，补充兼容或拒绝旧包的测试，重新训练并验收，不能只替换模型或词典中的一个文件。

开发与验收按以下顺序推进：

1. 运行 `uv run --extra test python scripts/verify_plugin.py`。协议与故障用例提供可复现的本地开发基线；示例的HTTP响应是模拟夹具，未调用真实工具或GPU。
2. 在隔离的真实部署中验证Agent、认证、跨入口关联、时钟、Redis持久化/接管和P99/吞吐。固定版本补丁检查只能验证应用与语法。
3. 用代表性真实日志训练/留出评价；先运行影子候选，取得对应版本的真实服务器证据，再通过发布门禁切换正式模型。
4. 接入自有C2/KV映射与加载策略，测量内存、传输、误加载和最终收益。这些结果决定系统是否达到上线条件。

当前本地通过支持接口开发与联调，不代表真实语义泛化、线上校准、GPU运行或KV收益已被证明；未完成项持续记录在[验证文档](verification.md)。

## 接入方验收

验证身份到 KV 对象的映射、候选移除与租约过期后的撤销、缓存预算及密度成本单位，并比较使用/不使用预测的真实加载命中率、GPU 等待、额外传输、byte-seconds 和总体吞吐。将编码、树预测和接口开销一起计入。插件单元测试或公开代理数据不能替代这些实验。
