# 训练数据准备

## 标签和输入

目标为 `T = t_return - t_out`：完整工具响应的服务端发送边界到后续请求的服务端入口。工具函数执行时间、LLM 推理时间和续接排队都不能替代这个标签。标签单位为秒，时间戳为 UTC Unix 秒。

只使用 t_out 时已经存在的工具名、描述、schema 和实际参数。工作流、Agent、请求 ID、generation 只用于关联、分组和幂等；后续消息、结果和真实耗时不进入特征。

每个 `(tenant, workflow_id, agent_id)` 一轮一个工具，最多一个待回返。并发分支使用不同 agent_id。请求重试复用 request_id。工具 ID 存在时必须匹配；没有工具 ID 时要求双方声明串行协议。

## 真实采集

1. 在目标 Linux 环境应用固定 SGLang 补丁，配置持久化 Redis。没有模型也可采集，接口返回 unknown。
2. 将 [Agent 示例](../examples/agent_client.py) 的演示执行器替换为真实工具，在代表性任务、工具规模和负载下运行。
3. 等待期间每五秒 heartbeat，结束 terminate，取消 cancel。保留失败、失联和长尾事件，不能改写成正常返回。
4. 确认事件已写入 history，再固定导出边界和截断时刻。跨主机运行需要真实校时证据。

公开工具调用语料通常没有本项目的端到端时间标签，可用于构造真实回放任务，不能直接冒充有标签数据。固定 sleep 和模拟验收夹具仅验证流程。当前仓库不再提供公开代理数据下载或旧合成回放入口。

## 导出、快照和检查

```bash
uv run timepredict export --output events.jsonl
uv run timepredict snapshot --events events.jsonl --cutoff UNIX_UTC_SECONDS --output samples.jsonl
uv run timepredict check-data --data samples.jsonl
uv run timepredict train --data samples.jsonl --data-kind observed --output candidates
```

JSONL 和模型包使用 UTF-8。输入 JSONL 可带 UTF-8 BOM；非标准 NaN/Infinity 和错误行会被拒绝，并报告行号。精确标签为 `[T,T]`，右删失上界在 JSON 中写为 `null`，内部恢复成正无穷。

正式样本由 snapshot 生成，携带原始 out/ingress 事件、身份、边界、关联证据和 SHA256。训练会重新核对这些信息。摘要用于一致性检查，数据来源仍必须可信。

工作流开始时间不能晚于已观察边界，结束时间不能明显早于 OUT、返回或 cutoff。同一 workflow 的元数据必须一致。跨时钟终止事件允许现有5ms预算内的UTC偏差，但划分时仍按整个工作流最晚观察时间扩展结束边界，不用该容差放行跨分界标签。

## 右删失与排除

行政截断可以记录“截至 cutoff 仍未返回”，标签为 `[cutoff-t_out, null]`。它要求来源与 Agent 心跳新鲜，来源/启动身份匹配，观察没有缺口，out 与 cutoff 的时钟误差均不超过 0.0025 秒。普通 OUT 事件或其他来源的心跳不能替代匹配的 source_heartbeat。

正式右删失样本要求 `workflow_end=null`。工作流已经结束的未返回调用属于生命周期排除，不能通过填写 cutoff 改成行政删失。

手动生成删失样本时附加 `--cutoff-clock-error MEASURED_SECONDS`；这个值必须来自实际测量。缺少 cutoff 证据时仍可导出完整返回，等待样本记为 `cutoff_clock_unverified` 并排除。周期重训自动读取本机校时状态。

取消、终止前未返回、发送失败、并发冲突、采集缺口、来源丢失和不可信时钟均单独排除，不当作行政删失。IPCW 评价假设删失独立；按“等得太久”主动停止的样本不能直接套用该假设。不要只采集快速完成的调用。

续接可能到达任意入口，因此一个入口的缺口也影响其他入口发出的调用。快照排除与缺口时间区间重叠、无法证明观察完整的标签，包含可能错配到后续重复历史的完整返回。跨来源时间界限同时计入两端时钟误差；更早的可信回返证据可以纠正临时关联并解除排除。

## 划分和评价

数据按完整 workflow 和开始时间约 70%/15%/15% 划分。跨边界或尚未结束的早期 workflow 进入 quarantine；最晚测试区间可包含尚未结束的 workflow。至少七个不同开始时间只是程序下限，不代表样本量足够。训练和验证必须有完整返回。

`--holdout-tool NAME` 将含该工具的整个 workflow 从普通训练/验证/测试中移除，单独评估未见工具。默认结构化模型保留未知工具列，可能输出 `unknown_tool=true` 的预测；这不保证未知工具上的精度。是否可执行仍取决于校准、观察新鲜度、影子状态和窗口规则。

报告包含 AFT 负对数似然、IPCW Brier/窗口校准/分位数覆盖、样本支持、条件窗口以及全局和按工具经验基线。不受支持的指标返回不可用状态，不能当作零误差。训练只生成候选，正式发布还需要 [README](../README.md) 中的服务器验收和质量门禁。
