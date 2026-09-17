# 训练数据准备

## 什么数据可以训练

一条样本对应一次**真实工具发出到真实续接请求到达**的等待，不是一个孤立工具函数的执行时间，也不是 LLM 推理时长。

需要保留发出时的工具名、完整参数、工具 schema/描述快照，以及正确关联的耗时标签和 workflow 时间。标签统一为秒，时间戳统一为 UTC Unix 秒。工作流和 generation 只用于分组/去重，不进入模型特征。

ToolBench、API-Bank、BFCL 等工具调用语料可用于构造待回放任务，但通常没有本项目定义的端到端回返时间标签，不能直接作为有标签训练集。正确做法是把合法任务交给真实 Agent 和工具执行器，在目标或代表性服务器链路上回放，由 SGLang 适配层重新采集标签。不要使用模型猜测的耗时、固定 sleep 或公开请求示例中的假时间冒充真实数据。

## 推荐采集流程

1. 在 Linux 服务器部署已打补丁的 SGLang 和 Redis；预测服务可以先不加载模型，只做采集。
2. 将 `examples/agent_client.py` 的固定演示执行器换成真实工具。每次业务 workflow 使用独立 ID；请求重试复用 Request ID。每个 Agent 串行调用一个工具。
3. 在代表性负载下采集不同工具、参数长度/规模、成功慢调用及长尾。保留取消、终止、失联等事件，不把它们改写成已返回。
4. Agent 等待期间每 5 秒心跳，完成时 terminate。缺少终止会使跨时间分界的 workflow 被隔离；缺少心跳会失去删失样本的观察依据。
5. 等到待使用事件已经进入 history，再导出并固定截取时间。截取之后才被采集到的标签不会倒灌到该快照。保留导出时间、部署/工具版本、机器配置、负载和校时记录，便于解释数据分布。
6. 执行 snapshot、check-data，再进行训练。不要先删除尚未归档的 Redis 日志。

不同工具实现、后端配置或集群负载可能改变耗时分布。首版不使用机器 ID 或负载作为特征，优先为稳定的部署/工具配置分别建立数据版本和模型，不要把不相容的数据无标记地混在一起。

## 标准样本格式

推荐直接使用 `timepredict snapshot` 生成 JSONL。手工导入也必须符合相同格式。以下仅演示字段，**不是可直接训练的数据集**：

```json
{"generation":"unique-call-record","workflow":"tenant/workflow-1","workflow_start":1790000000.0,"workflow_end":1790000060.0,"call":{"name":"search_documents","arguments":{"query":"KV cache","max_results":10},"description":"Search documents","schema":{"type":"object","properties":{"query":{"type":"string"},"max_results":{"type":"integer"}}}},"lower":3.2,"upper":3.2}
```

| 字段 | 要求 |
|---|---|
| generation | 全数据集唯一、非空字符串，重复样本直接拒绝 |
| workflow | 包含租户隔离含义的非空工作流标识，同一 workflow 的起止元数据完全一致 |
| workflow_start / workflow_end | 有限 UTC Unix 秒；未结束的 end 为 null，已结束的 end≥start |
| call.name / arguments | 非空工具名；参数为 JSON 对象或能解析为对象的 JSON 字符串 |
| call.schema / description | 发出时快照；schema 为对象，缺省为 `{}`；描述不进入首版特征 |
| lower / upper | 已返回为 `[T,T]`，等待且完整观察到截取时刻为 `[a,null]`；T/a 必须为有限正数，单位秒 |

JSON 文件中禁止 NaN 和 Infinity；右删失只用 null 表达无限上界。首版不接受区间删失 `[L,U]`、左删失、毫秒未换算的标签或工具单独计时冒充完整回返时间。

参数超过 256 KiB、深度超过 32、解析失败或数值不能表达时不支持预测。初始特征不包含工具结果、后续 query、已等待时间或最终耗时。新字段只影响通用统计；词典只从训练集建立并冻结，最多 4096 个结构字段。

## 校验、划分、训练

```bash
uv run timepredict check-data --data samples.jsonl
uv run timepredict train --data samples.jsonl --output candidates \
  --threads 16 --rounds 500 --data-kind observed
uv run timepredict train --data samples.jsonl --output holdout-candidates \
  --threads 16 --holdout-tool reserved_tool --data-kind observed
```

`check-data` 不训练模型，检查格式、标签、去重及 workflow 元数据，输出工具数量、精确/删失样本数、各集合统计和数据 SHA-256。无法形成时间划分或训练/验证集没有真实返回时拒绝继续。`train` 本身也执行相同基础校验，不能跳过检查直接让错误格式进入 XGBoost。

按 workflow 开始时间划分约 70%/15%/15%，整个 workflow 不跨集合；跨分界仍在运行的 workflow 进入 quarantine。至少 7 个不同 workflow 开始时间只是代码下限，不代表数据量足够。门禁还检查支持样本数、有效样本量、窗口/分位数校准和基线比较；请按每种工具及长尾覆盖情况决定采集量。

取消、终止前未返回、失联、发送失败、并发冲突、采集缺口及不可信时钟排除训练。行政截取的右删失要求各入口观察正常、对应 Agent/来源心跳新鲜且时钟有界。不要只保留返回快的完整样本，否则会产生选择偏差。

整工具留出会将含该工具的整个 workflow 从普通集合移出，单独评估未见工具。在线遇到未知工具仍输出 unknown_tool，离线泛化报告不能自动改变这一行为。

报告包含删失负对数似然、IPCW 校准/覆盖率、有效样本量及全局/按工具基线；缺少支持的指标为 null。IPCW 假设行政删失独立，若人为按“等待很久”终止观察，不应直接当作独立删失。通过训练不等于可以发布，还需 README 中的影子运行和服务器门禁。

## 当前状态

本次完成了数据入口和训练链路复核，清理前在 WSL 运行 70 项测试并通过真实 Redis 链路验证。未保留合成训练包，也未下载没有匹配耗时标签的公开语料。后续应首先确定真实工具清单、执行器、目标部署和代表性任务来源，再采集训练集。
