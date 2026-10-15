# TimePredictModel

独立的工具回返时间预测插件。目标是 `T = t_return - t_out`：完整工具响应的服务端发送边界到后续请求的服务端入口，单位为秒，不包含续接排队或推理。

本模块负责采集、关联、特征转换、AFT 训练、模型验收发布、条件剩余时间和窗口接口。外部模块负责 C2 密度评分、KV 节点映射、加载和收益评价。接入契约见 [INTEGRATION.md](INTEGRATION.md)。

## 实现约定

- 每个 `(tenant, workflow_id, agent_id)` 最多一轮待回返，每轮一个工具，`n=1`、`parallel_tool_calls=false`。优先核对已有 tool_call_id；缺少 ID 时依赖已声明且无冲突的串行协议。request_id 只负责重试幂等及轮次标识，身份不输入模型。
- 默认结构化基线：工具名 one-hot、未知工具列、参数结构统计、按工具名和 JSON Pointer 保留的数字/布尔/长度/枚举列。缺失为 NaN，字段词典只在训练集拟合，发布时冻结。新参数仍影响通用统计，重训后可增加独立列。
- 对象键排序、数组保留顺序，数字不混合求和；字符串长度不等于文件大小，超时配置不当作执行标签。
- `--feature-mode semantic` 为独立增强实验，联合编码工具名、描述、实际参数，不自动取代结构化基线。语义模式无需人为工具分类器。旧 action_family 模型仅兼容加载。
- XGBoost `survival:aft` / `normal`，深度4、学习率0.05、默认500轮、验证损失30轮不改善停止。统一 sigma 在验证集选择；读取原始 margin，恢复为 log 秒的 mu。
- 精确标签 `[T,T]`；真实观察截断 `[a,+∞)`，保存 cutoff 而不伪造 t_return。取消、终止、发送失败、并发冲突、来源丢失分别排除。
- 按时间及完整工作流划分；跨边界工作流隔离。`--holdout-tool` 将相关整个工作流从训练/验证/常规测试移除，专门评估未知工具。

`log(T)=mu+sigma*Z, Z~N(0,1)`。条件窗口概率为 `(F(a+H)-F(a))/(1-F(a))`。剩余P50/P90、均值及 `E[min(R,H)]` 都从条件分布计算，等待期间不重新编码或运行树。生存概率低于1e-12时拒绝极端尾部外推。

## 安装和测试

Python 3.11+：

```bash
uv sync --extra test
uv run pytest -q
```

基线不需要 PyTorch 或嵌入权重。语义实验使用 `uv sync --extra semantic`，提前缓存 `sentence-transformers/all-MiniLM-L6-v2`，revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`；可设置 `TIMEPREDICT_EMBEDDING_CACHE`。在线只读本地模型，不临时下载。

单元测试的哈希编码器和模拟验收数据只用于明确的夹具，不是正式来源或服务器验收证据。真实编码器、Redis、GPU SGLang、跨主机校时及性能另行验证。

## 采集部署

1. 在 SGLang v0.5.19 应用 `patches/sglang-v0.5.19.patch` 并安装本包。GPU进程只采集事件，不执行预测。
2. 配置持久化 Redis，设置 `TIMEPREDICT_REDIS_URL`、可选 `TIMEPREDICT_PREFIX`。
3. SGLang 启用 API key，每副本一个 HTTP/tokenizer worker，可多副本。Agent 携带 X-Workflow-ID、X-Agent-ID、X-Request-ID；重试复用 request_id。
4. Agent 等待期间每5秒 heartbeat，结束上报 terminate，取消上报 cancel。见 `examples/agent_client.py`。
5. 内部 tokens.json 为 `{"internal-token":"sha256-of-sglang-api-key"}`。租户命名空间来自已认证 SGLang key 的摘要。令牌文件不要提交仓库。

```bash
uv run timepredict serve --window-h 2 --tokens-file /etc/timepredict/tokens.json --models models
```

无模型时返回 unknown，继续采集。跨主机必须有校时证据；时钟不可信时降级。正式服务模板见 `deploy/timepredict.service`。事件经 Redis Streams 顺序、幂等处理，状态、通知和首次历史事件原子提交。生命周期为 prepared→waiting→returned，迟到预测不能复活已闭合记录。

## 快照、训练、验收发布

```bash
uv run timepredict export --output events.jsonl
uv run timepredict snapshot --events events.jsonl --cutoff UNIX_UTC_SECONDS --output samples.jsonl
uv run timepredict train --data samples.jsonl --data-kind observed --feature-mode structured --output candidates
```

正式样本携带关联方式、边界、原始事件及摘要。训练验证事件来源、调用内容、业务身份和真实时间差。数据需来自可信服务端；摘要用于一致性检查，不是来源认证签名。

生成右删失样本时，截断时钟也必须有校时证据。手动快照可提供 `--cutoff-clock-error MEASURED_SECONDS`（误差范围0至0.0025秒，必须来自实际测量）；未提供时只保留完整返回，等待样本记为 `cutoff_clock_unverified` 并排除。周期重训自动读取本机校时状态；无校时证据时同样不制造删失标签。

训练输出模型、词典、sigma、最佳迭代、依赖版本、数据摘要、覆盖率、窗口/条件窗口校准、经验基线和曲线。训练不更新线上 CURRENT。

```bash
uv run timepredict serve --window-h 2 --tokens-file /etc/timepredict/tokens.json --shadow-candidate candidates/VERSION
```

影子模型提供分布供评价，始终 in_window=false。使用 `deploy/benchmark_server.py` 和 `deploy/verify_failover.py` 在隔离 Linux/GPU 部署生成验收证据；固定延时验收负载不替代真实日志上的精度实验。

```bash
uv run timepredict promote --candidate candidates/VERSION --serving-dir models --evidence server-acceptance.json --policy deploy/gate-policy.json
```

正式标签、校准/基线/样本支持及对应版本的服务器验收全部通过，才原子更新 CURRENT。加载校验包摘要、特征契约和归档证据，之后才设置 distribution_calibrated=true。等待中的调用保留原预测版本。证据过期后不再对新调用发出可用预测。P99、吞吐和校准门槛是验收目标，不是已测得结果。

## 周期重训和对照实验

```bash
uv run timepredict retrain --output training --feature-mode structured
uv run timepredict ablation --data samples.jsonl --data-kind observed --output ablation --holdout-tool unseen_tool
```

retrain 固定历史流边界及UTC截断，在独立进程训练候选；可续租的 Redis 租约避免重叠训练，数据和配置不变则复用候选。`deploy/timepredict-train.service`、`deploy/timepredict-train.timer` 提供每小时模板。

要验收后自动发布，同时提供 --serving-dir、--evidence、--policy；证据需匹配新版本。失败时保留线上模型并写入 training/last-run.json。模板默认仅训练候选。

ablation 共享数据、划分和训练预算，保存基线与增强的已知/未知工具质量，以及编码+树预测的缓存/非缓存开销。实际投机收益由外部模块实验决定。

## 外部模块接口

| 接口 | 内容 |
|---|---|
| GET /internal/contract | API版本、单位、分布和责任约定 |
| GET /internal/window?H_seconds=2 | 全量候选、通知游标、有效期 |
| GET /internal/notices?after=CURSOR | 增量更新和移除通知 |
| POST /internal/lifecycle/heartbeat | Agent等待存活证据 |
| POST /internal/lifecycle/cancel | 取消Agent |
| POST /internal/lifecycle/terminate | 工作流结束 |
| GET /internal/health | 时钟、来源、模型、开销 |

接口使用内部 Bearer token，按租户隔离。仅使用 ready、in_window、非shadow且未过期的候选；unknown不是零秒。`src/timepredict/client.py` 提供 Python 消费端，`examples/scheduler_client.py` 可接入既有调度器。断网时有效期仍由本地单调时钟执行。期望驻留成本从条件分布计算，不能以中位数代替。

## 独立实验

train-regression、evaluate、replay-generate、公开数据 experiment 保留为独立实验。直接P50/P90回归没有完整生存分布，不能正式发布；代理及合成回放不可生产发布。旧回放精度不代表新AFT基线真实精度。

回放遵守Agent内串行调用，在时间划分间设置工作负载边界；t_return是模拟后续HTTP请求入口。它验证协议和训练契约，不证明真实SGLang性能或KV收益。
