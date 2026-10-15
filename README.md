# TimePredictModel

独立的工具回返时间预测插件。目标是 `T = t_return - t_out`：完整工具响应的服务端发送边界到后续请求的服务端入口，单位为秒，不包含续接排队或推理。

本模块负责采集、关联、特征转换、AFT 训练、模型验收发布、条件剩余时间和窗口接口。外部模块负责 C2 密度评分、KV 节点映射、加载和收益评价。

当前状态：模块本地验收通过；真实 Redis、GPU SGLang、跨主机校时、语义编码器质量及生产性能仍需目标环境验收。完整检查范围见 [验证记录](docs/verification.md)，不能将本地通过解释为生产就绪。

| 文档 | 内容 |
|---|---|
| [数据准备](docs/data.md) | 标签、采集、快照、时间划分和训练前检查 |
| [故障恢复与容量管理](docs/operations.md) | 来源安全退出、死信恢复、readiness、归档与预算 |
| [接入契约](docs/integration.md) | API v1、Python 消费端、分布和候选租约 |
| [验证记录](docs/verification.md) | 实现映射、本地检查、已修复问题和未验证项 |

## 文件布局

| 目录 | 保留内容 |
|---|---|
| `src/timepredict/` | 采集、状态机、特征、训练、发布、服务及客户端 |
| `tests/` | 协议、数值、并发、训练、接口和故障回归测试 |
| `patches/` | 固定 SGLang 补丁及上游摘要清单 |
| `deploy/` | 服务和重训模板、Redis/Docker 配置、目标服务器验收脚本 |
| `examples/` | Agent 和外部调度器接入示例 |
| `scripts/` | 本地验收和固定补丁验证入口 |
| `docs/` | 数据、接口和验证文档 |

`outputs/` 仅保存本地生成的验证证据，不提交。`.venv/` 是本地运行环境。旧代理数据下载、直接分位数回归、合成回放及重复审计入口已移除；相关历史仍可从 Git 获取。旧直接回归模型需用 AFT 重训；既有 AFT v3/v4 特征包保留兼容加载。

## 实现约定

- 每个 `(tenant, workflow_id, agent_id)` 最多一轮待回返，每轮一个工具，`n=1`、`parallel_tool_calls=false`。优先核对已有 tool_call_id；缺少 ID 时依赖已声明且无冲突的串行协议。request_id 只负责重试幂等及轮次标识，身份不输入模型。
- 默认结构化基线：工具名 one-hot、未知工具列、参数结构统计、按工具名和 JSON Pointer 保留的数字/布尔/长度/枚举列。缺失为 NaN，字段词典只在训练集拟合，发布时冻结。新参数仍影响通用统计，重训后可增加独立列。
- 对象键排序、数组保留顺序，数字不混合求和；字符串长度不等于文件大小，超时配置不当作执行标签。
- `--feature-mode semantic` 为独立增强实验，联合编码工具名、描述、实际参数，不自动取代结构化基线。语义模式无需人为工具分类器。旧 action_family 模型仅兼容加载。
- XGBoost `survival:aft` / `normal`，深度4、学习率0.05、默认500轮、验证损失30轮不改善停止。统一 sigma 在验证集选择；读取原始 margin，恢复为 log 秒的 mu。
- 精确标签 `[T,T]`；真实观察截断 `[a,+∞)`，保存 cutoff 而不伪造 t_return。取消、终止、发送失败、并发冲突、来源丢失分别排除。
- 按时间及完整工作流划分；跨边界工作流隔离。`--holdout-tool` 将相关整个工作流从训练/验证/常规测试移除，专门评估未知工具。

`log(T)=mu+sigma*Z, Z~N(0,1)`。条件窗口概率为 `(F(a+H)-F(a))/(1-F(a))`。剩余P50/P90、均值及 `E[min(R,H)]` 都从条件分布计算，等待期间不重新编码或运行树。生存概率低于1e-12时拒绝极端尾部外推。

截断驻留成本使用截断一阶矩，窄窗口采用单位区间积分，避免“微秒级调用、秒级窗口”时数值积分漏掉概率质量而错误返回零成本。

## 安装和测试

Python 3.11+：

```bash
uv sync --extra test
uv run --extra test python scripts/verify_plugin.py
uv run python scripts/verify_sglang_patch.py
```

第一项验证使用本地 CPU、fakeredis/Lua 和明确的模拟验收夹具，写入 `outputs/plugin-verification.json`。补丁验证需要联网下载三个固定上游文件，核对 SHA256、检查并实际应用补丁、验证修改后 Python 语法，写入 `outputs/sglang-patch-verification.json`；它不启动 GPU 服务。

基线不需要 PyTorch 或嵌入权重。语义实验使用 `uv sync --extra semantic`，提前缓存 `sentence-transformers/all-MiniLM-L6-v2`，revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`；可设置 `TIMEPREDICT_EMBEDDING_CACHE`。在线只读本地模型，不临时下载。

单元测试的哈希编码器和模拟验收数据只用于明确的夹具，不是正式来源或服务器验收证据。真实编码器、Redis、GPU SGLang、跨主机校时及性能另行验证。

## 采集部署

1. 在 SGLang v0.5.19 应用 `patches/sglang-v0.5.19.patch` 并安装本包。GPU进程只采集事件，不执行预测。
2. 配置持久化 Redis，设置 `TIMEPREDICT_REDIS_URL`、可选 `TIMEPREDICT_PREFIX`。
3. SGLang 启用 API key，每副本一个 HTTP/tokenizer worker，可多副本。Agent 携带 X-Workflow-ID、X-Agent-ID、X-Request-ID；重试复用 request_id。
4. Agent 等待期间每5秒 heartbeat，结束上报 terminate，取消上报 cancel。见 `examples/agent_client.py`。
5. 内部 tokens.json 为 `{"internal-token":"sha256-of-sglang-api-key"}`，必须是非空 JSON 对象，键为非空 ASCII 令牌，值为非空租户字符串；null 租户会在启动时被拒绝。租户命名空间来自已认证 SGLang key 的摘要。令牌文件不要提交仓库。

采集入口使用跨进程重启保持稳定的逻辑 ID，默认由 hostname、HTTP host/port 组成。容器 hostname 会改变时，为每个副本设置独立且稳定的 `TIMEPREDICT_OBSERVER_ID`，同一 ID 不应同时分配给两个存活副本。启动实例以不同 boot 标识区分，旧 boot 的迟到事件不能撤销新 boot。

入口 ID 按完整字符串区分，含 `/` 的 ID 与其前缀不视为同一入口。新旧 boot 按可证明的时钟区间排序，包含旧 boot 首次迟到的情况；区间重叠、时钟未验证或同一时刻无法判序时，相关候选降级 unknown，直到出现顺序可证明的替代入口。

任意入口的采集缺口或失联都可能遗漏另一入口调用的续接。插件按缺口时间区间阻断相关候选和训练标签；替代入口出现后，只放行能够证明在恢复边界之后发出的新调用。旧调用不会自动恢复，缺口前已完成的调用可由时间证据保留。旧版源记录缺少稳定 ID 或缺口边界时保守返回 unknown；迁移应排空调用后切换采集 prefix 或从完整事件历史重建状态。

```bash
uv run timepredict serve --window-h 2 --tokens-file /etc/timepredict/tokens.json --models models
```

无模型时返回 unknown，继续采集。跨主机必须有校时证据；时钟不可信时降级。正式服务模板见 `deploy/timepredict.service`。事件经 Redis Streams 顺序、幂等处理，状态、通知和首次历史事件原子提交。生命周期为 prepared→waiting→returned，迟到预测不能复活已闭合记录。

消费任务必须处于leader租约上下文。接管时优先认领全部待确认消息，逐页处理后再读取新消息，不等待消息空闲10秒，避免快速切换时把正常序列误判为缺口。每次状态提交仍检查租约，旧leader失去token后不能写入。

流式发送只把完整的 SSE `data: [DONE]` 行识别为结束；参数中的同名文本不会提前建立 OUT。Agent 示例遇到 length/error/abort 或缺失流结束标记时停止执行工具，并报告取消/终止。

## 快照、训练、验收发布

未知工具默认不能沿用已知工具的校准资格。候选和影子实验仍可测量未知工具；正式服务仅对明确留出并逐工具通过验收的名称放行。启用`require_tool_holdout=true`时，策略须给出`max_unknown_tool_nll`，并检查留出整体及每个工具的样本支持、窗口、分位数和条件窗口；新出现但未验收的名称返回`unknown_tool_not_calibrated`。

```bash
uv run timepredict export --output events.jsonl
uv run timepredict snapshot --events events.jsonl --cutoff UNIX_UTC_SECONDS --output samples.jsonl
uv run timepredict train --data samples.jsonl --data-kind observed --feature-mode structured --output candidates
```

正式样本携带关联方式、边界、原始事件及摘要。训练验证事件来源、调用内容、业务身份和真实时间差。数据需来自可信服务端；摘要用于一致性检查，不是来源认证签名。模型与配置文件统一采用 UTF-8，支持中文工具名和参数字段。

生成右删失样本时，截断时钟也必须有校时证据，并存在匹配来源/启动及 Agent 的新鲜心跳。普通 OUT 事件不能代替来源心跳。手动快照可提供 `--cutoff-clock-error MEASURED_SECONDS`（误差范围0至0.0025秒，必须来自实际测量）；未提供时只保留完整返回，等待样本记为 `cutoff_clock_unverified` 并排除。周期重训自动读取本机校时状态；无校时证据时同样不制造删失标签。

训练输出模型、词典、sigma、最佳迭代、依赖版本、数据摘要、覆盖率、窗口/条件窗口校准、经验基线和曲线。训练不更新线上 CURRENT。

候选以返回的VERSION目录定位，不再自动生成`best-model`软链接；正式服务仍只经质量门禁更新CURRENT。报告同时记录训练轮数、留出工具和training_signature，后者绑定相关源码、运行库版本及可选编码器配置，用于重训复用检查。

工作流时间元数据必须覆盖已观察的 OUT/返回/截断边界。时间划分取整个工作流最晚的已知观察时间，不能用偏早的结束时间把未来标签带入训练集；已结束工作流不能冒充行政右删失。

```bash
uv run timepredict serve --window-h 2 --tokens-file /etc/timepredict/tokens.json --shadow-candidate candidates/VERSION
```

影子模型提供分布供评价，始终 in_window=false。使用 `deploy/benchmark_server.py` 和 `deploy/verify_failover.py` 在隔离 Linux/GPU 部署生成验收证据；固定延时验收负载不替代真实日志上的精度实验。

```bash
uv run timepredict promote --candidate candidates/VERSION --serving-dir models --evidence server-acceptance.json --policy deploy/gate-policy.json
```

正式标签、校准/基线/样本支持及对应版本的服务器验收全部通过，才原子更新 CURRENT。加载要求关键文件均列入摘要清单，核对实际 normal AFT 目标、sigma、最佳迭代、时间缩放及报告/特征契约，再验证归档证据。CURRENT 只允许指向正式目录内的直接子目录。附带材料不能越界或覆盖核心文件，其内容也参与候选版本计算。摘要用于检测不一致，包及验收材料仍须来自可信来源。

门禁拒绝非有限支持量、非整数样本数、超过样本数的有效样本量、越界概率、负 Brier 和布尔错误计数；非对象报告段返回拒绝。通过后才设置 distribution_calibrated=true。等待中的调用保留原预测版本。证据过期后不再对新调用发出可用预测。P99、吞吐和校准门槛是验收目标，不是已测得结果。

候选及正式发布文件均在切换前执行文件同步，POSIX 同时同步目录；同步失败不再被忽略。若 CURRENT 已替换但最终目录同步失败，报告 `PromotionCommitUncertain`，需要核对 CURRENT、模型和存储状态。Windows 不执行目录 fsync，本地通过不证明断电恢复能力。

## 周期重训和对照实验

```bash
uv run timepredict retrain --output training --feature-mode structured
uv run timepredict ablation --data samples.jsonl --data-kind observed --output ablation --holdout-tool unseen_tool
```

retrain 固定历史流边界及UTC截断，在独立进程训练候选；可续租的 Redis 租约避免正常运行时重叠训练。复用要求收据、候选完整性、候选报告中的数据摘要/训练配置及training_signature全部匹配；相关源码或依赖改变也会触发重训。旧报告缺少这些字段时重新训练，既有受支持AFT包仍可独立加载。损坏或非对象last-run.json先保留为`.invalid-last-run-*.json`，再重建收据。`deploy/timepredict-train.service`、`deploy/timepredict-train.timer`提供每小时模板。

取消异步任务不能立即停止已经运行的训练线程，候选写入可能继续；正式CURRENT提交仍受取消和租约检查保护。training_signature用于缓存一致性，不替代特征契约升级、来源认证或发布门禁。

要验收后自动发布，同时提供 --serving-dir、--evidence、--policy；证据需匹配新版本。提交前的验收或文件同步失败不切换 CURRENT；提交后的目录同步错误记录为 `publication_outcome_uncertain`，不能假定仍在使用旧模型。训练收据本身写入失败时也需直接核对 CURRENT。模板默认仅训练候选。

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

接口使用内部 Bearer token，按租户隔离。仅使用 ready、in_window、非shadow且未过期的候选；unknown不是零秒。`src/timepredict/client.py` 提供 Python 消费端，`examples/scheduler_client.py` 可接入既有调度器。断网时有效期仍由本地单调时钟执行，本地解析、建立缓存和计算整批条件分布的耗时都会消耗租约；SDK输出前再次剔除已过期候选。期望驻留成本从条件分布计算，不能以中位数代替。

增量通知重放还会核对读取时的共享状态和观察租约；已闭合的轮次不会因移除通知在下一页而继续作为有效候选。非法通知游标返回422，通知流或共享状态不可用返回503。SDK的输入、内部缓存和返回行相互隔离，修改返回行的分布字段不会改变后续预测。

SDK只为当前候选保留计时信息，版本索引按业务键保留，不逐轮积累。长期接收通知时仍应周期拉取全量快照，恢复遗漏并回收已结束业务键的版本索引。采集事件无法序列化（含过深嵌套）时丢弃该事件、保留序号缺口，后台继续发送后续事件；相关观察按缺口规则降级。

坏消息经死信隔离后继续消费，但观察缺口仍阻断候选。永久下线和缺口恢复使用默认关闭的独立管理员接口；`/internal/ready`以503报告模型、来源或观察不可用。暂时性模型/依赖失败对仍等待轮次退避重试。通知默认保留100000条，落后游标触发SDK清空缓存并重拉快照。历史归档、状态/重训预算及每日维护模板见[故障恢复与容量管理](docs/operations.md)。单JSON状态和全历史回放仍有规模上限，预算超限会拒绝，不会静默删减训练数据。

后续开发的接口入口、扩展约束和验收顺序见[接入文档的开发交接部分](docs/integration.md#后续开发交接)。

## 目标环境验收

`deploy/benchmark_server.py` 比较固定工具负载下采集前后的服务开销并验证流式/非流式与跨入口关联；`deploy/verify_failover.py` 在专用部署执行操作者指定的停止命令并检查接管；`deploy/benchmark_scale.py` 在独立 Redis prefix 下测量状态事务、有效候选分布计算、内存及离线评估开销。各脚本使用 `--help` 查看参数。

容量脚本使用明确的模拟分布，输出 `scale_acceptance`，不能作为正式发布证据。服务器验收也不能代替真实日志上的预测质量和外部 KV 收益实验。发布门禁策略在 `deploy/gate-policy.json`，应按目标环境采集证据，不能填入假测量值。
