# 当前代码阅读与逐文件说明

本文按当前目录的实际源码说明。新增注释仅使用注释语法，保持程序表达式、函数签名、默认值和控制流不变。文档中的实现差异是阅读结果，不表示本次已修复。JSON、模型二进制及模型清单保持原样，以免破坏合法 JSON 或模型校验摘要。

## 1. 这个项目在做什么

TimePredictModel 希望预测：SGLang 发出一次工具调用之后，Agent 带着工具结果再次请求 SGLang，需要等待多久。目标时间为 `T = t_return - t_out`，单位为秒。它包含传输、工具执行及 Agent 编排时间，不包含续接进入 SGLang 后的模型排队与推理时间。

例如，SGLang 让 Agent 查询文档，Agent 执行查询后携带对应 `tool_call_id` 再发请求。项目收集这两个时间边界，构建工具调用样本，通过工具名称、描述、schema、参数数值/长度/结构和语义向量学习耗时。预测器是独立 CPU 服务；SGLang 的 GPU 进程只采集事件。

项目还维护“哪些 Agent 正在等待工具结果”，通过 HTTP 快照和 Redis 通知提供预测候选。完整分布路径理论上可以回答“已经等了 a 秒，未来 H 秒内会回返的概率是多少”，为外部调度提供信息。仓库不实现 KV 节点映射、实际预加载或缓存收益评估。

当前包含两条模型路径：

| 路径 | 实现 | 标签与输出 | 当前限制 |
|---|---|---|---|
| 直接 log 时间回归 | `regression.py` | 精确耗时；平方误差 P50、分位数回归 P90，再用验证残差修正 | 两个分位数不定义完整生存分布；包标记不能生产发布 |
| AFT 生存模型 | `training.py` | 精确或右删失标签；`survival:aft` 拟合 log 时间位置，固定 sigma | 保留训练、删失诊断、门禁代码；校准及正式来源链路未完成 |

P50/P90 是分位数估计，不能解释为单条调用一定会在对应时间内结束。直接路径中所谓 P50 的基础训练目标实际上是 log 时间平方误差，而非原始 0.5 分位数损失；随后才通过验证残差的 0.5 分位数加偏移。

## 2. 一次调用怎样穿过系统

1. Agent 发送 OpenAI chat 请求，携带认证及 workflow/agent/request 三个 ID；每个 Agent 一轮只允许一个工具，`n=1`、`parallel_tool_calls=false`。
2. `ObservationMiddleware` 在 HTTP 入口记录初始时间；上游认证与校验通过后，`accepted()` 发送 `ingress`。租户由已认证 key 的 SHA-256 派生。
3. `Capture` 从完整 JSON 或 SSE 增量拼出工具 ID、名称和参数；响应终止发送前记录边界并发送 `prepared`，发送成功后发送 `out`。
4. `Publisher` 非阻塞入本地队列，后台按序写 Redis `:events`。采集失败、队列溢出或来源序号缺口会使观察不可靠。
5. `Worker` 经 Redis 租约选出 leader，消费事件；`RedisStore` 调用 `reduce_event()`，在一个事务中提交状态、通知和首次事件历史。提交后才确认消费。
6. 等待轮次被调度到 CPU 线程预测。写回时再确认它仍等待且尚未预测，避免工具已经返回后迟到预测复活记录。
7. Agent 执行工具并续接。`ingress` 优先匹配工具 ID；当前实现还保留无 ID 的串行业务键降级，记录 `serial_fallback`/`degraded`。
8. 状态机计算可信时间差，闭合记录并发布移除通知。来源失联、取消、终止、发送失败、并发冲突和不可信时钟分别处理。
9. 消费端读取全量 `/internal/window`，从返回的游标续读 `/internal/notices`。`CandidateCache` 按 revision 过滤旧消息，并自行过期。

## 3. 训练数据和预测怎么产生

离线流程为：事件历史 → `export` JSONL → `snapshot` 截取 → `check-data` → 时间划分 → 仅训练集拟合动作族与词典 → 模型训练/验证 → 独立测试 → 候选包 → 服务器验收 → `promote`。

精确样本为 `[T,T]`。右删失为 `[a,null]`，表示观察到截取时已等待 a 秒，尚不知道最终 T。取消、失联和观察缺口不是正常右删失。内部计算把 null 上界转为 `math.inf`，写 JSON 时再转换回 null。

数据按 workflow 开始时间约 70%/15%/15% 分训练/验证/测试。整个工作流不跨集合；跨时间分界或未结束的早期工作流进入 quarantine。至少需要七个不同开始时间，且各正式集合非空。

特征向量依次包含：工具/动作族指示、未知工具指示、384 维语义向量、九项结构统计、最多 4096 个结构字段。参数数值保留数值，字符串使用字符/字节长度，对象递归，数组保留长度及内部统计，schema 枚举展开指示列；缺失结构列为 NaN。业务身份和耗时标签不直接进入特征。

语义文本包含操作名称、描述、schema、角色和参数结构，排除具体参数值。真实编码器使用固定 revision 的 `all-MiniLM-L6-v2`，从本地缓存读取；长文本按容量分块，均值聚合后归一化。训练和在线必须符合相同编码器契约。

## 4. Python 源码逐文件说明

所有路径均相对于仓库根目录。每个 Python 文件已增加中文文件说明以及类、函数、嵌套函数和关键分支注释。

| 文件 | 具体职责、主要入口与协作关系 |
|---|---|
| `src/timepredict/__init__.py` | 包版本 `0.1.0`。只声明版本，不在导入时加载 SGLang、模型或启动服务。 |
| `src/timepredict/cli.py` | `main()` 是安装后的 `timepredict` 入口。注册 serve/export/snapshot/check-data/train/train-regression/evaluate/experiment/replay-generate/cache-manifest/audit/promote；按分支延迟导入依赖。serve 构造 RedisStore、Clock、Worker 和 FastAPI；export 分批读 history；其余命令转交专用模块。 |
| `src/timepredict/sglang.py` | 上游接入层。`safe_hook` 隔离同步采集异常；`latest_tool_ids` 只取最新工具轮次；`Capture` 拼接并检查调用；`accepted` 记录认证请求；`observe_nonstream/observe_stream` 捕获输出；`ObservationMiddleware` 管理发布器并记录发送边界；`install` 按环境配置启用。 |
| `src/timepredict/publisher.py` | `Publisher` 用默认 4096 容量异步队列把入口采集与 Redis 延迟隔离。emit 先递增序号再非阻塞入队；run 对当前事件指数退避重试；五秒心跳携带丢弃/队列/开销指标；退出最多等待三秒排空。 |
| `src/timepredict/clock.py` | `Clock` 保存进程 source、启动 boot、UTC/单调时间及 chrony 误差。refresh 查询 chronyc，stamp 检查 5ms 跳时；`elapsed` 同启动用单调差、跨来源用 UTC 且要求误差总和≤5ms。未知或负时间差明确拒绝。 |
| `src/timepredict/state.py` | 核心确定性状态机。empty_state 初始化 records/pending/seen/sources/orphans/heartbeats/workflows 等；reduce_event 管理 prepared/out/ingress 和生命周期。generation 是业务键与 request_id 的摘要；notice 递增 revision；close 匹配续接、计算标签；reconcile_intervals 检查重叠并处理乱序；apply_prediction 拒绝迟到写回；compact 清理热历史并保留幂等 tombstone。 |
| `src/timepredict/store.py` | `RedisStore` 管理 prefix 下 `:state/:events/:history/:notices`。mutate 用 WATCH 检测状态及 leader token 改变，再 MULTI/EXEC 一次提交状态、通知与新事件历史；冲突重试并记录 CAS 指标。apply/prediction 是两种状态变更包装。当前是整个 namespace 一份 JSON 文档。 |
| `src/timepredict/service.py` | FastAPI 与 Worker。window_rows 计算租户可见状态和窗口；Worker supervise/consume/tick/predict_one 负责重启、消费、模型加载、观察失效和 CPU 预测；run_elected 获取十五秒租约、每三秒续租。API 提供 window/notices/lifecycle/health，使用令牌映射隔离 tenant。 |
| `src/timepredict/freshness.py` | `observation_status` 独立于 leader 检查处理器两秒、来源和 Agent 三十秒新鲜度，返回 unknown 原因及最短 TTL；跨时钟不可靠时也降级。当前还检查其他未退休来源，单一来源过期可使读侧整体观察不可靠。 |
| `src/timepredict/candidates.py` | `CandidateCache` 是消费侧辅助类。snapshot 设置全量版本下界；notice 拒绝旧 generation/revision，remove 只撤销对应轮次；_put 扣除网络往返时间，仅保存 ready/in_window；candidates 使用单调时间主动过期。不会执行调度或缓存预加载。 |
| `src/timepredict/metrics.py` | `Histograms` 保留每种延迟最近 8192 个有效值，另外累计总 count。snapshot 输出窗口 P50/P90/P99/max 与保留数；它是排序样本分位数统计，并非无限历史统计。 |
| `src/timepredict/features.py` | 严格解析参数并提取结构特征。canonical 固定 JSON 顺序且拒绝 NaN；parse_arguments 约束大小/深度；extract 生成 JSON Pointer 字段及统计；FeatureSpec.fit 仅训练集建词典，transform 冻结顺序并拼入语义向量。数值须有限且可用 float32 表示。 |
| `src/timepredict/semantic.py` | descriptor_text 生成不含实际参数值的操作描述。FrozenSemanticEncoder 是测试哈希替身；SemanticEncoder 使用真实固定模型、本地缓存、实例复用及文本 LRU；_chunks 避免截断，encode_text 聚合归一化。显式 `TIMEPREDICT_TEST_ENCODER=1` 才启用替身。 |
| `src/timepredict/actions.py` | `ActionFamilies` 把结构/语义相近调用归入操作族，避免只能精确匹配工具名。描述符包含名称、描述、字段、类型、枚举和角色；相似度组合词集合、字符 n-gram、字段/类型、IDF 和向量点积。默认阈值 0.42、第一二名分差 0.08；低分/歧义分别返回 unknown_action/ambiguous_action。 |
| `src/timepredict/distribution.py` | `LogNormal` 提供条件剩余时间数学。log_survival 在对数域计算生存概率；summarize 返回剩余 P50/P90、H 内回返概率、剩余期望及 H 截断期望。概率过小或非有限结果抛 UnreliableDistribution。只有有效且已校准分布才应被在线窗口使用。 |
| `src/timepredict/dataset.py` | validate_rows 检查非空数据、唯一 generation、一致 workflow 元数据、秒级精确/右删失标签及调用参数；validate_label_provenance 额外验证正式 SGLang 来源和工具 ID 证据；describe 给出规模与工具统计。格式合法和来源合法是不同检查。 |
| `src/timepredict/training.py` | AFT 路径。snapshot 回放事件截取样本；split_rows 时间划分；matrix 设置标签上下界；nll/KM/IPCW 处理删失评估；baseline_reports 拟合全局/按工具基线；train 搜索 sigma、提前停止、评价并调用 publish。还包含摘要、数据检查及 SVG/逐轮准确率诊断。 |
| `src/timepredict/regression.py` | 直接回归路径。train_regression 要求真实语义编码器及精确标签；合成回放须提供有序时间边界。训练 P50/P90 双 booster、验证残差偏移，报告精度/覆盖/pinball/基线与 60% 验收。encoder_contract 绑定本地权重摘要；RegressionModel 恢复包；evaluate_model 独立测精确样本。输出包一直标记不能生产发布。 |
| `src/timepredict/model.py` | 统一加载 Model 与 AFT publish。Model 解析 CURRENT 或目录，依据 model_type 转入 RegressionModel 或检查 AFT 摘要、特征、依赖及 promotion；predict 返回参数与状态。publish 写 staging、fsync、原子重命名候选并尝试 best-model 链接；不改生产 CURRENT。 |
| `src/timepredict/release.py` | evaluate_gate 汇总来源、真实编码器、有效样本、窗口/分位数校准、基线、服务器功能/性能及证据时效检查。promote 检查通过才复制候选、策略和证据，生成 promotion.json 并原子切换 CURRENT。与直接回归包的 report 格式不兼容。 |
| `src/timepredict/replay.py` | generate 开本地 ThreadingHTTPServer，用 sleep/http/file/database 四种受控任务测量。文件和 SQLite 有实际操作，但时长还包含参数控制的 sleep；http_tool 也不能解读成真实外部 HTTP 工具性能。保存完整样本、分割文件及清单；本地 tool_end 被用作 t_return，标记 observed_replay。 |
| `src/timepredict/datasets.py` | 固定四个公开数据源 revision。AgentTrace 检查 TOOL span 边界；Trace Commons 按工具 ID 配对；Betterwright 优先浏览器步骤、再退回模型轮次；Exgentic 不造耗时，只提语义描述。load_proxy_rows 检查本地文件 hash 或流式加载；write_local_manifest 写配置 revision 和 hash。不同标签不可混合等同正式回返。 |
| `src/timepredict/experiment.py` | run 对 AgentTrace、Trace Commons、Betterwright 分别调用 AFT train，保存独立候选路径和报告；Exgentic 仅统计无标签语义覆盖。失败数据源单独记录。当前实现会保留候选，不是只留下摘要文件。 |
| `src/timepredict/accuracy.py` | 面向旧 AFT 包的诊断入口。render 校验原始队列摘要，重建工作流划分，逐轮计算 exp(margin) 相对误差曲线，测试仅突出原选定轮次；需要可选 Parquet/绘图依赖。不重新训练、不修复旧导入问题，也不适配 RegressionModel。 |
| `src/timepredict/audit.py` | run 从已有证据 JSON 读取检查状态，输出 implemented/missing/not_verified/verified_in_wsl/not_verified_server 等。默认找输出旁 final-audit.json；不实际启动 Redis、GPU 或性能测试，生产发布始终标 false。 |

## 5. 测试、脚本和示例逐文件说明

| 文件 | 作用 |
|---|---|
| `tests/test_contract.py` | 七项基础契约：无工具 ID 的串行降级、非法参数拒绝、测试向量稳定、条件概率、流式参数拼接、来源失联隔离、不相关动作匹配。需要项目依赖；动作族测试正常会触发编码器加载。 |
| `tests/test_leakage.py` | 三项防泄漏检查：描述文本排除实际参数值、特定测试调用的词典不含标签字段、三个正式集合工作流互斥。不是任意输入的通用字段黑名单实现。 |
| `tests/test_replay_contract.py` | 在临时目录实测 100 条受控回放，检查第一条样本边界顺序、来源标记及参数不含结果/时间。 |
| `tests/test_training_smoke.py` | 十个合成工作流、三轮 AFT 训练；检查包文件、accuracy_curves、独立 Model 加载和 ready 预测版本。只证明最小打包契约。 |
| `scripts/run_wsl_tests.sh` | 切换根目录，设置 PYTHONPATH，禁用 pytest 系统插件自动加载，顺序运行 compileall、pytest、verify_wsl.py；需已准备 Python/项目依赖/真实编码器或显式测试替身。 |
| `scripts/verify_wsl.py` | 生成源码 SHA-256 和环境证据：语法、pytest、prepared→waiting、补丁标记及等价参数描述。Redis 是否安装与是否实际验证分开记录；GPU、跨主机 chrony 和服务器性能标未验证。 |
| `examples/agent_client.py` | 可运行的串行单工具客户端示例。创建独立 workflow/request ID，五秒心跳，支持 SSE/JSON，最多五轮；仅执行 search_documents 演示 sleep 与固定结果，工具 ID 回填续接；正常 terminate，异常 cancel/terminate。实际采集要替换演示工具。 |

## 6. 部署文件逐个说明

| 文件 | 作用及输入输出 |
|---|---|
| `deploy/Dockerfile` | Python 3.11 slim CPU 镜像，安装 libgomp1/chrony，复制并 pip 安装包，非 root 运行 timepredict。启动参数、Redis、认证文件、模型和语义缓存需外部提供；安装 chrony 不证明校时服务已运行。 |
| `deploy/train.sh` | Linux 训练包装器，要求 DATASET/MODEL_DIR，默认四线程、500 轮、data-kind observed；额外参数透传。不会替调用方补齐正式来源证据。 |
| `deploy/timepredict.service` | systemd 模板，专用 timepredict 用户，读 service.env，在 localhost:8098 启动正式模型服务，故障三秒重启。路径、账户及权限需要实际部署。 |
| `deploy/redis.conf` | Redis 保护模式及 AOF 配置，everysec 刷盘、禁止内存淘汰；认证/ACL/TLS 用部署密钥。不会自动归档/裁剪事件流。 |
| `deploy/gate-policy.json` | 显式发布阈值：至少 100 支持样本/50 有效样本，H=2 秒、age=1/2 秒，窗口误差≤0.1、覆盖误差≤0.05、Brier 不退步；入口 P99≤2ms、吞吐下降≤1%、预测可用 P99≤100ms；证据最多七天。JSON 标准不支持注释，字段解释在此保存。 |
| `deploy/benchmark_server.py` | Linux 完整往返验收。读取工具请求 JSONL 与两种认证 key，交替测试 baseline/instrumented 部署，流式与非流式混合、续接使用另一入口；固定工具延时；检查 Redis 闭合、跨入口、时钟、预测和 failover 证据，输出 server_acceptance JSON。 |
| `deploy/benchmark_scale.py` | Linux 状态容量验收。每组规模生成独立 Redis prefix，测 CAS P99/重试/状态字节/内存/窗口耗时，清理专用键，再测带删失的离线评估与峰值 RSS。当前合成预测缺校准标记导致 ready_rows 条件不通过，见后文。 |
| `deploy/verify_failover.py` | Linux leader 接管演练。要求独立验收部署、旧 leader 健康且有模型；执行操作者提供的 JSON argv 停止命令，不走 shell；确认不同进程接管同模型，并检查新心跳进入 history，输出 failover.json。脚本不自动重启被停止的副本。 |

## 7. 补丁、文档和工程配置

| 文件/目录 | 作用 |
|---|---|
| `patches/sglang-v0.5.19.patch` | 固定上游版本的三文件补丁：HTTP 启动时 install；请求验证后 accepted；stream/nonstream 输出时 observe。新增中文注释位于补丁新增行，并同步更新 hunk 行数。 |
| `patches/sglang-v0.5.19.json` | 保存上游 commit 与三个原始文件 SHA-256，便于人工/外部检查版本基线。当前 install 不会自动读取它验证上游文件。 |
| `README.md` | 安装、接入、事件采集、训练、模型发布、HTTP 消费与服务器验收说明；现在链接本文。包含部分历史描述，阅读时应与实际实现边界对照。 |
| `DATASET.md` | 正式数据定义、采集建议、JSONL 字段、删失规则、时间划分及公开代理数据说明。部分早期描述与当前 schema/语义/来源要求不同，见后文。 |
| `IMPLEMENTATION_AUDIT.md` | 人工审计摘要，区分已实现逻辑与未通过服务器证据的部分，不是自动测试执行器。 |
| `CODE_GUIDE.md` | 本文：完整目录阅读指南、逐文件职责、主要流程及代码与文档差异。 |
| `pyproject.toml` | hatchling 构建，包名 sglang-timepredict，Python≥3.11；声明数值计算、XGBoost、Redis、FastAPI、HTTP、Sentence Transformers 依赖；experiments 添加 datasets/pyarrow/matplotlib；timepredict CLI 入口与 src wheel 布局。 |
| `uv.lock` | uv 生成的依赖锁定文件，记录版本、来源与摘要，用于可复现安装。自动生成内容保持原样，不逐条添加注释。 |
| `.gitignore` | 忽略虚拟环境、缓存、密钥、采集数据、候选/正式模型、outputs 和验收证据；因此目录中有文件不代表被 Git 跟踪。 |
| `.gitattributes` | 文本统一 LF；patch 使用特殊空白规则，保留上游上下文。 |
| `.git/` | Git 内部提交历史、对象、引用和配置，不属于应用源码，不编辑。 |

## 8. outputs 中现有文件逐个说明

outputs 被忽略，是本地运行产物。此处说明任务开始时实际存在的文件；本次未重训、重写模型包或更新旧证据。

| 文件 | 作用 |
|---|---|
| `outputs/final-audit.json` | 既有 WSL 审计状态，标记 incomplete、正式标签缺失、真实编码器/Redis未验证、生产不允许。source_digest 绑定旧源码字节；注释改变字节后不再代表当前源码的新验证。 |
| `outputs/tracecommons-evaluation.json` | 3000 条、19 个工作流的公开代理泛化摘要。测试 111 条，P50 10% 容差命中约 2.70%，未超过按工具基线且未达到 60% 阈值；标记不适用生产。 |
| `outputs/replay-model/p50-model.ubj` | 直接回归 P50 XGBoost 树的二进制文件，需要匹配特征/轮次/偏移加载。 |
| `outputs/replay-model/p90-model.ubj` | 独立 P90 分位数回归树二进制，不是把 P50 乘系数得到。 |
| `outputs/replay-model/features.json` | 结构字段、工具/动作族词典和语义维度/version/revision/backend，确定向量列顺序。 |
| `outputs/replay-model/action-families.json` | 动作族代表描述、向量、成员名称、阈值、IDF 及编码器契约。 |
| `outputs/replay-model/embedding-manifest.json` | 冻结语义模型身份、维度、版本、分块方式以及本地文件摘要，用于恢复时契约检查。 |
| `outputs/replay-model/manifest.json` | 模型类型、包版本、标签、双模型 best iteration、校准偏移、依赖版本、文件 hash 和不可生产发布标记；包加载依据。 |
| `outputs/replay-model/dataset-manifest.json` | 规范化样本摘要，各集合 generation/数量及标签来源；通常不是原始训练样本。 |
| `outputs/replay-model/report.json` | 分割规模、双模型验证指标、精度/覆盖/pinball、基线、验收和逐轮曲线等诊断数据。 |
| `outputs/replay-model/accuracy_curve.png` | P50/P90 训练/验证准确率曲线与最终选定轮次的测试点；横轴为 boosting round，容差分三档。 |
| `outputs/replay-model/calibration_curve.png` | 名义 0.5/0.9 分位数与测试覆盖率的对比图；只展示分位数校准，不证明完整分布校准。 |

## 9. 当前实现中需要区分的边界

1. **正式来源链路尚未闭合。** snapshot 只输出基础样本字段，未携带 `sglang_return_observed`、工具 ID 验证、t_out/t_return、source_manifest/event_evidence。直接把 snapshot 输出交给 `train --data-kind observed` 会被 provenance 校验拒绝。取消/失联与普通右删失仍需区分；本次不补造来源证据。
2. **窗口当前不会自动使用 AFT 参数。** window_rows 要求 prediction 中 `distribution_calibrated=true`，但当前 AFT Model.predict 只返回 mu/sigma/model_version。直接路径明确标 false。因此两条现有路径都不能仅凭 ready 预测生成有效分布窗口候选；缺失标记返回 unknown/distribution_not_calibrated。
3. **直接模型正式加载分支提前返回。** Model 遇到直接回归 model_type 后，先创建 RegressionModel 并 return，没有执行 AFT 分支的 require_promoted 校验。RegressionModel 会核验文件契约，但这不是 promotion 门禁。发布 evaluate_gate 与直接模型 report 格式也不兼容；`production_releasable=false` 不能被视为完整服务端强制发布隔离。
4. **prepared→out 快速分支与新建 out 不同。** 前者设置 waiting 后直接返回，没有走后面新建 out 的来源错误/发送失败 reason、孤立入口和并发调和检查。不能把所有 out 路径都描述为已经执行这些检查；真实采集链路需专门验证。
5. **关联允许降级。** 当前状态机与契约测试允许缺少工具 ID 时按串行业务键关联，明确记为 serial_fallback/degraded；正式 observed 校验只接受经过验证的工具 ID。README 的严格描述不能替代这个实际分支。
6. **测试语义替身存在限制。** SemanticEncoder.encode_text 的测试分支把文本作为 `arguments.text` 传给 FrozenSemanticEncoder，而后者描述符排除具体参数值。因此不同文本可能生成相同向量，不能用测试替身结果证明真实语义匹配质量。
7. **规模夹具与窗口接口不一致。** benchmark_scale 构造的预测缺 distribution_calibrated，window_rows 会降级 unknown，ready_rows==size 条件不能通过。它也不能被解读为已测完整分布计算容量。
8. **AFT 候选选择保留历史做法。** train 直接比较各 sigma 的 booster.best_score，README 关于模型选择的部分描述与此不一致。旧 accuracy.py 还未扣除包 time_scale，不适合直接套用到缩放训练的当前 AFT 包。
9. **直接 P90 的夹紧规则存在差异。** 训练报告 predict(q=.9) 与未加偏移的原始 P50 比较，RegressionModel.quantiles 最终与加偏移 P50 比较；特殊情况下结果可能不同，打包时有恢复预测一致性检查。逐轮曲线也未加最终校准偏移，不能当成最终校准预测曲线。
10. **部分历史文档描述已过时。** DATASET.md 称描述不进特征，而当前语义描述明确使用 description；称曾有 70 项测试，当前四文件实际包含 12 项。experiment.py 仍保留独立候选。回放不包含真实工具结果网络续接至 SGLang；高分不能外推生产。以本次实际源码和本次验证记录为准。

## 10. 建议阅读顺序与本次注释原则

先读 `cli.py → sglang.py → publisher.py → state.py → store.py → service.py`，理解在线链路；再读 `dataset.py → features.py → semantic.py → actions.py → regression.py/training.py → model.py → release.py`，理解训练和发布；最后阅读测试、部署验收、已有 outputs。

本次使用中文 `#` 注释，覆盖 34 个 Python 文件的文件职责、所有类/函数定义和关键实现边界；Shell、Dockerfile、systemd、Redis、TOML、Git 配置以及补丁新增代码也补充说明。未对生成的 uv.lock、JSON、模型二进制或图片注入注释。Python 修改前后以忽略位置属性的 AST 比较，确保没有改变可执行语法树。

## 11. 本次实际验证记录

- 对全部 34 个 Python 文件逐一与 Git HEAD 比较忽略位置属性的 AST，全部一致；全部源码可编译。
- `git diff --check` 通过；`git apply --numstat patches/sglang-v0.5.19.patch` 能正确解析三个文件的补丁。未在真实上游 checkout 应用补丁，因此不据此宣称 SGLang 已运行验证。
- 当前 Windows Python 使用 `PYTHONPATH=src`、关闭 pytest 插件自动加载，并显式设置 `TIMEPREDICT_TEST_ENCODER=1`，运行现有 12 项测试：**11 通过，1 失败**。缺失 Redis 客户端只安装到系统临时验证目录，未修改项目依赖配置或全局环境。
- 失败为 `test_replay_boundaries_and_feature_exclusion`：100 条并发回放的工作流跨时间分界，quarantine 后出现空集合，`split_rows` 抛出 `empty temporal split after crossing-workflow quarantine`。在独立临时目录恢复修改前 HEAD 的全部 Python 源码并运行同样测试，也得到 **11 通过、同一项失败**。
- 上述测试仅验证本地契约，显式哈希替身不代表真实 Sentence Transformer 质量。未重训现有模型、修改既有审计证据，或执行 Redis 集成、GPU SGLang、真实服务器性能与故障演练。
