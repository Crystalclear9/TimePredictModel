# TimePredictModel

源码逐文件职责、运行流程、模型产物和当前实现差异详见 [代码阅读与文件说明](CODE_GUIDE.md)。相关 Python 源码、测试及部署脚本已添加中文注释。

**面向 SGLang 工具调用的回返时间预测与训练系统。**

当前尚未完成原方案的全部验收。2026-09-20 复查发现公开数据参数丢失、不同目标 NLL 不可直接比较、实验覆盖不足等问题，详见 [实现审计](IMPLEMENTATION_AUDIT.md)。现存公开数据模型仅供诊断，不能作为已验证的生产回返预测器。

TimePredictModel 从 SGLang 的 OpenAI 兼容接口采集工具调用事件，关联后续工具结果请求，使用结构化特征与 XGBoost GBDT 直接回归 `log(T)`。P50 使用平方误差回归，P90 使用独立分位数回归；AFT 仅用于删失诊断。在线服务在分布校准完成前只暴露 P50/P90，并明确返回 `distribution_not_calibrated`，不会把两个分位数伪装成 LogNormal 生存分布。

项目面向 Linux 服务器训练与部署。预测在独立 CPU 服务中执行，XGBoost 不进入 SGLang GPU 推理进程。本仓库包含源码、固定版本补丁、Agent 示例和部署工具，不包含预训练模型或可直接用于生产的数据集。

| 项目 | 说明 |
|---|---|
| SGLang 基线 | `v0.5.19` · [`0bcd822`](https://github.com/sgl-project/sglang/tree/0bcd822377da7b5718e674eaf9c870d349424dd1) |
| 建模方法 | 结构化特征 + 冻结 Sentence Transformer 语义向量 + XGBoost `survival:aft`，对数正态分布 |
| 服务依赖 | Python 3.11+、Redis 7.0+、chrony |
| 输出接口 | 认证 HTTP API、版本化候选通知、全量快照 |
| 当前范围 | 调用采集、样本关联、离线训练、影子预测和窗口候选 |
| 范围之外 | C2 密度评分、KV 节点映射、实际预加载及缓存收益评估 |

## 导航

- [系统设计](#系统设计)
- [安装与接入](#安装与接入)
- [数据准备与训练](#数据准备与训练)
- [模型发布](#模型发布)
- [接口与消费约定](#接口与消费约定)
- [服务器验收](#服务器验收)
- [运行边界与验证状态](#运行边界与验证状态)
- [仓库结构](#仓库结构)

## 系统设计

```mermaid
flowchart LR
    A[Agent] --> S[SGLang HTTP 入口]
    S --> R[Redis 事件流与共享状态]
    R --> P[独立 CPU 预测服务]
    P --> C[C2 窗口与候选接口]
    R --> D[事件导出与训练快照]
    D --> T[离线 XGBoost 直接回归]
    T --> M[候选模型与评估报告]
    M --> G[影子验收与发布门禁]
    G --> P
```

### 预测目标

$$
T = t_{\mathrm{return}} - t_{\mathrm{out}}
$$

- **发出时刻**：完整非流式 JSON 响应或流式 `[DONE]` 标记提交给 ASGI `send` 之前。发送成功后异步发布事件，发送异常单独记录。
- **回返时刻**：携带匹配 `tool_call_id` 的续接请求进入 SGLang 时。认证和请求校验成功后提交关联事件，但使用最初的入口时间。

因此，标签包含网络、工具执行和 Agent 编排耗时，不包含续接请求进入 SGLang 后的排队与推理时间。发出边界表示服务端提交发送，不等同于客户端接收完成。

模型在工具发出后预测一次并缓存分布参数。等待期间通过条件生存分布计算剩余时间：

$$
P(R\leq h\mid T>a)=1-\frac{S(a+h)}{S(a)}
$$

其中，$a$ 为已经等待的秒数，$S$ 为总回返时间的生存函数。接口同时返回剩余 P50/P90、$E[R]$ 和 $E[\min(R,H)]$，分别用于时机判断和成本估计。

### 调用协议

每个 `(tenant, workflow_id, agent_id)` 同时只允许一轮待回返调用，每轮只支持一个工具。客户端设置 `n=1`、`parallel_tool_calls=false`，收到完整响应后再执行工具。

| 请求头 | 约定 |
|---|---|
| `Authorization` | SGLang 已配置的 Bearer API key |
| `X-Workflow-ID` | 当前业务工作流实例标识 |
| `X-Agent-ID` | 工作流中的 Agent 实例标识 |
| `X-Request-ID` | 逻辑请求标识；传输重试时保持不变 |

续接请求使用 OpenAI 协议已有的 `tool_call_id` 关联结果。只有最新续接轮次中匹配的工具结果才闭合记录；单纯再次收到同一 Agent 的请求不足以认定回返。业务 ID 用于关联与分组，不进入模型特征。

## 安装与接入

以下命令在 Linux shell 中执行。`/path/to/...`、模型名、parser 名、地址和密钥均需替换为实际配置。SGLang 的 GPU 依赖按其官方说明单独安装。

### 1. 安装预测器

```bash
git clone https://github.com/Crystalclear9/TimePredictModel.git
cd TimePredictModel
uv sync --locked --python 3.11
```

在运行 SGLang 的 Python 环境中同样安装本包，使补丁能够导入采集适配层：

```bash
python -m pip install /path/to/TimePredictModel
```

### 2. 应用固定版本补丁

```bash
git -C /path/to/sglang checkout 0bcd822377da7b5718e674eaf9c870d349424dd1
git -C /path/to/sglang apply --check /path/to/TimePredictModel/patches/sglang-v0.5.19.patch
git -C /path/to/sglang apply /path/to/TimePredictModel/patches/sglang-v0.5.19.patch
```

补丁接入 `http_server.py`、`serving_base.py` 和 `serving_chat.py`。每个 SGLang 副本使用一个 HTTP/tokenizer worker；多入口通过独立副本部署，副本内部仍可使用 tensor parallel 多 GPU。

### 3. 配置认证与共享状态

SGLang 入口需要以下环境变量：

```bash
export TIMEPREDICT_REDIS_URL='redis://private-redis:6379/0'
export TIMEPREDICT_PREFIX='timepredict'
export TIMEPREDICT_SOURCE='ingress-host-01'
export SGLANG_API_KEY='replace-with-your-sglang-key'
```

同一业务命名空间的入口使用相同 SGLang key、Redis 地址和 prefix；不同入口使用可区分的 source 名。服务端根据已认证 key 的 SHA-256 确定 tenant。密钥轮换会改变该命名空间，应在旧轮次结束后切换。

预测服务通过 JSON 文件将内部访问令牌映射到 tenant，例如 `/etc/timepredict/tokens.json`：

```json
{
  "replace-with-your-internal-token": "sha256-of-your-sglang-api-key"
}
```

右侧填写 SGLang key 的实际 SHA-256 十六进制摘要。可在已设置 `SGLANG_API_KEY` 的 shell 中计算：

```bash
python -c 'import hashlib, os; print(hashlib.sha256(os.environ["SGLANG_API_KEY"].encode()).hexdigest())'
```

### 4. 启动采集链路

在 SGLang 环境启动入口：

```bash
python -m sglang.launch_server \
  --model-path /models/tool-capable-model \
  --tool-call-parser your-model-parser \
  --api-key "$SGLANG_API_KEY" \
  --host 0.0.0.0 --port 30000
```

在本项目环境启动预测服务。首次采集可以不提供模型，关联与标签采集仍正常运行：

```bash
uv run timepredict serve \
  --redis-url "$TIMEPREDICT_REDIS_URL" --prefix timepredict \
  --tokens-file /etc/timepredict/tokens.json \
  --host 0.0.0.0 --port 8098 --window-h 2
```

若未指定 `--host`，预测服务默认仅监听 `127.0.0.1`。跨机器部署需要显式配置监听地址，并确保入口和预测服务都能读取已同步的 chrony 状态。

运行 Agent 示例：

```bash
export TIMEPREDICT_API_KEY='replace-with-your-internal-token'
uv run python examples/agent_client.py \
  --model your-model --sglang http://ingress:30000 \
  --predictor http://predictor:8098 --stream
```

移除 `--stream` 可切换为非流式。示例使用固定演示执行器，只用于说明接入方式；采集训练数据时应替换为真实工具。Agent 等待期间每 5 秒发送心跳，工作流完成发送 `terminate`，异常结束发送 `cancel` 和 `terminate`。

## 数据准备与训练

训练需要**真实测量且正确关联的工具回返时间**。只有工具调用文本、没有相应延迟标签的公开语料，不能直接作为本项目的有标签训练集。样本字段、删失规则与采集建议见 [训练数据指南](DATASET.md)。

### 公开数据集初步实验

准确率采用 `abs(P50 - T) / T <= tolerance` 的命中率，默认重点查看 20% 容限，另报告 10% 和 50%。该指标只用于精确标签；含删失样本时命令会拒绝输出这种简单命中率。公开诊断模型已清理，不再作为当前结果展示。

```bash
python -m timepredict.accuracy --parquet /path/to/agenttrace.parquet \
  --model /external/diagnostics/public-proxy-model --output /external/diagnostics
```

此命令校验原始训练样本摘要，使用每轮树前缀恢复训练/验证曲线，测试集只展示原先选定轮次的结果。它没有重新训练或修复旧模型的参数丢失问题。

仓库提供固定 revision 的受控实验入口。除原有的 `Exgentic/agent-llm-traces`、`pagarsky/agent-trace` 外，还接入了时间边界更明确的 `trace-commons/agent-traces`（`tool_use`→`tool_result`）和 `ProCreations/betterwright-agent-traces`（浏览器步骤 `ms` 遥测）。它们分别标记为 `proxy_agent_tool_return`、`proxy_tool_duration` 和 `proxy_browser_step_duration`，不会混合比较不同标签定义。公开数据集仍没有可证明的 SGLang `t_out` 与 Agent 回返边界，因此不可直接发布到生产服务。

此前 Betterwright 等公开代理实验已经清理；它们的标签定义不满足生产目标，因此不再保留模型结果。

### 可控回放与直接回归验证

当前主验证使用内置工具的实测 HTTP 回放，避免把公开数据中的错误时间边界当作 SGLang 标签。回放只验证参数结构是否能学习耗时关系，不代表服务器性能：

```bash
uv run timepredict replay-generate --output /external/timepredict-replay \
  --rows 5000 --seed 42 --concurrency 32
uv run timepredict train-regression \
  --data /external/timepredict-replay/samples.jsonl \
  --output outputs/replay-model --rounds 500
uv run timepredict evaluate \
  --model outputs/replay-model \
  --data /external/timepredict-replay/test.jsonl
```

WSL/Linux 源码验证使用 `bash scripts/run_wsl_tests.sh`。脚本固定 `PYTHONPATH=src` 并关闭系统 pytest 插件自动加载，避免 Windows 全局 Hydra/ANTLR 插件污染测试环境。

回放模型使用 XGBoost 分位数回归预测 `log(T)`，独立输出 P50/P90 和逐轮相对误差准确率曲线。测试集 P50 ≤10% 准确率必须达到 60% 且优于基线，否则报告为 `failed_acceptance`。模型包默认 `production_releasable=false`，只有真实 SGLang 采集标签才能进入生产发布。

当前重做后的固定种子回放结果：5,000 条调用、500 个 workflow、750 条测试样本，P50 相对误差 ≤10% 为 **96.93%**，≤20% 为 **99.87%**，≤50% 为 **100%**；P50/P90 覆盖率分别为 **52.67%/87.73%**，全局中位数基线的 ≤10% 命中率为 **16.67%**。语义 embedding 已排除具体参数值，仅保留操作契约结构。这是跨 workflow 的可控回放契约验证结果，不是生产服务器精度。模型包标记 `production_releasable=false`，因为真实 SGLang 标签链路尚未完成。

独立泛化测试使用 `trace-commons/agent-traces` revision `112ebd4d03ce852b00e935d523107c3d0c9a65bf`。该数据集按 `tool_use`/`tool_result` ID 配对，标签为 `proxy_agent_tool_return`，不是正式 SGLang 标签。抽取 3,000 条、19 个 workflow 后，严格 workflow 划分为 13/2/3 个 workflow；测试集 P50 ≤10% 命中率为 **2.70%**，全局中位数为 2.70%，按工具中位数为 3.60%，实验判定 `failed_acceptance`。这说明当前回放高分不能外推到跨会话 Agent 轨迹，也说明该公开数据的 workflow 数量不足以作为稳定生产训练集。

精简报告见 `outputs/tracecommons-evaluation.json`；该实验模型保存在外部临时目录，不进入正式模型包。

在 WSL/Linux 的实验环境中安装可选依赖并运行：

```bash
uv pip install -e '.[experiments]'
uv run timepredict experiment \
  --output /external/diagnostics/public-proxy \
  --cache-dir /tmp/timepredict-hf-cache \
  --threads 4 --rounds 500
```

如果使用已经下载到本地的 Parquet 文件，必须先生成并校验固定 revision/hash 清单：

```bash
uv run timepredict cache-manifest --cache-dir /tmp/timepredict-hf-cache --dataset agenttrace
```

公开数据入口只生成独立诊断报告，不保留重复模型、原始 Parquet 或中间 checkpoint。

### 导出、截取与校验

在事件已进入历史流后导出数据，选择明确的 UTC Unix 秒作为截取边界：

```bash
uv run timepredict export \
  --redis-url "$TIMEPREDICT_REDIS_URL" --prefix timepredict \
  --output events.jsonl

# 将占位值替换为本次数据快照的实际截取时间。
uv run timepredict snapshot \
  --events events.jsonl --cutoff ACTUAL_UTC_UNIX_SECONDS \
  --output samples.jsonl

uv run timepredict check-data --data samples.jsonl
```

`check-data` 检查标签、重复样本、workflow 元数据和时间划分，输出各集合的精确/删失样本数及数据摘要。真实回返标签为 `[T,T]`；截取时仍被完整观察的等待记录为 `[a,null]`，其中 `null` 表示右删失上界。取消、失联和采集缺口不作为普通删失样本。

### 训练候选模型

```bash
uv run timepredict train --data samples.jsonl --output candidates \
  --threads 16 --rounds 500 --data-kind observed
```

线程数由训练服务器资源配额决定。`--data-kind` 应如实标记数据来源；合成数据使用 `synthetic`，不能通过正式发布门禁。

| 训练配置 | 默认值或规则 |
|---|---|
| 目标函数 / 分布 | `survival:aft` / `normal` |
| 树方法 / 设备 | `hist` / CPU |
| 最大深度 / 学习率 | 4 / 0.05 |
| 训练轮数 / 提前停止 | 默认最多 500 轮 / 30 轮 |
| σ 候选 | 0.25、0.5、1、1.5、2 |
| 模型选择 | 同标签验证集：P50 相对误差/log-MAE，P90 pinball loss；测试集只在最终轮次使用 |
| 数据划分 | 按 workflow 开始时间约 70% / 15% / 15%；跨分界工作流隔离 |
| 特征词典 | 仅使用训练集建立，最多 4096 个结构字段，在线冻结 |

训练产物位于 `candidates/<version>/`，包含树模型、特征词典、σ、最佳迭代、依赖版本、校验摘要和评估报告。报告覆盖删失似然、分位数覆盖率、窗口校准及经验基线比较。整工具留出实验可通过 `--holdout-tool TOOL_NAME` 单独运行。

## 模型发布

模型生命周期分为**候选训练、影子验收、正式发布**三个阶段。训练不会修改正式服务的 `CURRENT` 指针。

### 影子运行

在独立验收部署运行候选模型。验收 SGLang 入口与预测服务必须使用同一个验收 prefix；同一 prefix 的 worker 共享选主租约，不应用于同时运行不同模型。

```bash
uv run timepredict serve \
  --redis-url "$TIMEPREDICT_REDIS_URL" --prefix timepredict-acceptance \
  --tokens-file /etc/timepredict/tokens.json \
  --host 0.0.0.0 --port 8098 --window-h 2 \
  --shadow-candidate candidates/VERSION
```

影子模式计算分布，但返回 `shadow=true`、`in_window=false`，不会生成可执行的入窗候选。

### 门禁与正式加载

完成服务器验收后，用对应模型版本的证据执行发布：

```bash
uv run timepredict promote \
  --candidate candidates/VERSION --serving-dir models \
  --evidence server-acceptance.json --policy deploy/gate-policy.json
```

[发布策略](deploy/gate-policy.json) 检查数据来源、删失支持、校准、基线比较、服务器功能和性能证据。阈值属于显式工程策略，应在实验前确定。检查失败不修改 `CURRENT`；通过后保存模型、原始证据和策略，再原子切换版本指针。

正式服务在前述 `serve` 命令上增加 `--models models`。服务每 5 秒检查模型更新，加载失败时保留上次成功版本；已有等待轮次继续使用其缓存的原分布参数。

## 接口与消费约定

所有内部 API 均要求 `Authorization: Bearer <internal-token>`。

| 接口 | 作用 |
|---|---|
| `GET /internal/window?H_seconds=2` | 当前全量快照：业务键、版本、状态、已等待时间、剩余分位数、概率和期望成本 |
| `GET /internal/notices?after=0-0` | 根据 Redis stream cursor 获取候选更新和移除事件 |
| `POST /internal/lifecycle/heartbeat` | 更新 Agent 存活状态；请求体包含 `workflow_id` 和 `agent_id` |
| `POST /internal/lifecycle/cancel` | 取消指定 Agent 的等待轮次；请求体包含 `workflow_id` 和 `agent_id` |
| `POST /internal/lifecycle/terminate` | 终止整个工作流；请求体包含 `workflow_id` |
| `GET /internal/health` | 模型版本、时钟、处理器新鲜度、来源、队列、延迟及 CAS 指标 |

查询示例：

```bash
curl --fail --get 'http://predictor:8098/internal/window' \
  --header "Authorization: Bearer $TIMEPREDICT_API_KEY" \
  --data-urlencode 'H_seconds=2'
```

消费者首先读取全量快照，之后从返回的 `notice_cursor` 继续消费通知。只在 `status=ready`、非影子模式且剩余 P50≤H 时入窗；`unknown` 必须保留为不确定状态。

[CandidateCache](src/timepredict/candidates.py) 提供版本过滤和本地过期处理。调用 `snapshot` / `notice` 时传入实测 `roundtrip_seconds`，扣除网络耗时，并使用单调时间执行 `valid_for_seconds`。每个缓存使用固定 H，通知中的 H 由服务的 `--window-h` 决定。

处理器超过 2 秒未刷新，或来源/Agent 超过 30 秒失去观察时，接口降级为 unknown。消费端不能仅依赖下一条移除通知来撤销过期候选。健康接口的 `degraded` 状态仍使用 HTTP 200，应检查响应体。

## 服务器验收

正式发布前需要在目标部署上验证真实 SGLang 流式/非流式请求、跨入口关联、校时和故障切换，并测量入口开销、预测可用延迟、队列积压与吞吐变化。

| 工具 | 验证内容 |
|---|---|
| [verify_failover.py](deploy/verify_failover.py) | 受控停止 leader，确认备用 worker 接管并继续消费事件 |
| [benchmark_server.py](deploy/benchmark_server.py) | 完整单工具往返、交替入口、闭合验证、延迟与基准吞吐比较 |
| [benchmark_scale.py](deploy/benchmark_scale.py) | 状态规模、CAS 延迟/重试、窗口计算和离线评估资源消耗 |

<details>
<summary>服务器验收命令</summary>

在独立验收部署上执行。故障演练中的停止命令由部署方明确提供，完成后自行恢复旧副本：

```bash
uv run python deploy/verify_failover.py \
  --before http://leader:8098 --after http://standby:8098 \
  --redis-url "$TIMEPREDICT_REDIS_URL" --prefix timepredict-acceptance \
  --stop-command '["ssh","leader-host","sudo","systemctl","stop","timepredict-acceptance"]' \
  --output failover.json
```

切换完成后，压测地址指向新的 leader：

```bash
uv run python deploy/benchmark_server.py \
  --baseline http://baseline-a:30000 http://baseline-b:30000 \
  --instrumented http://instrumented-a:30000 http://instrumented-b:30000 \
  --predictor http://current-leader:8098 \
  --redis-url "$TIMEPREDICT_REDIS_URL" --prefix timepredict-acceptance \
  --failover-evidence failover.json --requests workload.jsonl \
  --output server-acceptance.json
```

`workload.jsonl` 每行是能产生一个工具调用的 OpenAI chat 请求。两组部署使用相同硬件、模型、入口数量和请求参数。脚本读取 `SGLANG_API_KEY`、`TIMEPREDICT_API_KEY`；工具执行使用固定延时与结果，因此输出只作为验收证据，不作为训练数据。

规模预算由部署方确定，以下数值仅为命令示例：

```bash
uv run python deploy/benchmark_scale.py \
  --redis-url "$TIMEPREDICT_REDIS_URL" \
  --sizes 100 1000 5000 --events 1000 --training-rows 100000 \
  --max-mutate-p99 0.1 --max-state-bytes 64000000 \
  --max-window-seconds 0.1 --output scale-acceptance.json
```

</details>

首版性能目标为入口新增 P99≤2 ms、吞吐下降≤1%。这些是待验证目标，不是已测得的结果。滚动直方图应在独立验收副本上采集；容量评估还需覆盖真实参数大小和保留周期内的状态积累。

## 运行边界与验证状态

| 方面 | 当前约定 |
|---|---|
| 故障降级 | Redis 或预测服务不可用不阻塞 SGLang 正常处理请求；采集缺口单独记录 |
| 原子性 | 每个 prefix 由租约选举 worker；状态、版本和通知通过 Redis CAS 更新，旧持有者无法继续写入 |
| 计时 | 同启动进程优先单调时间；跨机器使用 UTC，校时误差之和不超过 5 ms |
| 数据保留 | 幂等范围 7 天，闭合记录热保留 1 小时；事件和历史流不自动裁剪，需归档后管理保留周期 |
| 扩展性 | 每个 namespace 使用一个 CAS 状态文档，当前没有自动分片；支持规模以服务器实测为准 |
| 不支持的输入 | 多工具、参数不完整、超出输入边界、未知工具和不可用时钟不会生成有效预测 |

固定版本的 Qwen parser 在单个增量块包含完整工具对象的边界情况下，可能只输出工具名。适配层将此类不完整参数标记为未覆盖。

当前仓库保留一组精简的确定性契约测试，覆盖串行关联、严格参数拒绝、语义编码稳定性和分布数学边界；真实 Redis 链路、GPU SGLang、跨主机校时、真实数据精度和目标服务器性能仍需在 Linux 服务器上验收。公开数据不能证明 `t_return-t_out`，因此不会把代理工具耗时模型标记为生产可发布，也尚未验证对 KV 决策的实际收益。

语义编码器默认使用 `sentence-transformers/all-MiniLM-L6-v2`，模型缓存通过 `TIMEPREDICT_EMBEDDING_CACHE` 放在仓库之外。训练和生产服务缺少编码器时会明确失败；仅回归测试可显式设置 `TIMEPREDICT_TEST_ENCODER=1` 注入测试替身，测试替身生成的模型不能发布。

源码级 WSL 验证入口：

```bash
PYTHONPATH=src python scripts/verify_wsl.py
python -m timepredict audit --output outputs/audit.json --evidence outputs/verification-evidence.json
```

验证报告会区分 `verified_in_wsl`、`not_verified_server` 和 `missing`；真实 Sentence Transformer、Redis、GPU SGLang 和服务器性能只有在对应服务实际运行时才会标记通过。

## 仓库结构

```text
TimePredictModel/
├── src/timepredict/       # 采集、状态、特征、训练、预测与发布
├── patches/              # 固定版本 SGLang 补丁及上游摘要
├── examples/             # Agent 接入示例
├── deploy/               # 服务配置、训练入口和验收工具
├── DATASET.md            # 数据采集与样本格式
├── pyproject.toml        # 包配置与 CLI 入口
└── uv.lock               # 生产依赖锁定
```

部署模板包括 [systemd 服务](deploy/timepredict.service)、[Redis 配置](deploy/redis.conf)、[训练入口](deploy/train.sh) 和 [Dockerfile](deploy/Dockerfile)。模板中的路径、账户及资源参数应按实际服务器配置。
