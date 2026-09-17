# SGLang 工具回返时间预测器

面向 Linux 服务器的结构化特征 + XGBoost AFT 预测器。固定适配 SGLang v0.5.19，commit `0bcd822377da7b5718e674eaf9c870d349424dd1`。本地只在 WSL 验证，不按本机硬件调优。

预测目标是完整工具调用提交发送至下一轮工具结果请求进入 SGLang 的时间，包含工具、网络和 Agent 编排耗时。独立 CPU 服务输出条件剩余分布与 C2 窗口候选，不执行 KV 映射或预加载。

## 精简后的目录

- `src/timepredict/`：采集、关联、特征、数据校验、训练、预测、发布源码。
- `patches/`：固定版本 SGLang 补丁与上游校验摘要。
- `examples/agent_client.py`：流式/非流式 Agent 采集示例。
- `deploy/`：服务配置、训练入口、发布策略及服务器验收脚本。
- `DATASET.md`：真实训练数据的采集、格式与检查方法。
- `pyproject.toml`、`uv.lock`：安装配置和生产依赖锁定。

按要求不保留本地测试目录、验证产物、合成模型、下载的上游源码或虚拟环境。清理前 WSL 完成 **70 项测试**，真实 Redis 链路和固定版本补丁检查通过。此记录不代表真实数据精度、GPU 服务或服务器性能已经验收。

## 安装与接入

要求 Linux、Python 3.11+、Redis 7.0+、已同步的 chrony。安装：

```bash
uv sync --locked --python 3.11
# 或在已有服务器 Python 环境中：python -m pip install .
```

在 SGLang 的 Python 环境中也安装本包，然后应用补丁：

```bash
git -C /path/to/sglang checkout 0bcd822377da7b5718e674eaf9c870d349424dd1
git -C /path/to/sglang apply --check /path/to/TimePredictModel/patches/sglang-v0.5.19.patch
git -C /path/to/sglang apply /path/to/TimePredictModel/patches/sglang-v0.5.19.patch
export TIMEPREDICT_REDIS_URL='redis://private-redis:6379/0'
export TIMEPREDICT_PREFIX='timepredict'
export TIMEPREDICT_SOURCE='unique-ingress-hostname'
export SGLANG_API_KEY='deployment-secret'
python -m sglang.launch_server --model-path /models/tool-capable-model \
  --tool-call-parser your-model-parser --api-key "$SGLANG_API_KEY" \
  --host 0.0.0.0 --port 30000
```

每个副本使用一个 HTTP/tokenizer worker；多入口使用独立副本，可在副本内使用 TP 多 GPU。补丁在入口最初计时、认证与校验成功后关联，在完整 JSON 或 SSE `[DONE]` 提交 ASGI send 前计时，发送成功后异步发布；这不是客户端收齐响应的时间。

Agent 发送 Authorization 和 `X-Workflow-ID`、`X-Agent-ID`、`X-Request-ID`。逻辑请求重试保持 Request ID；每个 Agent 同时只有一轮、一个工具，设置 `n=1`、`parallel_tool_calls=false`。收到完整响应后才执行工具。工具结果必须携带原协议 tool_call_id；历史工具结果不会关闭新的轮次。

tenant 是经过服务认证的 SGLang API key 的 SHA-256。创建 `/etc/timepredict/tokens.json`，内容为 `{"internal-token":"sha256-of-sglang-api-key"}`；多入口需使用相同业务 key。预测服务：

```bash
uv run timepredict serve --redis-url "$TIMEPREDICT_REDIS_URL" \
  --tokens-file /etc/timepredict/tokens.json --window-h 2 --port 8098
export TIMEPREDICT_API_KEY='internal-token'
uv run python examples/agent_client.py --model your-model \
  --sglang http://ingress:30000 --predictor http://predictor:8098 --stream
```

首次采集可不指定模型，仍采集真实标签。示例工具是固定演示函数，**必须替换成真实工具执行器才能获得有意义的训练数据**。心跳每 5 秒，workflow 正常完成发送 terminate，异常结束发送 cancel/terminate。

## 训练与发布

先阅读 `DATASET.md`，不要将没有真实延迟标签的工具调用语料当作可直接训练的数据。

```bash
uv run timepredict export --redis-url "$TIMEPREDICT_REDIS_URL" --output events.jsonl
uv run timepredict snapshot --events events.jsonl --cutoff ACTUAL_UTC_UNIX_SECONDS --output samples.jsonl
uv run timepredict check-data --data samples.jsonl
uv run timepredict train --data samples.jsonl --output candidates \
  --threads 16 --rounds 500 --data-kind observed
```

线程数按训练服务器资源配额设置。固定 AFT normal、CPU hist、深度 4、学习率 0.05、最多 500 轮、early stop 30，σ 网格为 0.25/0.5/1/1.5/2。训练只创建不可变候选目录，不更新 CURRENT；测试数据必须标记 synthetic。

影子验收使用独立 SGLang 验收副本和相同独立 prefix，避免与正式服务争抢租约：

```bash
uv run timepredict serve --redis-url "$TIMEPREDICT_REDIS_URL" --prefix timepredict-acceptance \
  --tokens-file /etc/timepredict/tokens.json --window-h 2 \
  --shadow-candidate candidates/VERSION
```

影子结果 `shadow=true`、`in_window=false`。真实数据校准、服务器功能/性能验收后发布：

```bash
uv run timepredict promote --candidate candidates/VERSION --serving-dir models \
  --evidence server-acceptance.json --policy deploy/gate-policy.json
```

门禁要求真实数据、有效删失支持、校准、基线比较及对应模型版本的服务器证据；示例阈值是工程策略，需在实验前确定。拒绝不会修改 CURRENT；批准后复制模型、证据和策略，再原子切换指针。正式 serve 使用 `--models models`，每 5 秒检查更新，已有等待轮次继续使用原分布。未经门禁的旧候选不能直接正式加载。

## API 与运行边界

内部 API 都使用 `Authorization: Bearer <internal-token>`。

| API | 用途 |
|---|---|
| `GET /internal/window?H_seconds=2` | 全量快照、业务键、generation/revision、age、分布、剩余 P50/P90、窗口概率、期望及截断期望成本 |
| `GET /internal/notices?after=0-0` | 版本化候选更新与移除；首次使用快照返回的 notice_cursor |
| `POST /internal/lifecycle/{heartbeat,cancel,terminate}` | JSON 包含 workflow_id；heartbeat/cancel 还要求 agent_id |
| `GET /internal/health` | 时钟、processor、来源、队列、丢弃、延迟分位数、CAS 指标；degraded 仍返回 HTTP 200 |

仅 ready 且剩余 P50≤H 才入窗。剩余时间通过条件生存分布计算，不是初始预测减去等待时间。消费者应使用 `CandidateCache`，给 snapshot/notice 传入实测 roundtrip_seconds，按单调时间执行 TTL。每个缓存使用固定 H；忽略旧版本和 generation。处理器超过 2 秒未刷新、观察超过 30 秒失联时降级 unknown。

Redis Streams 支持重试去重和乱序关联；共享状态使用租约与原子 CAS，旧 worker 不能继续写。预测异常有界重试后降级，原始标签仍采集。Redis 或预测器不可用不阻止 SGLang 请求。队列溢出/来源重启/采集缺口不当成普通删失。

跨机器时钟误差之和≤5 ms，同启动进程优先单调时间；chrony 不可用不伪造误差。状态保留有 7 天幂等范围、闭合记录热保留 1 小时；原始 events/history/notices 不自动裁剪，需要先可靠归档。当前每个 namespace 使用一个 CAS 文档，没有自动分片，容量必须在服务器按真实状态/保留规模验收。

固定上游 Qwen parser 在整个工具对象仅一个增量块的边界可能只输出工具名；本预测器将不完整参数标为不覆盖，不补造预测。

## 正式发布前的服务器验收

以下保留的是部署验收源码，不是本地测试产物。只在独立验收部署执行故障演练，显式指定停止旧 leader 的命令，之后自行恢复该副本：

```bash
uv run python deploy/verify_failover.py --before http://leader:8098 --after http://standby:8098 \
  --redis-url "$TIMEPREDICT_REDIS_URL" --prefix timepredict-acceptance \
  --stop-command '["ssh","leader-host","sudo","systemctl","stop","timepredict-acceptance"]' \
  --output failover.json
uv run python deploy/benchmark_server.py \
  --baseline http://baseline-a:30000 http://baseline-b:30000 \
  --instrumented http://instrumented-a:30000 http://instrumented-b:30000 \
  --predictor http://current-leader:8098 --redis-url "$TIMEPREDICT_REDIS_URL" \
  --prefix timepredict-acceptance --failover-evidence failover.json \
  --requests workload.jsonl --output server-acceptance.json
uv run python deploy/benchmark_scale.py --redis-url "$TIMEPREDICT_REDIS_URL" \
  --sizes 100 1000 5000 --events 1000 --training-rows 100000 \
  --max-mutate-p99 0.1 --max-state-bytes 64000000 --max-window-seconds 0.1 \
  --output scale-acceptance.json
```

workload.jsonl 每行是可产生一个工具调用的 OpenAI chat 请求。压测工具使用固定工具延时/结果，不用于生成训练标签。两组部署保持相同硬件、模型、入口数量及请求参数。检查入口新增 P99≤2 ms、吞吐下降≤1%、预测可用延迟、跨入口真实闭合与校时；采集滚动直方图要求独立验收副本。规模命令中的预算只是示例，需按实际部署确定。
