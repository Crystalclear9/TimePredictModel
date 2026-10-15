# 实现与验证记录

复核日期：2026-10-15。范围为独立时间预测插件；外部 C2 密度、KV 映射及加载策略由接入方实现。

功能主线已实现，本地测试通过。现有证据不代表零缺陷或生产就绪，真实环境未验证项单独列出。

## 要求到实现的映射

| 要求 | 实现 | 本地检查 |
|---|---|---|
| 服务端发出到续接入口的时间标签 | sglang.py / clock.py | 流式/非流式边界、发送失败、工具片段拼接 |
| 单工具串行、已有 ID 优先、重试幂等 | state.py | 两/三轮全部 24/720 种事件排列，有/无 ID、重复历史、最早入口 |
| 取消/终止/失联独立分类 | state.py | 终止及重复取消排列、迟到终止、旧取消不影响新轮次 |
| t_out 时冻结工具名/描述/实际参数 | Capture / features.py | 定义关联、身份及后续结果不进入向量 |
| 结构化数值、类型、路径和未知工具列 | FeatureSpec | JSON Pointer、NaN 缺失、schema、固定维度、未知工具 |
| 冻结本地联合语义编码器及增强对照 | semantic.py / pipeline.py | 真实参数、规范顺序、并发共享、相同数据/预算和缓存计时；编码质量未实测 |
| XGBoost normal AFT 和验证集统一 sigma | training.py / model.py | 实际 CPU 训练、原始 margin、秒单位恢复及独立加载 |
| 精确和行政右删失标签 | snapshot / dataset.py | 事件摘要、身份/边界、匹配来源与 Agent 心跳、cutoff 时钟 |
| 时间与完整 workflow 隔离、整工具留出 | split_rows / train | 工作流不交叉，留出工作流不进入词典或训练 |
| 条件剩余 P50/P90、概率、均值及截断期望 | distribution.py | 48 组参数与独立 SciPy 生存函数、反函数、积分对照 |
| 预测一次、缓存版本、等待时条件化 | Worker / WindowConsumer | 异步预测、迟到结果拒绝、age 更新、断网租约过期 |
| 校准和窗口规则、unknown/影子不可加载 | service.py / client.py | 鉴权、租户隔离、状态降级、概率更新和候选判定 |
| 特征与模型一起发布、门禁和原子切换 | model.py / release.py | 摘要、归档证据、CURRENT、提交前训练租约检查 |
| 独立周期重训 | pipeline.py / timer | 固定历史边界、续租、空数据等待、候选完整性复用和损坏恢复 |
| 固定版本 SGLang 适配 | patches / verify_sglang_patch.py | 三个上游摘要、git apply 检查与实际应用、修改后语法检查 |
| 目标容量探针 | deploy/benchmark_scale.py | 模拟状态进入 ready 后运行分布计算；临时 Redis prefix 清理 |
| 外部 C2/KV 接入 | client.py / examples | [API 契约](integration.md) 提供时间分布和驻留成本输入 |

## 本轮修复

1. SSE 中相同的合法名称片段被错误去重，工具名与定义关联会出错。现在按协议逐片拼接。
2. 普通 OUT 被当作来源存活依据时，snapshot 会导出缺少 source_heartbeat 的删失样本，随后导致正式训练拒绝整批数据。现在缺少匹配心跳时单独排除，并检查心跳 TTL 有限。
3. 重训只比较数据/配置，候选包丢失、损坏或放错版本目录也会被当作可复用。现在先校验候选和版本，必要时重新训练；相同版本的损坏包先隔离为 `.invalid-*`，再重建。缺失、损坏和版本不匹配恢复均有回归检查。
4. 容量脚本的模拟预测缺少校准标记，实际一直降级 unknown。现在明确以测试分布执行 ready 数值路径，证据类型仍为 scale_acceptance，不能用于正式发布。

此前修复的跨轮次乱序、取消时序、错误工具 ID、严格 JSON、Windows 高精度时钟、UTF-8 模型文件、版本/租约和语义缓存并发回归仍保留。

## 清理范围

仓库保留插件主线、可选语义对照、AFT 兼容加载、测试、示例、补丁和部署/验收材料。移除旧代理数据适配、独立直接回归、合成回放、旧绘图和重复审计/WSL 脚本，相关命令与额外依赖一起移除。旧直接回归包给出明确错误，不再尝试导入已删除模块。

文档统一为 README 和 docs 下的数据、接入、验证三份；旧实验产物、重复审计结果、构建文件及缓存不作为交付材料。运行环境 `.venv/` 保留且不提交。历史实验代码可从 Git 历史取回。

## 可复现验证

```bash
uv sync --extra test
uv run --extra test python scripts/verify_plugin.py
uv run python scripts/verify_sglang_patch.py
uv run python -m compileall -q src tests scripts deploy examples
uv lock --check
```

当前 154 项本地测试通过。测试证据绑定源码 SHA256，写入 `outputs/plugin-verification.json`，明确包含 `server_acceptance_verified=false` 和 `production_publish_allowed=false`。Redis 事务、流和租约使用 fakeredis/Lua；发布验收对象是明确的模拟夹具。容量探针的本地测试也使用 fakeredis，并模拟内存查询。

补丁检查下载固定 commit 的三个原始文件，核对 SHA256、实际应用补丁并验证语法。它不导入 GPU 服务。构建检查另验证 wheel/sdist 和清理后的包入口。

## 仍需目标环境证据

真实 Redis 持久化及故障恢复、GPU SGLang、跨机器校时、真实语义编码器、代表性任务上的分位数/窗口校准、P99/吞吐，以及接入后的真实 KV 收益均未完成目标环境验收。服务器脚本和单元测试只是验证手段，不是已经取得的生产证据。

运行流程见 [README](../README.md)，数据约束见 [数据准备](data.md)，接口约束见 [接入契约](integration.md)。
