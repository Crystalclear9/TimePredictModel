# 实现与验证记录

复核日期：2026-10-15。交付范围为独立时间预测插件；外部C2密度、KV映射及加载策略由接入方实现。原始需求的收敛版本采用单工具串行、结构化GBDT/AFT基线及可选联合语义对照。

当前源码与独立安装的wheel均通过296项本地测试，未将同一套用例重复计数。通过范围支持接口开发与联调，不代表零缺陷或生产就绪。以往逐轮复核记录可从Git历史查阅，本文件保留当前结论和必要限制。

## 要求到实现与证据

| 要求 | 实现 | 本地检查 |
|---|---|---|
| 服务端发送到续接入口的标签 | sglang.py / clock.py | 流式/非流式、完整DONE行、参数中同名文本、发送失败、片段拼接 |
| 单工具串行、已有ID优先、重试幂等 | state.py | 两/三轮全部24/720种事件排列，有/无ID、重复历史、最早入口纠正 |
| 取消/终止/失联分别处理 | state.py / freshness.py | 迟到生命周期、跨入口缺口、故障时间边界及保守降级 |
| 冻结调用及自动结构特征 | Capture / FeatureSpec | 实际参数、路径/类型/枚举、缺失NaN、未知工具、身份和标签不进入特征 |
| 可选联合语义对照 | semantic.py / pipeline.py | 名称/描述/实参联合编码、规范顺序、缓存并发、相同划分与训练预算 |
| normal AFT与验证集统一sigma | training.py / model.py | 真实CPU训练、原始margin、秒单位恢复、最佳迭代及独立加载 |
| 精确/行政右删失标签 | snapshot / dataset.py | 事件摘要、身份/时钟、匹配来源和Agent心跳、cutoff、已结束workflow排除 |
| 工作流与时间隔离、整工具留出 | split_rows / train | 元数据一致、最晚观察边界、跨界隔离、留出工作流不进入训练词典 |
| 条件剩余分布与驻留成本 | LogNormal | 基础参数组、微秒尺度、320组跨尺度独立正态密度积分对照 |
| 缓存、版本及本地有效期 | Worker / WindowConsumer | 迟到预测拒绝、分页重放、对象副本、age推进、慢计算、输出前过期、千轮通知的旧计时释放和乱序移除 |
| API隔离及unknown/影子规则 | service.py / client.py | 令牌绑定、租户隔离、错误游标、Redis读取故障、非执行状态过滤 |
| 完整模型包与质量门禁 | model.py / release.py | 必需摘要、目标/参数/报告一致、归档证据、原子CURRENT、存储故障注入 |
| 独立周期重训和可靠复用 | pipeline.py / timer | 固定历史边界、续租/取消、损坏候选/收据恢复、数据/配置/实现签名匹配 |
| 来源身份与worker接管 | state.py / store.py / Worker | 完整ID、boot判序、重叠降级、1/101条pending优先处理、失去租约禁止提交 |
| Agent示例与外部调度接入 | examples / client.py | 模拟HTTP正常续接、工具ID、截断响应不执行工具、取消与结束 |
| 固定SGLang适配和容量探针 | patches / deploy | 固定上游摘要、实际应用补丁与语法；模拟ready分布计算和隔离prefix清理 |

## 本轮发现与修复

新增6项测试，其中2项在修复前失败，其余用于保护乱序与租约语义：

1. 连续接收1000轮通知、尚未读取候选时，旧版保留1000份计时数据；移除后也需等待读取才能清理。现在只保留当前候选的计时数据，替换和移除立即释放。
2. 事件中存在超过JSON编码器递归限制的嵌套时，RecursionError会终止发送任务，后续心跳无法送出。现在与其他序列化失败一致：丢弃并计数、保留seq缺口、继续发送；相关观察仍按缺口规则失效，不将失败事件当作成功采集。

附加检查覆盖：不同轮次的较新tombstone不遮蔽当前轮次的有效移除；旧/重复通知不延长租约；内部update阻挡迟到候选；全量快照清理版本索引后旧通知仍不能复活候选。

## 冗余清理

删除消费端重复的逐generation版本字典，统一使用业务键版本上界及当前候选自身revision判断。避免固定业务键的版本索引随轮次增长，也移除了SDK依赖该冗余字典判断接收结果的路径。不同业务键的版本索引仍需周期全量快照回收，不能宣称只消费通知时内存完全有界。

此前已清理的无调用函数、未用导入、重复分支和best-model软链接不再新增。ActionFamilies、旧AFT读取兼容路径，以及用于故障区间判断的来源历史仍有用途，予以保留。没有为减少文件数而删除有效测试、部署脚本或必需模块；历史复核记录保留于Git。

## 故障与迁移边界

- 任一入口缺口都可能遗漏其他入口的续接。按故障区间和两端时钟误差保守排除相关候选/标签；恢复不自动放行旧调用。
- 同一逻辑入口的boot必须能够按可信区间排序；未知时钟、重叠或同一时刻无法判序时降级。稳定入口ID不能同时分配给两个存活副本。
- Redis事务前检查history/notices类型并WATCH流键和leader租约，避免类型错误导致部分写入。接管时先处理pending再读新消息；所有worker须遵守同一选主及写入路径。
- 修复不自动补回旧版缺失历史或清除错误bad状态。需核对原始events/归档及来源seq；旧history也可能已乱序，不能直接假设重放即可恢复。
- 发布前同步失败不切换CURRENT；指针替换后目录同步失败报告publication_outcome_uncertain，不能假定仍使用旧模型。取消异步任务不能强制停止训练线程，正式提交另受取消和租约检查保护。
- 摘要用于一致性检查，数据、模型包、报告和服务器证据仍须来自可信来源。
- 重训复用继续核对数据、训练配置、源码/依赖签名及候选报告；坏收据保留为`.invalid-last-run-*.json`后重训。旧报告缺少复用字段时重新训练，既有受支持AFT包仍可加载。training_signature不是来源认证，也不替代特征迁移或发布门禁。

## 可复现验证

```bash
uv sync --extra test
uv run --extra test python scripts/verify_plugin.py
uv run python scripts/verify_sglang_patch.py
uv run python -m compileall -q src tests scripts deploy examples
uv lock --check
```

本地证据写入`outputs/plugin-verification.json`并绑定源码SHA256，明确包含`server_acceptance_verified=false`和`production_publish_allowed=false`。Redis流、事务与租约使用fakeredis/Lua；发布门禁使用明确模拟验收夹具；Agent示例使用httpx.MockTransport。POSIX同步故障是模拟系统调用，Windows未验证目录fsync。

补丁检查下载固定commit的三个上游文件，核对摘要、实际应用补丁并检查语法，不导入GPU服务。构建检查包含wheel/sdist；独立wheel环境从site-packages导入插件后运行同一套296项测试，示例文件仍由工作区提供。构建及缓存放在临时目录，不作为源代码提交。

## 尚需目标环境验收

真实Redis持久化与进程故障恢复、GPU SGLang、跨主机校时、真实语义编码器质量、代表性日志上的长尾/分位数/窗口校准、P99/吞吐，以及外部C2/KV接入后的实际收益，均未取得完整目标环境证据。固定延时负载和单元测试不能替代这些实验。

运行流程见[README](../README.md)，数据约束见[数据准备](data.md)，接口及后续开发见[接入契约](integration.md#后续开发交接)。
