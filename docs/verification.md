# 实现与验证记录

复核日期：2026-10-15。交付范围为独立时间预测插件；外部C2密度、KV映射及加载策略由接入方实现。原始需求的收敛版本采用单工具串行、结构化GBDT/AFT基线及可选联合语义对照。

当前源码与独立安装的wheel均通过333项本地测试，未将同一套用例重复计数。通过范围支持接口开发与联调，不代表零缺陷或生产就绪。以往逐轮复核记录可从Git历史查阅，本文件保留当前结论和必要限制。

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

针对审查中确认的6项不足新增37项测试，包括明确模拟的故障注入和实际CPU训练/打包链路：

1. 未知工具不再继承整体校准资格。留出要求检查明确NLL上限、样本/有效样本、窗口/分位数/条件窗口，覆盖整体和每个工具；正式资格仅限已验收名称。包含真实CPU训练、发布后加载及未验收名称拒绝测试。
2. 非法流消息隔离到deadletters，原始字段、缺口状态、历史在租约事务内提交后才ACK；坏消息后有效事件继续处理。错误stream类型、失去租约或死信预算满时不部分提交，也不把遗漏当完整观察。
3. 独立管理员接口记录来源安全退出及transport恢复，校验时钟、seq及外部隔离声明。故障区间内旧调用继续排除；退出来源重新出现会重新阻断。测试不证明真实路由隔离已经完成。
4. 模型/依赖暂时不可用可按2至30秒退避重试；tick在模型恢复后调度，ready和闭合记录不覆盖。
5. health区分live/ready；ready综合模型、来源、校时、消费状态和缺口，专用ready接口返回503。
6. 通知保留上限和SDK重同步、热状态/死信/回放预算、历史归档校验与回读、磁盘失败和租约接管、pending/未消费事件保留。归档不完整或缺失时拒绝部分历史训练，重训的CPU回放移到线程，避免阻塞续租。

数据文件未在本机执行生产裁剪，运维timer未自动启用。真实Redis、故障恢复及容量仍需目标环境验收；单JSON状态和全历史训练并未改成分区/增量架构。

## 冗余清理

统一重训/导出的历史读取路径，移除重复分页逻辑。事件校验和归档分别集中在events.py、retention.py，避免将维护命令或临时审计脚本散落到服务中。通知/处理器批次改为最多每0.5秒一次，预测调度仍每0.1秒检查，降低重复写入。

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

补丁检查下载固定commit的三个上游文件，核对摘要、实际应用补丁并检查语法，不导入GPU服务。构建检查包含wheel/sdist；独立wheel环境从site-packages导入插件后运行同一套333项测试，示例文件仍由工作区提供。构建及缓存放在临时目录，不作为源代码提交。

## 尚需目标环境验收

真实Redis持久化与进程故障恢复、GPU SGLang、跨主机校时、真实语义编码器质量、代表性日志上的长尾/分位数/窗口校准、P99/吞吐，以及外部C2/KV接入后的实际收益，均未取得完整目标环境证据。固定延时负载和单元测试不能替代这些实验。

运行流程见[README](../README.md)，数据约束见[数据准备](data.md)，接口及后续开发见[接入契约](integration.md#后续开发交接)。
