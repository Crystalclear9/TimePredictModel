# 故障恢复与容量管理

这些流程管理插件内部观察状态，不实现外部C2密度或KV加载。真实Redis、磁盘和目标负载仍需验收。

## 健康与就绪

`GET /internal/health`使用租户令牌。Redis和状态可读时返回HTTP 200，`live=true`只表示服务可响应。`ready`和`readiness_reasons`综合处理器心跳、活动来源、校时、模型校准/过期、消费错误及未恢复的传输缺口。

负载均衡就绪探针使用`GET /internal/ready`：就绪为200，否则503。影子模式、没有模型、证据过期、所有来源退出都不就绪；影子实验仍可读取健康和窗口接口。只读API副本读取leader写入的模型元数据。

`model_unavailable`、`model_acceptance_expired`及`prediction_dependency_unavailable`可恢复。仍等待的轮次按2、4、8、16、30秒上限退避重试；ready结果保持原模型版本，闭合记录和永久不支持输入不重新预测。一个预测任务内的依赖重试固定使用开始时捕获的模型。

## 独立管理员权限

默认关闭管理接口。添加`serve --admin-tokens-file /etc/timepredict/admin-tokens.json`开启；文件是独立ASCII令牌数组，例如`["REPLACE_WITH_OPERATOR_SECRET"]`，不能与租户令牌重合，也不提交仓库。租户令牌不能调用下述管理接口。

管理操作向同一事件流入队，`accepted=true`不表示已归约完成。提交端时钟必须未跳变、误差不超过0.0025秒。管理员确认的是外部事实；插件不能自行证明路由隔离或原始日志完整。

`routing_fenced`与`observation_reestablished`必须是JSON布尔值`true`，数字1、1.0、字符串及false/null均返回422，不做真假值转换。

## 永久下线采集副本

先从路由移除目标副本，排空/阻断其入口，确认它不会再接收续接请求。取得健康响应中的完整source/boot键及最新seq，用管理员令牌POST `/internal/admin/retire-source`：

```json
{"source_key":"replica-A/BOOT","expected_seq":123,"routing_fenced":true,"reason":"入口已从路由移除并停止接收请求"}
```

未知来源或提交前序号变化返回409；排队期间序号变化时reducer拒绝应用。应核对健康状态的`retired`和`failure_until`，重新完成隔离检查，不要盲目重试。

退出保留原故障区间，只有能证明OUT在退出边界之后的新调用可恢复。旧调用不自动放行。退出boot若在边界之后再次发事件，会撤回结束边界并标记`source_retirement_violated`，必须重新隔离。可信顺序无法证明时保持降级。

## 坏消息隔离与恢复

消费者校验对象、kind、身份、时钟、seq、协议标志及关键字段，拒绝非有限JSON、过深结构和超过1MiB的解析输入。坏消息写入`PREFIX:deadletters`，含原始字段、stream ID和原因；同一租约事务还记录`transport_gap`及历史，成功后才ACK。重复投递不重复隔离，后续有效消息继续处理。

事件信封各层对象禁止重复JSON键，解析后的全部键和值必须可编码为UTF-8；不能用转义隐藏孤立代理码点。此规则检查事件JSON本身，arguments若为JSON字符串仍由特征层另行检查。额外验证包含一次严格重编码，开销需纳入目标环境测量。

无法识别来源的坏消息可能包含遗漏续接，缺口未恢复时全局候选不可执行。操作者核查死信和原始日志、修复发送端并确认完整采集恢复后，POST `/internal/admin/recover-transport`：

```json
{"gap_id":"原坏消息的Redis-stream-ID","observation_reestablished":true,"reason":"已核对原始日志并恢复完整采集"}
```

健康响应提供`transport_gaps`。恢复只放行新边界后的调用，旧歧义标签继续排除；若来源seq也有缺口，还须完成来源重启/退出流程。死信含调用参数，按原始日志的权限和保留规则管理，不能把入死信当作观察恢复。

## 容量预算

| 预算 | 默认 | 超限行为 |
|---|---|---|
| notices，`serve --notice-limit` | 100000条 | 保留最近通知；落后游标要求全量同步 |
| 状态JSON，`serve --max-state-bytes` | 64MiB | 拒绝事务，不部分写入；消费告警，候选按租约过期 |
| deadletters，`serve --deadletter-limit` | 10000条 | 保留pending并停止隔离新消息，不淘汰未核查证据 |
| 重训/导出历史 | 100000事件、128MiB输入 | `history_capacity_exceeded`，不静默截断工作流 |

重训/导出用`--max-history-events`、`--max-history-bytes`配置预算。事件包含心跳，不等于训练样本；反序列化、特征矩阵和XGBoost还需要额外内存。预算不是容量或吞吐承诺。

Worker每0.1秒检查预测任务，心跳/窗口通知批次最多每0.5秒刷新一次。状态仍采用单JSON、单leader CAS；更大规模需分区或增量回放。达到预算应告警并核查、扩容或迁移，不能直接删除观察证据。

## 通知重同步

先取得快照和`notice_cursor`。`/internal/notices`发现非零游标早于最老保留通知时返回`resync_required=true`及空events。`0-0`只供初始读取，不作为长期恢复游标。

使用`WindowConsumer.notice_page()`处理整页；重同步时清空候选并拒绝增量更新，直到接受新快照。`PredictionClient.poll_notices()`自动处理；快照拉取失败时仍不放行已清空候选。周期全量快照仍有必要，不要逐条转发而丢掉页面状态。

## 历史归档与裁剪

在持久、受控、可备份目录试运行：

```bash
timepredict archive-history --archive-dir /var/lib/timepredict/archives --older-than-days 7
timepredict retrain --output /var/lib/timepredict/training --archive-dir /var/lib/timepredict/archives
timepredict export --output events.jsonl --archive-dir /var/lib/timepredict/archives
```

归档按历史stream ID顺序生成不可变JSONL段，记录SHA256、数量、首尾ID。文件fsync、重新校验、索引原子发布及目录同步成功后，才允许可选`--trim-history`。Windows不具备POSIX目录同步证据。`--trim-events`只裁剪predictor已消费范围，保留最早pending和未投递消息；存在其他消费者组时拒绝该步骤。

每次维护使用独立命名的不可变索引，Redis标记中的`index_file`和摘要共同指定已提交版本。读取和维护都从标记定位索引，摘要针对实际解析的同一份字节校验；维护不能重新认可一个已被修改的目录。失去租约的旧任务即使完成本地写盘，也不能覆盖当前索引或更新标记。

旧格式标记没有`index_file`时，只在固定`archive-index.json`与原摘要匹配后读取；下一次成功维护迁移到不可变索引。损坏标记或索引必须从已核对备份恢复，不能通过再次维护自动重签。未提交索引/段可保留用于诊断，不可代替Redis标记指向的版本。

迁移前停用旧归档timer并停止所有旧归档进程，统一升级写入端后再恢复维护。旧版维护命令不认识新索引指针，不能与新版混跑；已经发生的索引损坏不会自动修复，须核对Redis标记和备份恢复。

归档有续租锁，裁剪检查租约和stream变更。写盘失败或租约丢失不会依据未发布归档裁剪。若raw events步骤失败，历史维护可能已完成；保留目录并核查后重跑。失败生成的孤立段不进入索引，可核对后整理，不能删除索引引用的段。

发布Redis归档标记后，重训/导出必须提供相同目录及匹配索引，校验后合并Redis剩余历史、排除重叠ID。缺失/篡改归档、索引不匹配、回放期间归档变更均拒绝，不能退回残余数据。归档须长期保留：它降低Redis驻留量，不减少全量训练数据或消除回放预算。死信不由此命令自动删除，需核查并另行保全。

每日模板为`deploy/timepredict-archive.service`及timer，不自动安装或启用。模板开启裁剪，启用前配置目录权限、备份和重训读取同一目录。历史归档不清空在线来源故障区间。

## 升级边界

旧版正式包若未开启留出要求，仍可加载，但未验收的工具名称不再参与实际调度。旧策略已要求留出、却缺少逐工具报告或NLL上限时，须重新训练/验收并发布。legacy动作族映射不等于具体工具名的验收证据，也受新规则约束。API v1保留既有字段并增加管理、就绪及重同步能力；已有通知消费者应升级整页处理，旧版消费者不能忽略裁剪后重同步要求。
