# 当前代码导航

当前主线为串行单工具、结构化特征、XGBoost AFT、验收发布、条件剩余分布和开放接口。完整使用流程见 README.md，跨模块契约见 INTEGRATION.md。

| 文件 | 职责 |
|---|---|
| sglang.py / patches/ | SGLang采集、流式拼接、入口/发送时间边界 |
| publisher.py | 有界异步采集队列、来源序号和心跳 |
| state.py | prepared/out/ingress、幂等关联、冲突和生命周期 |
| store.py | Redis Streams及WATCH/MULTI状态历史通知事务 |
| clock.py / freshness.py | 单调时间、跨来源时钟误差、观察租约 |
| features.py | 结构化基线和冻结字段词典，语义增强可选 |
| semantic.py | 本地固定编码器；joint_call_text保留实际参数 |
| dataset.py | 精确/删失标签、原始事件和身份/边界一致性 |
| training.py | 可信事件快照、工作流时间划分、AFT/σ选择、校准评价 |
| model.py | 摘要验证、兼容加载、margin转log秒、校准状态 |
| release.py | 来源/质量/服务器证据门禁及原子CURRENT切换 |
| distribution.py | 条件剩余分位数、窗口概率、期望和截断期望 |
| service.py | 认证API、异步预测、统一窗口刷新、leader租约 |
| candidates.py / client.py | 外部消费者缓存、版本检查、断网过期和本地条件化 |
| pipeline.py | 固定历史边界重训、训练租约、同预算语义对照 |
| cli.py | export/snapshot/train/ablation/retrain/serve/promote |
| deploy/timepredict-train.* | 与在线预测分离的周期训练模板 |
| examples/scheduler_client.py | 调用既有C2调度器前的时间接口消费示例 |

regression.py、replay.py、datasets.py、experiment.py、accuracy.py保留独立实验和历史兼容；旧action_family不是当前基线必需路径。直接分位数包无法进入正式生存分布发布。测试中使用的模拟来源/验收只验证程序连接关系，不是生产证据。

身份ID不进入模型，结果/后续query不进入特征，取消和终止不当观察截断。发出后预测一次，等待时复用μ/σ。外部模块负责密度、KV映射、加载与收益，插件只输出可消费的时间契约。
