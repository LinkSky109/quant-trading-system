# 量化交易系统 — 需求路线图 v1.0

> 生成时间: 2026-09-30 | 状态: 研发级 PRD
> 基于当前系统 v3.x 现状，补充后续需求并按优先级排序。

## 一、现状与证据

### 已完成（事实）
| 模块 | 状态 | 关键能力 |
|------|------|----------|
| 数据层 | ✅ 运行中 | QuantDash日线 + 腾讯财经实时行情/五档盘口，25只标的（15A股+5美股+5港股），多市场代码格式与交易成本自动适配 |
| 策略引擎 | ✅ 运行中 | 双均线/布林带/动量突破/RSI/MACD/网格，6策略，信号shift(1)防未来函数 |
| 策略组合 | ✅ 运行中 | 同标的多策略资金分配+组合净值+贡献度分析+冲突日志 |
| 策略排行榜 | ✅ 运行中 | 6策略批量回测+综合得分排名(夏普0.3+收益0.3+回撤0.2+胜率0.2)+信号加权聚合+动态权重+缓存 |
| 数据备份 | ✅ 运行中 | SQLite在线备份/恢复/清理/校验，每日15:30自动备份，恢复前自动兜底，4个API |
| Jev决策 | ✅ 运行中 | 本地laya-mlx模型，HTTP :8765，推理65ms，置信度阈值0.6 |
| Jev评估 | ✅ 运行中 | 决策后N日收益关联+置信度分桶+策略过滤效果评估 |
| 回测引擎 | ✅ 运行中 | 事件驱动，佣金万2.5+印花税万5+滑点0.1%，6项指标 |
| 风险管理 | ✅ 代码完成 | 止损3%/止盈8%/回撤10%/单标的20%/总仓80%/日亏2% |
| 模拟交易 | ✅ 运行中 | SimulatedBroker，实时交易引擎每5秒全流程记录 |
| 实盘接口 | ✅ 预留 | RealBroker抽象基类，XTP/QMT/同花顺/CTP枚举 |
| 监控日志 | ✅ 运行中 | 交易日志+Jev决策日志+每日绩效，SQLite持久化 |
| Web看板 | ✅ 运行中 | 实时行情/盘口/K线/账户/多策略对比/交易日志，WebSocket 1秒推送+轮询降级 |
| 回测走查 | ✅ 运行中 | 逐日快照（信号/Jev/操作/持仓/盈亏），信号vs操作对比，4个API，前端Tab |
| Jev训练数据 | ✅ 运行中 | 决策+未来收益标签，JSONL/CSV导出，按时间7:2:1划分，2个API |
| 系统健康监控 | ✅ 运行中 | 7项聚合监控+0-100评分，API性能中间件，30秒自动采集，前端卡片 |
| 组合优化器 | ✅ 代码完成 | 等权/最小方差/风险平价/均值方差四种方法，scipy SLSQP+降级，集成组合回测+API |
| 高级订单模拟 | ✅ 代码完成 | 限价/止损/止损限价/移动止盈，订单队列+当日有效，成交率+持仓时间指标 |
| API安全认证 | ✅ 代码完成 | Bearer Token+三级权限(read/trade/admin)，中间件路径映射，WebSocket认证，默认关闭 |
| 多因子分析 | ✅ 运行中 | 20因子/7大类，Spearman IC分析+IR+胜率，5层分层回测+多空+单调性，z-score暴露度，4个API |
| 专业报告生成 | ✅ 运行中 | 自包含HTML深色主题报告，10章节（净值/回撤/月度热力图/持仓/风险VaR-CVaR/信号/封面），ECharts图表，@media print，3个API |
| 事件驱动策略 | ✅ 运行中 | 7种事件检测（涨停/放量/跳空/财报/分红/拆股/指数调整），4种策略模式，CAR事件研究+t检验，3个API |

### 关键缺口（推断+事实）
1. ~~零单元测试~~ ✅ 已完成（276测试全通过）
2. ~~无持久化~~ ✅ 已完成（SQLite 4表，重启不丢数据）
3. ~~配置无校验~~ ✅ 已完成（7节37条规则，错误配置阻止启动）
4. ~~无告警通道~~ ✅ 已完成（Webhook推送+冷却去重+降级日志）
5. ~~单标的实时交易~~ ✅ 已完成（P1-1多标的并行扫描，15只标的独立决策）
6. ~~无数据质量监控~~ ✅ 已完成（P1-2 API断连/滞后/缺失K线/价格跳变/成交量异常检测）
7. ~~无组合级回测~~ ✅ 已完成（P1-3 多标的等权/波动率倒数/自定义权重组合回测）
8. ~~无每日报告自动生成~~ ✅ 已完成（P1-4 15:30自动生成+CSV导出+SQLite持久化）
9. ~~无参数优化~~ ✅ 已完成（P2-1 网格搜索优化，支持sharpe/return/drawdown目标排序）
10. ~~无绩效归因~~ ✅ 已完成（P2-2 五维归因：交易/时间/持仓/策略/风险调整）
11. ~~无多策略组合~~ ✅ 已完成（P0-12 同标的多策略资金分配+组合净值+贡献度分析）
12. ~~前端3秒轮询延迟高~~ ✅ 已完成（P0-13 WebSocket 1秒推送+断线重连+轮询降级）
13. ~~Jev决策无质量评估~~ ✅ 已完成（P0-14 决策后N日收益关联+置信度分桶+策略过滤效果）
14. ~~回测无法逐日复盘~~ ✅ 已完成（P0-15 逐日走查器，信号/Jev/操作/持仓/盈亏快照+信号vs操作对比）
15. ~~Jev决策无训练数据导出~~ ✅ 已完成（P0-16 决策+未来收益标签，JSONL/CSV，按时间划分无泄露）
16. ~~无统一系统健康视图~~ ✅ 已完成（P0-17 7项聚合+0-100评分+API性能中间件+30秒自动采集）
17. ~~仅支持A股~~ ✅ 已完成（多市场适配：美股/港股代码格式、交易成本、股票池25只、前端分组、29条测试）
18. ~~无数据备份恢复~~ ✅ 已完成（SQLite在线备份/恢复/清理/校验，每日自动备份，9条测试）
19. ~~无策略排名与信号聚合~~ ✅ 已完成（6策略综合得分排名+加权投票聚合+动态权重+缓存，12条测试）
20. ~~告警仅Webhook单渠道~~ ✅ 已完成（P2-13 多渠道通知：邮件/企微/钉钉/Server酱+模板+路由+静默，40条测试）
21. ~~技术指标仅MA/布林/RSI/MACD~~ ✅ 已完成（P2-14 19个专业指标+IndicatorCombo策略，28条测试）
22. ~~参数优化无时间序列交叉验证~~ ✅ 已完成（P2-15 Walk-Forward滚动优化+过拟合检测+参数热力图，21条测试）

## 二、目标与非目标

### 本期目标（P0，实盘前必须）
- 核心模块单元测试覆盖率 ≥ 80%
- SQLite持久化层（交易记录/账户快照/Jev决策审计）
- 配置启动校验（必填项/类型/范围/券商密钥格式）
- 风控告警Webhook通道

### 后续目标（P1/P2）
- P1: 多标的实时并行交易、数据质量监控、组合级回测
- P2: 策略参数网格优化、绩效归因分析、真实券商接入（需用户提供密钥）

### 非目标（本期不做）
- 真实券商下单（需用户提供券商API密钥和资金账号）
- 移动端APP
- 多用户/权限系统
- 机器学习策略自动训练

## 三、P0 详细需求

### P0-1 核心模块单元测试 ✅ 已完成
**问题依据**: 交易系统核心逻辑无测试，任何修改可能引入计算错误且无法发现。
**目标结果**: metrics/strategies/risk/broker/jev五大模块单元测试覆盖率≥80%。
**产品改动**: 新增 `tests/` 目录，pytest框架，test_*.py命名。

**完成状态（2026-09-30）**:
| 测试文件 | 用例数 | 模块覆盖率 |
|----------|--------|-----------|
| test_metrics.py | 42 | metrics.py 99% |
| test_strategies.py | 23 | 4个策略文件 100% |
| test_risk_manager.py | 28 | risk_manager.py 99% |
| test_broker.py | 34 | broker.py 100% |
| test_jev_engine.py | 23 | jev_engine.py 91% |
| test_persistence.py | 25 | database.py 100% |
| **合计** | **175** | **核心模块均≥91%** |

**关键验证点**:
- 策略信号shift(1)无未来函数（首行永无信号、T日信号T+1执行）
- 风控暂停时所有交易均被阻止（含卖出），需reset恢复
- 日亏限额用严格小于，恰好等于限额时仍允许
- 经纪商手续费最低5元、滑点买卖方向相反、印花税卖出单边
- Jev概率分布归一化、API失败自动降级mock、审计日志JSONL格式

**验收**: `pytest tests/ -v --tb=short` 全部通过，`pytest --cov=backtest --cov=strategies --cov=risk --cov=trading --cov=jev` 覆盖率≥80%。

### P0-2 SQLite持久化层 ✅ 已完成
**问题依据**: 服务重启后交易记录和账户状态丢失，无法做长期绩效追踪和审计。
**目标结果**: 交易记录/Jev决策/账户快照自动写入SQLite，服务重启后可恢复。
**产品改动**: 新增 `persistence/` 模块，`database.py`，realtime_trader每轮写入，server.py新增4个查询API。

**完成状态（2026-09-30）**:
| 数据表 | 写入时机 | 验证结果 |
|--------|----------|----------|
| `trades` | 每笔成交 | ✅ 卖出400股茅台，已实现盈亏+54005.89 |
| `jev_decisions` | 每轮Jev推理 | ✅ 11条记录，含market_state/probabilities JSON |
| `account_snapshots` | 每轮结束 | ✅ 11条快照，总资产1,054,005.89 |
| `daily_reports` | 每日收盘（预留） | ⬜ upsert方法已写，待接入每日调度 |

**新增API**: `GET /api/jev_decisions`、`GET /api/account_snapshots`、`GET /api/daily_reports`、`GET /api/trade_stats`
**单元测试**: test_persistence.py 25条全通过
**关键验证**: 服务重启后API仍返回历史记录（trades=1, jev_decisions=11, account_snapshots=11）
**技术细节**: WAL模式、线程安全连接池、JSON字段自动序列化、COALESCE防空查询

### P0-3 配置启动校验 ✅ 已完成
**问题依据**: config.yaml中API密钥/风控参数格式错误时，服务静默降级或运行异常。
**目标结果**: 启动时校验所有配置项，不合规直接报错退出，不带着错误配置运行。
**产品改动**: 新增 `config/validator.py`，server.py启动时调用 `validate_and_exit_on_error()`。

**完成状态（2026-09-30）**:
| 校验节 | 规则数 | 关键校验 |
|--------|--------|----------|
| data | 4 | provider类型/api_key格式(sk_+32位hex)/缓存有效期 |
| jev | 5 | URL格式/超时(0,60]/置信度[0,1]/重试次数[0,10] |
| risk | 7 | 止损(0,0.2]/止盈(0,1]/回撤(0,0.5]/仓位(0,1]/日亏(0,0.5]/单标的≤总仓位逻辑 |
| backtest | 4 | 初始资金>0/佣金[0,0.01]/印花税[0,0.01]/滑点[0,0.05] |
| strategies | 4 | 非空/label必填/快线<慢线/布林带std[0.5,5] |
| accounts | 8 | ID唯一/资金>0/类型合法/策略已定义/种子持仓格式与比例 |
| stock_pool | 5 | 非空/代码格式(A股/美股/港股)/不重复/名称必填/市值成交额>0 |

**单元测试**: test_config_validator.py 71条全通过
**集成验证**: 错误配置(API key格式错+止损率超限)正确阻止启动(退出码1)；正常配置0错误0警告通过
**总测试数**: 246条全通过

### P0-4 风控告警Webhook ✅ 已完成
**问题依据**: 风控触发仅记日志，用户无法实时感知账户风险。
**目标结果**: 止损/止盈/回撤超标/日亏限额/连续亏损触发时，推送告警到用户配置的Webhook。
**产品改动**: 新增 `monitoring/alert.py`，AlertManager类，realtime_trader集成，server.py新增3个API。

**完成状态（2026-09-30）**:
| 功能 | 实现 |
|------|------|
| 告警级别 | CRITICAL(回撤暂停/日亏限额) / WARNING(止损/止盈/仓位超限/连亏) / INFO(Jev过滤) |
| 便捷方法 | stop_loss() / take_profit() / drawdown_pause() / daily_loss_limit() / position_limit() / jev_filtered() / consecutive_losses() |
| Webhook推送 | HTTP POST JSON，含级别/类别/标的/当前值/阈值/时间戳 |
| 冷却去重 | 同类告警300秒内不重复推送（CRITICAL强制发送） |
| 降级机制 | Webhook不可用时自动降级为日志，标记不可用后不再尝试 |
| 历史记录 | 内存保留500条，支持按级别/类别过滤查询 |
| 配置 | config.yaml新增alert节(webhook_url/cooldown_seconds/max_history/timeout) |

**新增API**: `GET /api/alerts`、`POST /api/alerts/test`、`POST /api/alerts/trigger`
**集成点**: realtime_trader止损/止盈触发时调用alert_manager，Jev置信度不足时调用jev_filtered()
**单元测试**: test_alert_manager.py 30条全通过
**集成验证**: 触发测试告警→API返回→历史查询→Webhook未配置正确降级
**总测试数**: 276条全通过

### P1-1 多标的实时并行交易 ✅ 已完成
**问题依据**: 实时交易引擎每轮只处理当前选中的1只标的，账户资金利用率低，无法分散风险。
**目标结果**: 每轮循环遍历股票池所有标的，对每只独立执行策略信号→Jev决策→风控检查→下单，多标的共享账户资金和仓位。
**产品改动**: 重构 `web-dashboard/realtime_trader.py`，新增 `realtime_trading` 配置节，`/api/trade_status` 新增多标的状态字段。

**完成状态（2026-09-30）**:
| 功能 | 实现 |
|------|------|
| 多标的扫描 | 每轮遍历全部启用标的（默认15只），单只异常不影响其他 |
| 标的过滤 | `enabled_symbols` 指定子集 / `scan_all=false` 仅当前标的 |
| 持仓数上限 | `max_positions`（默认5），达上限时拒绝新开仓，允许卖出 |
| 仓位管理 | 单标的20%/总仓80%限制保持不变，多标的共享账户资金 |
| 风控集成 | 每只持仓独立止损/止盈；总回撤/日亏账户级检查 |
| Jev调用 | 每只标的独立调用（真实优先，失败降级mock），15只约1秒 |
| 持久化 | Jev决策+成交逐标的写入；账户快照每轮1次（非每标的） |
| 告警 | 每只标的止损/止盈/Jev过滤独立触发告警 |
| API兼容 | `/api/trade_log`、`/api/trade_status`、`/api/trade_toggle` 全部兼容 |
| 状态展示 | `trade_status` 新增 scan_symbols/symbol_count/max_positions/symbol_decisions |

**配置项** (`config.yaml` 新增 `realtime_trading` 节):
```yaml
realtime_trading:
  enabled_symbols: []   # 空=全部，或指定列表
  max_positions: 5      # 最大同时持仓标的数
  scan_all: true        # 是否扫描全部标的
```

**单元测试**: test_realtime_trader_multi.py 12条全通过
- 扫描列表选择（全量/子集/仅当前）
- 多标的一轮产生多只决策记录
- max_positions 拦截新开仓但允许卖出
- 仓位限制在多标的下生效
- 每标的独立止损
- 逐标的持久化 + 每轮1次快照
- 单只异常不影响其他
- get_status 多标的字段

**集成验证**:
- 开启交易后 trade_log 出现10只标的的独立决策记录（5只因QuantDash限流未初始化）
- Jev决策表累计91条（多轮×多标的），账户快照每5秒1条
- 种子持仓茅台被Jev卖出后 today_pnl=+61532.7，positions正确清空
- 服务重启后历史记录仍在（P0-2持久化验证）

**总测试数**: 288条全通过（276原有 + 12新增）

### P1-2 数据质量监控 ✅ 已完成
**问题依据**: 数据源异常（API断连/数据滞后/缺失K线/价格异常跳变）时无主动检测，可能用脏数据交易。
**目标结果**: 主动检测5类数据异常，触发告警并记录，检测失败不阻断交易。
**产品改动**: 新增 `monitoring/data_quality.py`，`DataQualityMonitor`类，集成到server.py tick循环（每10轮检测一次），新增2个API。

**完成状态（2026-09-30）**:
| 检测项 | 阈值 | 告警级别 |
|--------|------|----------|
| API断连 | 连续3次失败 | CRITICAL |
| 数据滞后 | 最新时间与当前差>5分钟 | WARNING |
| 缺失K线 | 最新日期非最近交易日或缺失>2天 | WARNING |
| 价格跳变 | >10%标记异常不告警，>20% | CRITICAL |
| 成交量异常 | 为0或为负 | WARNING |

**新增API**: `GET /api/data_quality`（最新报告+异常历史）、`GET /api/data_quality/anomalies`（仅异常历史）
**配置项**: config.yaml新增 `data_quality` 节（enabled/lag_threshold_seconds/price_jump_warning/price_jump_critical/max_consecutive_failures/anomaly_history_limit）
**单元测试**: test_data_quality.py 19条全通过
**集成验证**: 服务运行后API返回healthy状态，5项检测全部ok；检测逻辑try-except包裹，不影响正常tick

### P1-3 组合级回测 ✅ 已完成
**问题依据**: 仅支持单标的回测，无法评估多标的组合的整体绩效和资金分配效果。
**目标结果**: 支持多标的组合回测，资金在标的间分配，输出组合级绩效+单标的明细。
**产品改动**: 新增 `backtest/portfolio_engine.py`，`PortfolioBacktestEngine`类，复用`BacktestEngine`不重写逻辑，新增`POST /api/backtest_portfolio` API和示例脚本。

**完成状态（2026-09-30）**:
| 功能 | 实现 |
|------|------|
| 仓位分配 | 等权(equal) / 波动率倒数(volatility_inverse) / 自定义权重(custom) |
| 组合净值 | 各标的独立回测后按日期对齐(reindex+ffill)逐日求和 |
| 组合绩效 | 累计收益/年化/最大回撤/夏普/胜率/盈亏比/总盈利/总亏损 |
| 单标的明细 | 每只标的独立metrics+equity_curve+trades |
| 基准曲线 | 等权买入持有合并曲线 |

**新增API**: `POST /api/backtest_portfolio`（输入symbols列表+策略+日期+分配方式，返回组合绩效+净值曲线+单标的明细）
**示例脚本**: `examples/run_portfolio_backtest.py`（3只标的mock数据，输出对比表+净值曲线图）
**单元测试**: test_portfolio_engine.py 11条全通过
**集成验证**: 3只标的(茅台/宁德/比亚迪)等权回测返回241个净值点、28笔交易、组合累计收益-1.85%，权重各0.3333，与单标的回测API不冲突

### P1-4 每日报告自动生成+导出 ✅ 已完成
**问题依据**: daily_reports表的upsert方法已写但未被调用，无自动生成和导出能力。
**目标结果**: 每个交易日15:30后自动生成报告写入SQLite，支持API查询和CSV导出。
**产品改动**: 新增 `monitoring/daily_report_generator.py`，`DailyReportGenerator`类，扩展`persistence/database.py`的daily_reports表（+5列），集成到server.py tick循环，新增CSV导出API。

**完成状态（2026-09-30）**:
| 功能 | 实现 |
|------|------|
| 自动生成 | tick循环每轮调用auto_generate()，15:30后幂等生成acc_1/2/3报告 |
| 报告内容 | 日期/账户/初始资产/期末资产/日收益率/交易笔数/胜率/最大持仓/风控事件数/Jev决策统计 |
| 数据聚合 | 从trades/account_snapshots/jev_decisions表按日期LIKE过滤聚合 |
| 幂等保证 | upsert按(date+account_id)去重，重复调用更新不重复插入 |
| CSV导出 | `GET /api/daily_reports/export?account_id=XXX` 返回CSV文件下载 |

**数据库扩展**: daily_reports表新增5列（max_positions/risk_events_count/jev_decisions_count/jev_executed_count/extra_json），ALTER TABLE兼容已有数据
**新增API**: `GET /api/daily_reports/export`（CSV导出），`GET /api/daily_reports`已存在且返回完整字段
**单元测试**: test_daily_report_generator.py 13条全通过
**集成验证**: 服务启动后自动生成acc_1报告（初始资产1,054,005.89，期末1,061,532.70，日收益0.71%，2笔交易胜率100%，131条Jev决策），CSV导出含正确表头和数据行

### P2-1 策略参数网格搜索优化 ✅ 已完成
**问题依据**: 策略参数靠经验设定，无法系统评估不同参数组合的历史表现，难以找到最优参数。
**目标结果**: 支持策略参数的网格搜索优化，自动遍历参数组合并按优化目标排序返回 Top N。
**产品改动**: 新增 `optimization/grid_search.py`，`GridSearchOptimizer`类，复用`BacktestEngine`不重写回测逻辑，新增`POST /api/optimize/grid` API和示例脚本。

**完成状态（2026-09-30）**:
| 功能 | 实现 |
|------|------|
| 网格遍历 | `itertools.product` 生成所有参数组合，每组新建策略实例+回测引擎 |
| 优化目标 | sharpe（夏普降序）/ return（累计收益降序）/ drawdown（最大回撤升序） |
| 组合上限保护 | 默认 max_combos=200，超过时 `logging.warning` 警告但仍执行 |
| 进度回调 | `progress_callback(current, total, item)` 透传当前结果，回调异常不中断 |
| 容错 | 单组回测失败记录 error 字段并排末尾，不影响其他组 |
| 结果排序 | 按目标指标排序后返回 Top N，每组含 params + 完整9项绩效指标 |

**新增API**: `POST /api/optimize/grid`（输入标的/策略/参数网格/日期/目标/top_n，返回排序后的Top N参数组合及绩效）
**示例脚本**: `examples/run_grid_search.py`（双均线 fast=[3,5,8] slow=[15,20,30]，输出Top5表格）
**单元测试**: test_grid_search.py 13条全通过
**集成验证**: 茅台双均线 fast=[3,5] slow=[15,20] 共4组，返回4组结果按夏普降序排列，与现有回测API不冲突

### P2-2 绩效归因分析 ✅ 已完成
**问题依据**: 回测仅输出汇总绩效指标，无法拆解收益来源（哪些交易赚钱、哪些时间段表现好、哪些标的贡献大、信号质量如何）。
**目标结果**: 对回测结果进行五维归因分析，输出面向前端可视化的结构化归因报告。
**产品改动**: 新增 `analysis/attribution.py`，`PerformanceAttribution`类，新增`POST /api/attribution` API（简化版：传入symbol+strategy+日期，内部运行回测后归因）。

**完成状态（2026-09-30）**:
| 归因维度 | 实现 |
|----------|------|
| 交易归因 | 盈利/亏损交易贡献、笔数、最大单笔、平均值；盈亏贡献之和=总已实现盈亏（守恒验证） |
| 时间归因 | 按月/周统计收益率（首期相对回测起点计算，可拼接），最佳/最差月份，格式适合前端柱状图 |
| 持仓归因 | 组合回测时各标的已实现盈亏贡献及占比，贡献最大/拖累最大标的，格式适合饼图 |
| 策略归因 | 买入/卖出信号后N日平均收益（需传入行情data），评估买点/卖点质量 |
| 风险调整 | 收益回撤比、卡玛比率（年化/|回撤|）、索提诺比率（年化超额/下行标准差）；分母为0返回null |

**新增API**: `POST /api/attribution`（输入标的/策略/日期/lookforward_days，内部回测后返回五维归因报告+基础绩效）
**单元测试**: test_attribution.py 20条全通过（覆盖完整报告/月度分解/盈亏守恒/卡玛/索提诺/空数据/单标的/组合标的/信号质量）
**集成验证**: 茅台双均线回测归因返回完整5维度报告，盈利贡献+亏损贡献=总已实现盈亏(-14099.38)，月度收益12个月，卡玛/索提诺计算正确，与现有回测API不冲突

### P0-9 Jev服务托管与real模式联调（2026-09-30完成）

**目标**: Jev决策模型从mock模式切换为真实laya-mlx推理，量化服务与Jev服务双进程稳定运行。

**交付内容**:
- Jev HTTP服务(8765)稳定运行，模型加载约156秒，推理延迟60-90ms
- 量化服务(8766)Jev客户端自动检测8765可用性，`mode=real`，实时交易使用真实模型概率分布（非mock默认0.4/0.4/0.2）
- 一键启停脚本: `scripts/start_all.sh`（等待Jev模型加载后启动量化）、`scripts/stop_all.sh`、`scripts/status.sh`
- Jev健康监控: `/api/jev_health`端点 + `/api/health`返回Jev状态 + 后台tick每60秒检测，断连时自动降级mock并触发WARNING告警，恢复时自动切回real

**验收**: Jev `/api/health`返回`model_loaded:true`；量化`/api/jev_decision`返回`mode=real`且概率为真实模型输出；Jev断连不崩溃自动降级；三个脚本可正常使用。

### P0-10 策略库扩展（2026-09-30完成）

**目标**: 在现有3个策略基础上新增3个常用策略模板，保持可插拔架构。

**新增策略**:
| 策略 | 文件 | 逻辑 | 关键参数 |
|------|------|------|----------|
| RSI超买超卖 | `strategies/rsi.py` | RSI<30买入，RSI>70卖出 | period(14), oversold(30), overbought(70) |
| MACD金叉死叉 | `strategies/macd.py` | DIF上穿DEA买入，下穿卖出，柱状图放量过滤 | fast(12), slow(26), signal(9) |
| 网格交易 | `strategies/grid_trading.py` | 区间等距分网，跌破网格买，突破网格卖 | grid_count(10), upper/lower bound |

**交付内容**: 3个策略文件 + `strategies/__init__.py`导出 + `config/config.yaml`配置 + `web-dashboard/server.py`策略映射更新 + 16条单元测试。

**验收**: `/api/strategies`返回6个策略；每个策略回测正常运行；信号统一shift(1)无未来函数；全量测试386条通过。

### P0-11 回测报告HTML导出（2026-09-30完成）

**目标**: 支持将回测结果导出为自包含HTML报告，含净值曲线、绩效指标、交易明细。

**交付内容**:
- `backtest/report_generator.py` — ReportGenerator类，`generate_html_report()`生成深色主题自包含HTML
- 报告区块: 标题/参数、绩效指标卡片(9项)、净值曲线(ECharts策略vs基准)、回撤面积图、交易明细表、参数说明
- API: `GET /api/backtest/report?symbol=&strategy=&start_date=&end_date=&use_jev=` 返回HTML，同时缓存到`output/reports/`
- 示例: `examples/generate_report.py`
- 测试: `tests/test_report_generator.py` 6条

**验收**: HTML可浏览器直接打开；包含净值曲线和绩效指标；API返回`text/html`；测试通过。

### P0-12 策略组合引擎（2026-09-30完成）

**目标**: 支持同一标的上多策略同时运行，资金按权重分配到各策略子账户，输出组合级绩效和策略贡献度分析。

**交付内容**:
- `strategies/strategy_portfolio.py` — StrategyPortfolio类，资金隔离子账户模式：总资金按权重切分，各策略独立回测后合并净值
- 冲突信号处理：各子账户独立交易天然避免冲突，另输出conflict_log（加权投票参考），net_vote=Σ(weight·direction)
- 组合级绩效：累计收益/年化/最大回撤/夏普/胜率/盈亏比等9项指标
- 单策略明细：每个策略独立metrics+equity_curve+trades
- 贡献度分析：每策略pnl/return_pct/contribution_pct，占比之和≈1
- API: `POST /api/strategy_portfolio/backtest`（输入symbol+strategies+weights+日期，返回组合绩效+净值+明细+贡献度+冲突日志）
- 示例: `examples/run_strategy_portfolio.py`
- 测试: `tests/test_strategy_portfolio.py` 10条

**验收**: 3策略等权回测返回组合净值+绩效+单策略明细；等权时各1/3资金；权重分配正确；与单策略回测不冲突；测试通过。

### P0-13 WebSocket实时推送（2026-09-30完成）

**目标**: 将前端从3秒轮询改为WebSocket 1秒推送，降低延迟、减少请求量，行情/交易日志/告警/账户变化实时推送到前端。

**交付内容**:
- `server.py` 新增 ConnectionManager 类：管理活跃连接，每连接维护subscribed_symbols集合（默认订阅全部），支持subscribe/unsubscribe/broadcast
- WebSocket端点 `ws://localhost:8766/ws`：接受连接后发送connected欢迎消息，处理客户端subscribe/unsubscribe/ping指令
- 后台 ws_push_loop()：每1秒推送quote（按订阅）、trade（增量检测trader.trade_log）、alert（增量检测alert_manager）、account（总资产变动>0.01元）、jev（决策数增量），每30秒应用层ping心跳
- 前端 `quant_dashboard_realtime.html` 新增 WSManager IIFE：原生WebSocket API，指数退避重连（1s→30s封顶），按消息类型分发回调
- 连接状态指示器：绿=已连接/黄闪烁=重连中/红=已断开
- 行情卡片从WS推送更新，交易日志/告警实时追加到表格顶部（淡绿闪烁动画），账户卡片实时更新
- 降级策略：WS连接成功时行情走1秒推送，WS断开/重连中自动切回3秒轮询
- 所有现有REST API保持不变（轮询仍可用）

**验收**: WebSocket连接成功前端显示"已连接"；行情每秒更新；交易日志有新记录时实时出现；断连后自动重连；REST API仍正常工作。

### P0-14 Jev决策质量评估（2026-09-30完成）

**目标**: 建立Jev决策反馈闭环，评估AI决策质量，为模型优化提供数据支撑。

**交付内容**:
- `jev/decision_evaluator.py` — DecisionEvaluator类，从SQLite jev_decisions表读取历史决策
- 关联后续价格走势：决策日后N个交易日（默认5日）收益率，非交易日用searchsorted取最近前一交易日
- 决策质量判定：买入后涨→正确买入/跌→错误买入；卖出后跌→正确卖出/涨→错误卖出（卖飞）；hold及被否决决策按|收益|<2%→正确观望否则错误观望
- 置信度分桶统计：[0,0.4)/[0.4,0.6)/[0.6,0.8)/[0.8,1.0]，每桶统计决策数/正确数/正确率
- 按策略统计Jev过滤效果：每个strategy_signal的total/executed/vetoed/executed_accuracy
- 输出评估报告dict：summary/confidence_buckets/by_strategy/recent_decisions/params
- API: `GET /api/jev/evaluation?days=5&account_id=acc_1&limit=500`
- 测试: `tests/test_decision_evaluator.py` 13条（mock数据验证买卖对错/hold/分桶/边界/空库）

**验收**: 评估API返回正确率、分桶统计、策略对比；买入决策正确/错误判定逻辑正确；置信度分桶统计正确；测试通过。

### P0-15 回测走查器（2026-09-30完成）

**目标**: 支持逐日复盘回测过程，查看每日信号、持仓、盈亏、Jev决策，理解策略为什么买卖。当前回测只输出最终结果，无法查看中间过程。

**交付内容**:
- `backtest/engine.py` — 新增 `walkthrough_snapshots` 可选参数，`_daily_step` 中记录每日完整快照（信号/Jev决策/风控拦截/操作/持仓/现金/权益/盈亏），`walkthrough_snapshots is None` 时行为完全不变
- `backtest/walkthrough.py` — `BacktestWalkthrough`类：`run(data)->id`、`get_day(date)`、`get_range(start,end)`、`get_trades()`、`get_signal_vs_action()`（识别Jev过滤/风控拦截）、`to_dict()`；内存缓存+可选SQLite持久化
- `persistence/database.py` — 新增 `walkthroughs` 表 + `insert_walkthrough`/`get_walkthrough`/`list_walkthroughs` 方法
- API: `POST /api/walkthrough/run`（返回walkthrough_id）、`GET /api/walkthrough/{id}/day`、`GET /api/walkthrough/{id}/range`、`GET /api/walkthrough/{id}/trades`
- 前端: 回测区新增"逐日走查"Tab，含参数配置、逐日快照表格（日期/信号/Jev/操作/持仓/总资产/盈亏）、交易事件与信号对比表格
- 测试: `tests/test_walkthrough.py` 12条（mock数据验证逐日快照/查询/Jev过滤识别/风控拦截/daily_pnl/无走查行为不变/SQLite读写）

**验收**: 走查回测返回逐日快照含信号/Jev/操作/持仓/盈亏；信号vs操作对比识别Jev过滤/风控拦截；单日和范围查询正常；测试通过。

### P0-16 Jev训练数据导出（2026-09-30完成）

**目标**: 将历史Jev决策与后续价格走势关联，导出带标签的训练数据（JSONL/CSV），用于Jev模型微调。

**交付内容**:
- `jev/training_data.py` — `TrainingDataExporter`类：`load_decisions`（时间/标的/策略筛选）、`compute_labels`（后N日收益+最优动作标签buy/sell/hold+辅助标签max_drawdown/stop_loss_triggered）、`export_jsonl`/`export_csv`（CSV展开7个特征列）、`split_dataset`（**按时间升序切分7:2:1，严禁随机，无未来泄露**）、`compute_stats`（标签分布/平均收益/正确率）、`export_all`（一站式全流程）
- 标签判定: 未来收益>+2%→buy最优，<-2%→sell最优，否则→hold最优
- 在线拉取K线失败时静默跳过该标的，不抛异常
- API: `POST /api/jev/export_training`（触发导出，返回文件路径+统计+下载URL）、`GET /api/jev/training_stats`（最近导出统计）、`GET /training_data/{name}`（文件下载）
- 导出文件保存到 `output/training_data/`，命名 `jev_train/val/test_YYYYMMDD.jsonl/.csv`
- 示例: `examples/export_jev_training.py`（支持--symbol/--strategy/--start-date/--days参数）
- 测试: `tests/test_training_data.py` 13条（临时SQLite+构造K线验证三种标签/数据不足跳过/拉取失败容错/JSONL/CSV/时间切分无重叠/统计/端到端）

**验收**: JSONL每行含特征+标签+辅助信息；标签逻辑正确；按时间划分无未来泄露；统计正确；测试通过。

### P0-17 系统健康监控（2026-09-30完成）

**目标**: 全链路监控系统健康状态，统一展示服务进程、API延迟、Jev服务、数据库、数据质量、交易引擎、告警。

**交付内容**:
- `monitoring/system_health.py` — `ApiMetricsCollector`（线程安全，count/avg/P95/error_rate，deque滚动窗口）+ `SystemHealthMonitor`（7个collect_*方法独立try/except）
- 监控项: 量化服务(uptime/内存/CPU，psutil可选降级)、Jev服务(real/mock/disconnected三态)、API性能(各端点+全局汇总)、数据库(连接/各表行数/大小/WAL)、数据质量(复用最新报告)、交易引擎(运行/持仓/今日交易/盈亏/风控)、告警(各级别计数)
- 健康评分: 0-100加权（服务25+API20+DB15+数据质量15+交易引擎15+告警10），<60分标记critical
- `start_auto_collect(interval=30)` 后台线程自动采集，`stop_auto_collect()` 停止
- `server.py` 新增 `api_stats_middleware` HTTP中间件，统计所有 `/api/` 端点的调用次数/延迟/错误率
- API: `GET /api/system/health?refresh=true`（完整健康报告）、`GET /api/system/health/metrics`（仅API性能指标）
- 前端: 新增"系统健康监控"卡片，0-100大字号评分+健康级别+9项状态网格（颜色区分健康/异常），30秒自动刷新
- 测试: `tests/test_system_health.py` 26条（统计准确性/P95/错误率/DB表计数/告警分级/全健康=100/关键故障<60/全None依赖不崩/psutil降级/线程启停）

**验收**: 健康API返回所有监控项；API延迟统计准确（中间件记录）；健康评分计算合理；Jev服务状态正确（real/mock/disconnected）；测试通过。

### P2-3 专业组合优化器（2026-09-30完成）

**目标**: 在等权/波动率倒数基础上增加马科维茨均值方差、风险平价、最小方差三种专业组合优化方法。

**交付内容**:
- `optimization/portfolio_optimizer.py` — PortfolioOptimizer类 + OptimizeResult dataclass，四种方法 equal_weight/min_variance/risk_parity/mean_variance
- 优化引擎: scipy SLSQP 主路径，无 scipy 时降级为解析解+裁剪/循环坐标下降/随机搜索
- 约束: long-only、单标的 max_weight（默认0.3）、权重和=1；N×max_weight<1 时自动放宽
- 收益率口径: 日收益率 pct_change，μ=均值×252，Σ=协方差×252；风险贡献 RC_i=w_i(Σw)_i/σ_p
- 集成 `backtest/portfolio_engine.py` — _ALLOWED_METHODS 新增三种方法
- API: `POST /api/optimize/portfolio`（输入symbols+method+日期+max_weight，返回weights+expected_return+expected_volatility+sharpe）
- 示例: `examples/run_portfolio_optimization.py`
- 测试: `tests/test_portfolio_optimizer.py` 11条

**验收**: 四种方法权重和=1非负不超上限；风险平价RC差异<5%；最小方差波动≤等权；API正常返回；测试通过。

### P2-4 高级订单模拟（2026-09-30完成）

**目标**: 回测引擎从仅市价单扩展为支持限价单、止损单、止损限价、移动止盈，使回测更贴近实盘。

**交付内容**:
- `trading/broker.py` — Order扩展7字段（stop_price/limit_price/trailing_pct/trailing_amount/expiry/high_water_mark/triggered）；SimulatedBroker新增 pending_orders 队列、check_pending_orders()、get_pending_orders()；抽出 _settle_fill() 复用结算
- `backtest/engine.py` — BacktestEngine新增 order_type 参数（market/limit/stop/trailing_stop）；_daily_step在风控后信号前插入 _process_pending_orders()；limit模式信号转挂单，stop/trailing_stop买入后自动附带退出卖单；Trade新增 order_type/entry_date 字段
- `backtest/metrics.py` — 新增 calc_order_fill_rate()、calc_avg_holding_period()；calc_all_metrics 从9项扩展为11项
- API: BacktestReq新增 order_type 字段，_run_single_backtest 透传给引擎
- 测试: `tests/test_advanced_orders.py` 25条（限价触发/止损触发/移动止盈/撤单/当日有效/引擎四模式/成交率与持仓时间）

**验收**: 限价单价格达到时成交未达到保持pending；止损单跌破触发；移动止盈跟随最高价回撤卖出；回测结果含订单类型和成交信息；订单成交率与平均持仓时间指标正确；测试通过无回归。

### P2-5 API安全认证（2026-09-30完成）

**目标**: 为生产部署添加API Token认证，保护交易相关接口。当前所有API无认证。

**交付内容**:
- `security/auth.py` — AuthManager类：verify_token/has_permission/is_enabled/has_tokens/generate_token；权限层级 admin>trade>read；expires_at 空=永不过期
- `config/config.yaml` — 新增 security 节（enabled默认false，tokens列表，含3个示例token注释）
- `web-dashboard/server.py` — auth_middleware 中间件（路径前缀映射：公开/trade/admin/read）；enabled=false直接放行；无token→401，权限不足→403
- 接口分级: 公开(/api/health,/docs,/) / read(行情/回测/优化/Jev/数据质量/每日报告/系统健康) / trade(/api/trade_toggle) / admin(/api/alerts/trigger,/api/daily_reports/export)
- API: `POST /api/auth/verify`（公开，验证token有效性）
- WebSocket: /ws 连接时从 ?token= 校验，失败 close(code=4401)
- 导出 require_auth(permission) 依赖供未来单路由精细控制
- 测试: `tests/test_auth.py` 35条（token验证/权限矩阵/过期token/中间件401/403/WebSocket拒绝）

**验收**: enabled=false时所有接口正常访问；enabled=true时无token返回401；read不能访问trade(403)；过期token返回401；WebSocket无token拒绝；测试通过。

### P2-6 多因子分析引擎（2026-09-30完成）

**目标**: 构建专业多因子分析框架，支持因子计算、因子IC分析、因子分层回测、因子暴露度，从纯技术指标策略扩展到量化因子研究。

**交付内容**:
- `factors/factor_engine.py` — FactorEngine类，20个因子/7大类（价值4+成长3+动量3+质量3+波动率2+技术3+流动性2）；技术因子真实从K线计算，基本面因子MD5种子mock+预留`load_fundamental_data()`接口
- 核心方法: `calculate_factors` / `get_factor_list` / `factor_ic_analysis`（Spearman秩相关，返回ic_mean/ic_std/ic_ir/ic_win_rate/ic_series）/ `layered_backtest`（5层等权+多空Q5-Q1+Spearman单调性检验）/ `factor_exposure`（时序z-score）/ `build_factor_panel`
- `factors/server_routes.py` — `register_factor_routes(app, manager, SYMBOL_SET, ok, err)` 工厂，4个API
- API: `POST /api/factors/calculate` / `POST /api/factors/ic_analysis` / `POST /api/factors/layered_backtest` / `GET /api/factors/list`
- 测试: `tests/test_factor_engine.py` 23条（因子计算/IC分析/分层回测/z-score/边界）
- 示例: `examples/factor_analysis.py`

**验收**: 20个因子可计算；IC分析返回完整统计量；分层回测返回5层+多空+单调性；z-score标准化正确；测试通过；API端到端正常（25只股票横截面IC分析）。

### P2-7 专业回测报告生成器（2026-09-30完成）

**目标**: 将回测结果导出为专业HTML报告，包含净值曲线、月度热力图、持仓分析、风险指标、信号分析，可分享可打印。

**交付内容**:
- `backtest/report_generator.py` — 增强ReportGenerator，新增7方法（`_build_monthly_returns`/`_calc_var_cvar`/`_calc_max_consecutive_losses`/`_analyze_positions`/`_analyze_signals`/`_calc_sortino_ratio`/`_calc_calmar_ratio`）
- 新增4章节: 月度收益热力图（ECharts heatmap）/ 持仓分析（分桶柱状图+最大盈亏）/ 风险指标（VaR95/CVaR95/最大连续亏损）/ 信号分析（reason分布+胜率）
- 封面增强cover-grid，指标卡片新增索提诺/卡玛比率，新增`@media print`打印样式
- `backtest/report_routes.py` — `register_report_routes(app, deps)` 工厂，`ReportRouteDeps`依赖容器，3个API+路径穿越防护
- API: `POST /api/backtest/report`（生成报告返回下载URL）/ `GET /api/reports/list` / `GET /api/reports/{id}/download`
- 报告保存到 `output/reports/`
- 测试: `tests/test_report_generator.py` 25条（原6+新增19，覆盖月度矩阵/VaR/CVaR/持仓/信号/Sortino/Calmar/HTML章节/打印样式/路由）
- 示例: `examples/generate_backtest_report.py`

**验收**: HTML可在浏览器打开，含全部10章节；月度热力图数据正确；VaR/CVaR/最大连续亏损计算正确；索提诺/卡玛比率呈现；`@media print`样式存在；现有6测试不破坏；测试通过；API端到端生成+下载正常。

### P2-8 事件驱动策略框架（2026-09-30完成）

**目标**: 支持基于市场事件（涨停板/成交量异常/价格跳空/财报/分红/拆股）的策略和事件研究（CAR曲线+t检验）。

**交付内容**:
- `strategies/event_driven.py` — EventDrivenStrategy类，继承BaseStrategy；7种事件类型（limit_up/volume_spike/price_gap/earnings/dividend/stock_split/index_rebalance）
- 事件检测: `_detect_limit_up`（涨幅≥9.8%）/ `_detect_volume_spike`（>2×20日均量，shift(1)防自包含）/ `_detect_price_gap`（|跳空|>2%）/ `load_external_events`（财报/分红预留空接口）
- 4种策略模式（params["mode"]）: `pead`（跳空上涨买入持有N日）/ `limit_up_reversal`（涨停卖出均值回归）/ `limit_up_continuation`（涨停买入动量延续）/ `volume_spike`（放量上涨买入/放量下跌卖出）；进入-退出信号模式，基类shift(1)防未来函数
- 事件研究: `event_study()` 提取±window窗口异常收益（个股收益−全期均值），输出平均CAR曲线（2*window+1点）+横截面标准差+`scipy.stats.ttest_1samp`的t统计量/p值
- `strategies/server_routes.py` — `register_event_routes(app)` 工厂，3个API
- API: `POST /api/event/detect` / `POST /api/event/study` / `POST /api/event/backtest`
- `config/config.yaml` — 新增 `event_strategy` 节（default_mode/hold_days/limit_up_threshold/volume_spike_multiplier/price_gap_threshold/event_study_window）
- 测试: `tests/test_event_driven.py` 24条（事件检测/4种模式信号/事件研究CAR+t检验/可插拔回测/防未来函数/边界）
- 示例: `examples/event_driven_strategy.py`

**验收**: 3种K线事件可检测；4种模式均可生成信号；事件研究返回CAR曲线+t统计量+p值；策略继承BaseStrategy可被BacktestEngine插拔；shift(1)杜绝未来函数；测试通过；API端到端正常。

### P2-10 多因子选股策略（2026-10-01完成）

**目标**: 基于已有因子引擎构建多因子选股策略，定期计算股票池横截面因子得分，选取得分最高的N只买入、最低的卖出。

**交付内容**:
- `strategies/multi_factor.py` — MultiFactorStrategy类，继承BaseStrategy；因子加权合成（z-score/排名百分位标准化、方向可配、缺失值中位数填充/剔除）；定期调仓（rebalance_days默认5日）；top_n选股（默认5只）；等权分配；调仓历史记录
- `compute_cross_sectional_scores()` — 横截面 date×symbol 得分矩阵；`generate_rebalance_signals()` — 按调仓周期生成每标的signal DataFrame；`set_cross_section_data()` — 预加载数据供BacktestEngine多标的模式使用
- `web-dashboard/multifactor_routes.py` — `register_multifactor_routes(app, manager, symbol_set, ok, err)`，3个API
- API: `POST /api/multifactor/score`（横截面排名+单因子暴露）/ `POST /api/multifactor/backtest`（metrics/equity/trades/rebalance_history）/ `GET /api/multifactor/config`
- `config/config.yaml` — 新增 `multi_factor` 节（5因子: momentum_20 0.3 + volatility_20_inverse 0.2 + rsi_14 0.2反向 + amount_log 0.15 + macd_hist 0.15，rebalance_days=5, top_n=5）
- 测试: `tests/test_multi_factor.py` 15条（因子合成/标准化/方向/缺失值/选股/调仓/回测/边界）
- 示例: `examples/multi_factor_strategy.py`

**验收**: 因子加权合成得分正确（z-score标准化后按方向调整再加权）；top_n入选其余卖出；定期调仓逻辑正确；BacktestEngine多标的Dict模式回测正常；测试通过；API端到端正常。

### P2-11 机器学习预测策略（2026-10-01完成）

**目标**: 集成scikit-learn，构建基于机器学习的价格方向预测策略，从规则-based扩展到ML驱动。

**交付内容**:
- `ml/predictor.py` — MLPredictor类；特征工程（14个特征: 收益率1/5/10/20日、波动率5/20日、RSI、MACD三列、布林带位置、成交量变化率、均线偏离度5/20日）；标签构建（未来N日收益率方向分类/回归）；4种模型（RandomForest默认/GradientBoosting/LogisticRegression/SVM）；按时间前80%/后20%划分训练测试集（无数据泄露）；joblib/pickle保存加载；预测信号（概率>0.6买入/<0.4卖出/否则hold）
- `strategies/ml_strategy.py` — MLStrategy类，继承BaseStrategy；`train_model()`训练；`_compute_raw_signals()`逐日推理（严格避免未来函数）；模型定期重训练支持
- `web-dashboard/ml_routes.py` — `register_ml_routes(app, manager, symbol_set, ok, err)`，4个API
- API: `POST /api/ml/train`（训练+性能指标accuracy/AUC/混淆矩阵）/ `POST /api/ml/predict`（上涨概率+信号）/ `POST /api/ml/backtest`（ML策略回测）/ `GET /api/ml/models`（已训练模型列表）
- `ml/__init__.py` — 包初始化
- `requirements.txt` — 新增注释掉的可选依赖（scikit-learn>=1.3.0, joblib>=1.3.0）
- 测试: `tests/test_ml_predictor.py` 14条（特征工程/标签/训练/预测/保存加载/策略/回测），sklearn未安装时全部优雅skip
- 示例: `examples/ml_strategy.py`

**验收**: 特征工程生成>=12个特征；模型能训练并输出0-1预测概率；模型保存/加载正常；回测能正常运行；sklearn未安装时模块可正常import且测试优雅skip；API端到端正常（sklearn可用时）。

### P2-12 部署运维完善（2026-10-01完成）

**目标**: 完善生产部署能力：Docker容器化、systemd服务管理、健康检查端点、启动脚本增强、部署文档、环境变量支持。

**交付内容**:
- `Dockerfile` — python:3.11-slim基础镜像，安装依赖，暴露8766，HEALTHCHECK指向/healthz
- `docker-compose.yml` — quant服务 + 可选jev服务（profiles），数据卷挂载（data/output/models/config/logs），环境变量，健康检查
- `.dockerignore` — 排除__pycache__/.git/data/*.db/output/logs/cache/models等
- `deploy/quant-trading.service` — systemd unit文件（Restart=always, journal日志, 环境变量）
- `deploy/install_service.sh` — systemd安装脚本（复制+daemon-reload+enable+start）
- `web-dashboard/server.py` — 新增 `GET /healthz` 极简端点（返回{"status":"ok"}），加入公开路径
- `config/__init__.py` — `load_config()` 支持环境变量覆盖（QUANT_JEV_URL/QUANT_API_KEY/QUANT_LOG_LEVEL/QUANT_PORT）
- `.env.example` — 环境变量模板
- `scripts/start_all.sh` — 增强：Python版本检查、依赖检查、PID文件管理、等待/healthz就绪
- `scripts/stop_all.sh` — 增强：PID文件读取、优雅停止15秒后SIGKILL、PID文件清理
- `scripts/status.sh` — 增强：CPU/内存显示、最近日志tail、磁盘使用统计
- `DEPLOYMENT.md` — 完整部署指南（7章节：环境要求/本地启动/Docker部署/systemd部署/配置说明/健康检查/常见问题）
- `README.md` — 重写（项目概述/快速开始/功能列表/ASCII架构图/目录结构/API文档/部署文档链接）
- 测试: `tests/test_deployment.py` 27条（/healthz端点/环境变量覆盖/脚本语法/Dockerfile/docker-compose/systemd service）

**验收**: Dockerfile语法正确含所有必要指令；docker-compose.yml可被yaml解析；systemd service文件语法正确；/healthz返回{"status":"ok"}；三个脚本通过bash -n语法检查；DEPLOYMENT.md 7章节完整；README.md内容完整；环境变量覆盖功能正常；测试通过；现有认证/配置测试不受影响。

### P2-13 多渠道通知告警系统（2026-10-01完成）

**目标**: 在已有AlertManager（Webhook+冷却+降级）基础上，扩展多渠道通知能力，支持邮件/企业微信/钉钉/Server酱，含通知模板、路由规则、静默时间。

**交付内容**:
- `monitoring/notifier.py` — NotifierManager类，5个渠道适配器（Email/WeCom/DingTalk/ServerChan/Webhook）
  - 邮件: SMTP HTML邮件，支持SSL/TLS
  - 企业微信: 群机器人Webhook，text/markdown
  - 钉钉: 群机器人Webhook，text/markdown，HMAC-SHA256加签安全验证
  - Server酱: SendKey推送（微信接收）
- NotificationTemplate模板系统: 交易通知/风控告警/系统告警/每日报告，变量替换（{symbol}/{action}/{price}等）
- 通知路由: 按级别（CRITICAL→邮件+企微，WARNING→企微，INFO→仅日志）+ 按事件类型（trade→企微，risk→邮件+企微，system→邮件），取并集
- 静默时间: 23:00-08:00非CRITICAL只记录不推送，支持跨午夜
- AlertManager集成: set_notifier()回调，告警触发时自动推送，try/except包裹不阻断交易
- `config/config.yaml` — 新增notification节，默认全部disabled不影响现有功能
- API: `POST /api/notification/test`、`GET /api/notification/status`、`POST /api/notification/send`
- 测试: `tests/test_notifier.py` 40条（模板渲染/渠道适配/钉钉加签/路由规则/静默时间，网络请求全mock）
- 示例: `examples/send_notification.py`

**验收**: 4种渠道适配器代码完整；钉钉加签验证正确；模板变量替换正确；路由规则正确；静默时间非CRITICAL不发送；AlertManager集成不破坏现有Webhook；40条测试通过；默认全disabled不影响现有功能。

### P2-14 专业技术指标库（2026-10-01完成）

**目标**: 扩展技术指标库，在已有MA/布林带/RSI/MACD基础上，增加专业交易员常用的19个指标，覆盖趋势/震荡/成交量/波动率/形态五大类。

**交付内容**:
- `indicators/technical.py` — 19个指标纯pandas/numpy实现，向量化计算
  - 趋势类(5): ATR、ADX、DMI、Ichimoku一目均衡表、SAR抛物线转向
  - 震荡类(5): KDJ、CCI、WR威廉、ROC变动率、MOM动量
  - 成交量类(4): OBV能量潮、VWAP、MFI资金流量、CMF蔡金资金流
  - 波动率类(3): 布林带带宽、ATR比率、历史波动率(20日年化)
  - 形态类(2): 均线多头/空头排列检测、金叉/死叉检测
- TechnicalIndicators门面类: calculate()/list_indicators()/calculate_all()
- Wilder平滑采用教科书口径（SMA种子+递归衰减），前period-1周期严格NaN
- `strategies/indicator_combo.py` — IndicatorComboStrategy多指标组合策略（ADX>25+MACD金叉+RSI<70买入；ADX<20+MACD死叉卖出）
- `strategies/__init__.py` — 导出IndicatorComboStrategy
- `strategies/strategy_leaderboard.py` — 策略映射表新增indicator_combo
- `config/config.yaml` — strategies节新增indicator_combo配置
- API: `POST /api/indicators/calculate`、`GET /api/indicators/list`
- 测试: `tests/test_indicators.py` 28条（每个指标至少1个测试，已知数据验算，NaN处理，策略回测）
- 示例: `examples/technical_indicators.py`

**验收**: 19个指标实现；计算结果与已知值一致；NaN处理正确；IndicatorComboStrategy能正常回测；28条测试通过；不破坏现有utils/indicators.py。

### P2-15 策略参数滚动优化（Walk-Forward）（2026-10-01完成）

**目标**: 在已有网格搜索基础上，增加滚动优化（Walk-Forward Optimization）和样本外验证，避免过拟合。支持IS训练+OOS测试的多窗口滚动、过拟合检测、参数热力图、推荐参数。

**交付内容**:
- `optimization/walk_forward.py` — WalkForwardOptimizer类
  - 滚动窗口划分: IS/OOS无重叠、严格时间顺序、无未来函数，步长=OOS长度，最后OOS到数据末尾
  - 每窗口流程: IS内GridSearchOptimizer寻优 → 最优参数OOS回测 → 记录绩效
  - 参数空间: 连续+离散参数，组合数上限保护（默认500），支持随机搜索
  - 优化目标: sharpe/return/calmar（年化/|最大回撤|）/sortino（年化-无风险/下行标准差）
  - 过拟合检测: OOS/IS比率<0.5标high、<0.7标medium；参数稳定性评分
  - 结果输出: 窗口详情、合并OOS净值曲线（归一化复利拼接）、合并绩效、过拟合报告、推荐参数（数值中位数/类别众数）、参数热力图
- API: `POST /api/optimize/walk_forward`、`POST /api/optimize/random_search`、`GET /api/optimize/results/{result_id}`（内存缓存，上限50条）
- 测试: `tests/test_walk_forward.py` 21条（窗口划分/IS寻优+OOS测试/净值合并/过拟合检测/随机搜索/推荐参数/热力图，100条K线2窗口快速验证）
- 示例: `examples/walk_forward_optimization.py`

**验收**: 滚动窗口划分正确（无重叠/无未来函数/比例正确）；每窗口IS优化+OOS测试流程正确；合并OOS结果正确；过拟合检测逻辑正确；随机搜索功能正常；推荐参数计算正确；21条测试通过；不破坏现有GridSearchOptimizer。

## 四、P1 需求（简述）

| 编号 | 需求 | 价值 | 依赖 | 状态 |
|------|------|------|------|------|
| P1-1 | 多标的实时并行交易 | 充分利用资金，分散风险 | P0-2持久化 | ✅ 已完成 |
| P1-2 | 数据质量监控 | 防止用脏数据交易 | 无 | ✅ 已完成 |
| P1-3 | 组合级回测 | 评估多标的组合表现 | P0-1测试 | ✅ 已完成 |
| P1-4 | 每日报告自动生成+导出 | 绩效复盘 | P0-2持久化 | ✅ 已完成 |
| P2-1 | 策略参数网格搜索优化 | 自动寻找最优参数组合 | P1-3回测 | ✅ 已完成 |
| P2-2 | 绩效归因分析 | 拆解收益来源，指导策略改进 | P1-3回测 | ✅ 已完成 |
| P2-3 | 专业组合优化器 | 均值方差/风险平价/最小方差 | P1-3回测 | ✅ 已完成 |
| P2-4 | 高级订单模拟 | 限价/止损/移动止盈回测 | 回测引擎 | ✅ 已完成 |
| P2-5 | API安全认证 | Token认证+权限分级 | 无 | ✅ 已完成 |
| P2-6 | 多因子分析引擎 | 20因子/IC分析/分层回测/暴露度 | 数据层 | ✅ 已完成 |
| P2-7 | 专业回测报告生成器 | HTML报告/月度热力图/VaR/打印 | 回测引擎 | ✅ 已完成 |
| P2-8 | 事件驱动策略框架 | 事件检测/CAR事件研究/4种策略模式 | 策略基类 | ✅ 已完成 |
| P2-10 | 多因子选股策略 | 横截面因子加权打分/定期调仓/top_n选股 | 因子引擎+回测引擎 | ✅ 已完成 |
| P2-11 | 机器学习预测策略 | sklearn价格方向预测/4种模型/特征工程 | 策略基类+回测引擎 | ✅ 已完成 |
| P2-12 | 部署运维完善 | Docker/systemd/健康检查/环境变量/部署文档 | 无 | ✅ 已完成 |
| P2-13 | 多渠道通知告警系统 | 邮件/企微/钉钉/Server酱+模板+路由+静默 | AlertManager | ✅ 已完成 |
| P2-14 | 专业技术指标库 | 19个指标(趋势/震荡/成交量/波动率/形态)+组合策略 | 策略基类 | ✅ 已完成 |
| P2-15 | 策略参数滚动优化 | Walk-Forward IS/OOS+过拟合检测+参数热力图 | 网格搜索+回测 | ✅ 已完成 |
| P2-9 | 真实券商接入 | 实盘交易 | 需用户提供券商API密钥 | ⬜ 待用户提供密钥 |
| P3-1 | 多市场数据适配（美股/港股） | 覆盖全球主要市场 | 数据层 | ✅ 已完成 |
| P3-2 | 数据备份与恢复 | 生产环境数据安全 | 持久化层 | ✅ 已完成 |
| P3-3 | 策略排行榜与信号聚合 | 多策略智能选优 | 策略引擎+回测 | ✅ 已完成 |

## 五、风险与依赖

| 风险 | 影响 | 缓解 |
|------|------|------|
| 真实券商接入需用户提供密钥 | P2无法完成 | 已预留RealBroker接口，用户提供后实现 |
| QuantDash限流(10次/分) | 多标的K线初始化慢，25只标的启动需数分钟 | 已加缓存，启动时串行+退避，美股/港股不可用时降级mock |
| Jev模型置信度偏低(0.4左右) | 交易信号少 | 阈值可调，当前0.6偏保守，可回测优化 |
| 腾讯财经接口无SLA | 实时行情可能断连 | 已加降级，断连时保持最后价格 |
| 美股/港股数据权限受限 | QuantDash可能不支持非A股日线 | 已实现腾讯财经HTTP降级+mock兜底，数据质量监控标注延迟 |
| 数据库恢复操作风险 | 误恢复可能覆盖最新数据 | 恢复前自动备份当前库到pre_restore_*.db，可回滚 |

## 六、本期验收总标准

1. `pytest tests/` 全通过（703条：689 passed + 14 skipped，sklearn未安装时ML测试优雅skip），核心模块覆盖率≥80%
2. 服务重启后交易日志和账户状态不丢失
3. 错误配置启动时被拦截并报具体错误
4. 风控触发时Webhook收到告警
5. 所有现有功能（行情/回测/Jev/实时交易/看板）不受影响
6. P1-2数据质量监控：API返回质量报告，异常检测不阻断交易
7. P1-3组合级回测：3只标的返回组合净值+绩效+单标的明细
8. P1-4每日报告：15:30后自动生成，CSV可导出，upsert幂等
9. P2-1网格搜索：双均线4组参数返回按夏普排序的Top结果，每组含完整绩效
10. P2-2绩效归因：返回五维归因报告（交易/时间/持仓/策略/风险调整），盈亏贡献守恒
11. P0-12策略组合：3策略等权回测返回组合净值+绩效+单策略明细+贡献度分析，权重各1/3
12. P0-13 WebSocket：ws://localhost:8766/ws连接成功，行情1秒推送，交易日志实时追加，断线自动重连，REST轮询降级可用
13. P0-14 Jev评估：GET /api/jev/evaluation返回正确率+置信度分桶+策略过滤效果，买卖对错判定逻辑正确
14. P0-15 回测走查：POST /api/walkthrough/run返回逐日快照含信号/Jev/操作/持仓/盈亏，信号vs操作对比识别Jev过滤/风控拦截，单日/范围查询正常，前端"逐日走查"Tab可运行并展示表格
15. P0-16 Jev训练数据导出：POST /api/jev/export_training生成JSONL+CSV（train/val/test按时间划分），标签逻辑正确（涨→buy/跌→sell/横盘→hold），GET /api/jev/training_stats返回统计
16. P0-17 系统健康监控：GET /api/system/health返回7项监控+0-100评分+健康级别，API中间件统计各端点延迟/错误率，Jev服务正确识别real/mock/disconnected，前端健康卡片30秒自动刷新
17. P2-6 多因子分析：GET /api/factors/list返回20因子元数据，POST /api/factors/calculate返回单票20因子值+z-score暴露度，POST /api/factors/ic_analysis返回ic_mean/ic_std/ic_ir/ic_win_rate/ic_series（25只股票横截面），POST /api/factors/layered_backtest返回5层收益+多空+单调性检验
18. P2-7 专业报告：POST /api/backtest/report运行回测并生成HTML报告（含净值/回撤/月度热力图/持仓分析/风险VaR-CVaR/信号分析/索提诺/卡玛/@media print），GET /api/reports/list返回报告列表，GET /api/reports/{id}/download返回HTML文件（路径穿越防护）
19. P2-8 事件驱动：POST /api/event/detect返回涨停/放量/跳空事件列表，POST /api/event/study返回CAR曲线（2*window+1点）+t统计量+p值，POST /api/event/backtest运行事件驱动策略回测（4种模式pead/limit_up_reversal/limit_up_continuation/volume_spike），信号shift(1)杜绝未来函数
20. P2-10 多因子选股：POST /api/multifactor/score返回25只标的横截面因子得分排名+单因子暴露，POST /api/multifactor/backtest运行多因子策略回测（metrics/equity/trades/rebalance_history），GET /api/multifactor/config返回因子配置，z-score标准化后加权得分正确，top_n选股+定期调仓逻辑正确，BacktestEngine多标的模式正常
21. P2-11 机器学习预测：POST /api/ml/train训练模型返回accuracy/AUC/混淆矩阵（14个特征，前80%/后20%时间划分无泄露），POST /api/ml/predict返回上涨概率+信号（>0.6买/<0.4卖），POST /api/ml/backtest运行ML策略回测，GET /api/ml/models返回已训练模型列表，sklearn未安装时模块可import且测试优雅skip
22. P2-12 部署运维：GET /healthz返回{"status":"ok"}（Docker HEALTHCHECK可用），Dockerfile/docker-compose.yml语法正确，systemd service文件语法正确，环境变量覆盖（QUANT_JEV_URL/QUANT_API_KEY/QUANT_LOG_LEVEL/QUANT_PORT）生效，三个脚本通过bash -n，DEPLOYMENT.md 7章节完整，README.md重写完成

---

## 七、v4.0 增强批次进度（2026-10-02 完成）

> 基于 Agent Continuous Workflow 执行，任务ID: TASK-20261002-001，全局状态: GLOBAL_DONE
> 全量测试: **940 passed, 14 skipped, 0 failed**（基线778 + 新增162）

### 第1批（4项）— ✅ 已完成 DEV_DONE

| 编号 | 需求 | 优先级 | 模块 | 新增测试 | 状态 |
|------|------|--------|------|----------|------|
| REQ-P0-05 | 安全审计日志 | P0 | security/audit.py | 35 | ✅ 完成 |
| REQ-P0-06 | API限流 | P0 | security/rate_limit.py | 27 | ✅ 完成 |
| REQ-P1-01 | 回测走查器前端增强 | P1 | quant_dashboard_realtime.html | — | ✅ 完成 |
| REQ-P1-02 | 策略对比仪表盘增强 | P1 | quant_dashboard_realtime.html | — | ✅ 完成 |

**第1批交付内容**:
- **审计日志**: AuditLogger类 + SQLite audit_logs表 + JSONL双写 + SHA-256哈希链完整性校验 + 10种操作类型枚举 + 2个查询API + broker/trade_toggle/backup集成
- **API限流**: TokenBucket令牌桶 + 4级分级配额(read 100/trade 20/admin 10/backtest 5每分钟) + 白名单 + LRU清理 + 429+Retry-After + 状态API
- **走查器前端**: 日期导航控制栏 + 当日详情面板 + Jev过滤/风控拦截行级高亮 + 当日K线图标注买卖点 + 前端内存缓存 + use_jev开关
- **策略对比**: 9色调色板 + 6维指标雷达图 + 交易散点图(FIFO配对) + 月度收益热力图 + 策略相关性矩阵(Pearson) + 9项指标对比表 + 动画进度条

**第1批测试结果**: 840 passed, 14 skipped, 0 failed（基线778 + 新增62）

### 第2批（3项）— ✅ 已完成 DEV_DONE

| 编号 | 需求 | 优先级 | 模块 | 新增测试 | 状态 |
|------|------|--------|------|----------|------|
| REQ-P1-07 | 配置热加载 | P1 | config/hot_reload.py | 22 | ✅ 完成 |
| REQ-P1-08 | 结构化日志 | P1 | monitoring/structured_logging.py | 19 | ✅ 完成 |
| REQ-P2-11 | 通知中心前端 | P2 | quant_dashboard_realtime.html | — | ✅ 完成 |

**第2批交付内容**:
- **配置热加载**: HotReloadManager + 7节可热加载(strategies/risk/notification/stock_pool/alert/rate_limit/audit) + 4节不可热加载(data/jev/logging/security等) + diff对比 + 回调注册 + SIGHUP信号 + CONFIG_CHANGE审计 + 3个API
- **结构化日志**: StructuredJSONFormatter(JSON输出) + RequestIDMiddleware(纯ASGI实现,contextvars传递) + RotatingFileHandler(50MB×10) + trades/alerts/jev独立文件 + 3个API
- **通知中心前端**: Toast弹窗(4级颜色/自动消失/最多5条) + Web Audio提示音(CRITICAL 3声/WARNING 2声/INFO 1声/可静音) + 通知中心面板(筛选/已读管理) + 偏好设置(localStorage持久化) + WebSocket alert/trade集成 + 桌面通知权限

**第2批测试结果**: 881 passed, 14 skipped, 0 failed（840 + 新增41）

### 第3批（3项）— ✅ 已完成 DEV_DONE

| 编号 | 需求 | 优先级 | 模块 | 新增测试 | 状态 |
|------|------|--------|------|----------|------|
| REQ-P0-03 | 风险模型VaR/CVaR | P0 | risk/var_model.py | 22 | ✅ 完成 |
| REQ-P1-04 | 配对交易策略 | P1 | strategies/pairs_trading.py | 19 | ✅ 完成 |
| REQ-P1-05 | Jev推理性能优化 | P1 | jev/jev_engine.py | 18 | ✅ 完成 |

**第3批交付内容**:
- **VaR/CVaR风险模型**: VaRModel类 + 历史模拟法VaR(95%/99%) + 方差-协方差法VaR + CVaR(条件VaR) + 5个预设压力场景(2008金融危机/2015股灾/2020疫情/单日跌10%/单日涨5%) + Euler组合VaR分解(成分贡献度) + 阈值监控告警 + 3个API
- **配对交易策略**: PairsTradingStrategy(继承BaseStrategy) + Engle-Granger两步法协整检验(statsmodels优先/numpy降级) + OLS对冲比率 + 价差z-score信号(z≥2开仓/z≤0.5平仓/z≥3止损) + 标的对自动筛选(相关系数+协整检验) + 3个API
- **Jev性能优化**: 批量推理evaluate_batch + 异步队列+worker(asyncio.Queue) + LRU KV缓存(maxsize=100) + asyncio.Semaphore并发控制(默认3) + 延迟统计P50/P95/P99(deque环形缓冲) + /api/jev/performance端点

**第3批测试结果**: 940 passed, 14 skipped, 0 failed（881 + 新增59）

### v4.0 新增API端点汇总（15个）

| 模块 | 端点 | 方法 |
|------|------|------|
| 审计 | /api/audit/logs | GET |
| 审计 | /api/audit/verify | GET |
| 限流 | /api/rate_limit/status | GET |
| 热加载 | /api/config/reload | POST |
| 热加载 | /api/config/current | GET |
| 热加载 | /api/config/hot_reloadable | GET |
| 日志 | /api/logs/recent | GET |
| 日志 | /api/logs/level | POST |
| 日志 | /api/logs/files | GET |
| 风险 | /api/risk/var | POST |
| 风险 | /api/risk/stress_test | POST |
| 风险 | /api/risk/var_status | GET |
| 配对 | /api/pairs/screen | POST |
| 配对 | /api/pairs/backtest | POST |
| 配对 | /api/pairs/list | GET |
| Jev | /api/jev/performance | GET |

### v4.0 关键技术决策记录

1. **FastAPI中间件执行顺序**: 实测为**逆序注册**（后注册的中间件先执行，为最外层）。限流中间件注册在auth之前，确保auth先写入token_info后限流才能按token名称限流。RequestIDMiddleware注册在最后成为最外层，确保所有下游中间件能拿到X-Request-ID。
2. **审计日志哈希链**: 首条记录prev_hash为全零创世哈希，每条hash=SHA-256(全字段+prev_hash)，SQLite与JSONL双写互为备份。
3. **前端两需求同文件**: P1-01/P1-02/P2-11修改同一HTML，由单一代理串行执行避免合并冲突。
4. **限流默认关闭**: rate_limit.enabled=false，生产环境需手动开启，避免影响开发调试。
5. **扩展路由模式**: 第3批后端使用`_register_extension_routes()`模式（参考_routes_indicators/_routes_walkforward），新建_routes_risk.py和_routes_pairs.py独立文件，try/except隔离挂载。
6. **Jev服务端复用**: `/Users/link/myApp/ai/jev/jev_server.py`已存在v1.1.0且内置`/api/evaluate_batch`，未修改，客户端侧优先调用批量接口失败自动串行降级。
7. **协整检验降级**: statsmodels不可用时自动降级为numpy自实现ADF检验，确保无额外依赖也能运行。

## 八、v4.1 增强批次进度（2026-10-03 第1批完成）

任务 TASK-20261003-002，基线 940 passed + 14 skipped（954 total）。

### 第1批（4项后端）— ✅ 已完成 DEV_DONE

| 需求 | 模块 | 路由文件 | 测试 |
|---|---|---|---|
| REQ-P0-02 订单管理系统(OMS) | trading/oms.py | _routes_oms.py | +36 |
| REQ-P1-03 Brinson绩效归因 | analysis/brinson.py | _routes_brinson.py | +24 |
| REQ-P1-09 因子风险暴露监控 | risk/factor_exposure.py | _routes_factor_exposure.py | +18 |
| REQ-P1-10 算法交易(TWAP/VWAP) | trading/algorithmic.py | _routes_algo.py | +32 |

执行编排：OMS/Brinson/FactorExposure 三代理并行；OMS 交付后派发依赖它的 Algorithmic；server.py 的 API 注册由编排者统一收敛，避免多代理改同一文件。

### v4.1 新增API端点汇总（19个）

- **OMS（6）**: `POST /api/oms/orders`、`DELETE /api/oms/orders/{id}`、`GET /api/oms/orders`、`GET /api/oms/orders/{id}`、`GET /api/oms/fills`、`POST /api/oms/orders/{id}/reject`
- **Brinson（2）**: `POST /api/attribution/brinson`、`GET /api/attribution/brinson/{task_id}`
- **因子暴露（4）**: `POST /api/risk/factor_exposure`、`GET /api/risk/factor_exposure/history`、`GET /api/risk/factor_exposure/alerts`、`POST /api/risk/factor_exposure/neutralize`
- **算法交易（7）**: `POST /api/algo/twap`、`POST /api/algo/vwap`、`GET /api/algo/orders/{id}`、`POST /api/algo/orders/{id}/{pause,resume,stop}`、`GET /api/algo/orders/{id}/report`

### 验收结果

- 全量测试：**1050 passed, 14 skipped, 0 failed**（新增110条，零回归，1064 total）。
- 端到端：`scripts/e2e_v41_check.py` **37 passed, 0 failed**，19个新端点真实 HTTP 调用全部通过；服务已重启运行在 8766，前端页面 200。
- Brinson 闭合验证：单期闭合残差 0.00e+00；Cariño 两期链接残差 −1.65e−07（要求 <1e−4）。

### v4.1 关键技术决策记录

1. **OMS 状态机**: `ORDER_TRANSITIONS` 映射 + 自定义 `InvalidOrderStateError`；三桶订单簿（pending/active/historical）+ master 索引，RLock 保护；成交按量加权重算均价，超量抛错。
2. **模块内持久化**: 为避免并行代理改 persistence/database.py，OMS（orders/fills 表）与因子暴露（factor_exposure_history 表）在各自模块内用 sqlite3 直连 data/quant_trading.db，WAL + CREATE TABLE IF NOT EXISTS。
3. **db 路径锚定**: 路由单例最初用相对路径 `data/quant_trading.db`，从 web-dashboard 目录启动时解析失败；统一改为 `Path(__file__).resolve().parent.parent` 锚定的绝对路径。
4. **VWAP period 容错**: volume_profile 的 `period` 既可能是时长数值也可能是 "09:30-10:00" 标签，float 转换失败时回退 30 分钟。
5. **时间可测性**: OMS `check_timeouts(now=...)`、算法执行器 `tick(now=...)`/`execute_next_slice()` 支持注入时间，测试零长 sleep；TWAP 尾单吸收余数（100/3→33,33,34），VWAP 参与率截断缺口显式记 leftover。
6. **数据限制**: Brinson 行业分类为 mock（sector_source=mock_simplified，meta 含 P0-04 切换提示）；因子暴露的 FactorEngine 自动取分分支、算法后台 daemon 线程 tick 路径未在 E2E 跑通（代码防御性处理，与已测路径共用逻辑）。

## 九、v4.1 增强批次进度（2026-10-03 第2批完成）

任务 TASK-20261003-003，基线 1050 passed + 14 skipped（1064 total，第1批后）。

### 第2批（2后端 + 1前端）— ✅ 已完成 DEV_DONE

| 需求 | 模块 | 路由/资源 | 测试 |
|---|---|---|---|
| REQ-P2-09 数据校验与对账 | data/validation.py | _routes_data.py | +27 |
| REQ-P2-12 Jev可解释性 | jev/explainability.py | _routes_explain.py | +17 |
| REQ-P1-06 移动端适配/PWA | quant_dashboard_realtime.html | manifest.json / sw.js / icons | CDP 实测 |

三代理并行（不同文件，互不冲突）；server.py 第14/15组扩展路由与 PWA 静态路由由编排者统一注册。

### v4.1 第2批新增 API 端点（8个）+ PWA 静态资源

- **数据校验对账（4）**: `POST /api/data/reconcile`、`GET /api/data/quality`、`GET /api/data/anomalies`、`POST /api/data/anomalies/{id}/fix`
- **Jev 可解释性（4）**: `POST /api/jev/explain`、`POST /api/jev/feature_importance`、`POST /api/jev/counterfactual`、`GET /api/jev/explain/{decision_id}`
- **PWA 静态（4类）**: `GET /manifest.json`（application/manifest+json）、`GET /sw.js`（application/javascript，含 Service-Worker-Allowed: /）、`GET /icons/icon-192.png`、`GET /icons/icon-512.png`（icons 路由含路径穿越防护）

### 移动端实测结果（CDP 390×844，touch 模拟）

| 验证项 | 结果 |
|---|---|
| 单列布局 / 无横向溢出（scrollWidth==390） | ✅ |
| 底部固定 5 Tab（行情/交易/回测/通知/我的）显示 + 切换 active | ✅ display:flex，点击切换正常 |
| ECharts 全宽渲染（K线 340×280，共 4 实例） | ✅ |
| resize / orientationchange 图表重排（200ms 防抖） | ✅ |
| 触摸目标 ≥44px、模态框全屏 | ✅ |
| **Service Worker 注册成功，scope 根路径、controller 激活** | ✅ |
| 桌面 >1024 零回归（CSS 仅在 max-width 媒体查询内生效） | ✅ |

证据截图：`web-dashboard/_shots/b2_mobile_market.png`（单列+真实行情+K线全宽）、`b2_mobile_backtest.png`（Tab 切换）。

### 顺带修复的既有缺陷

`renderDayDetail()` 内同一作用域重复声明 `const closeVal`（3347 行与原第二处），会让整段主脚本解析失败、全页无数据/图表空白；第二个声明已重命名为 `closePrice`（现 3360 行）。这是页面此前空白的根因，修复后数据与 4 个图表实例正常加载。

### 验收结果

- 全量测试：**1094 passed, 14 skipped, 0 failed**（新增44条后端测试，零回归，1108 total）。
- 端到端：`scripts/e2e_v41_batch2_check.py` **37 passed, 0 failed**，8 个新端点 + 4 类 PWA 资源真实 HTTP 调用全部通过；服务已重启运行在 8766。
- 数据质量评分实测：completeness 0.9 / accuracy 0.8 / timeliness 0.8（权重 0.4/0.4/0.2）→ 总分 84.0；价格偏差 100 vs 100.4 → validated（0.4%），100 vs 101 → flagged（1%）。

### 第2批关键技术决策记录

1. **数据交叉验证**: 偏差率 `|p1−p2|/min(p1,p2)`，默认容忍 0.5%；四类异常（价格跳变 A股±10%/其余±20%、成交量 >5倍 或 <1/10、缺失K线按工作日启发式、OHLC 逻辑错误）；血缘三态 validated/flagged/corrected；备用源也缺时保持 open 不假装修复成功。
2. **Jev 解释（离线优先）**: `predict_callable` 构造注入，缺省走引擎 mock 概率路径，路由再降级内置 demo softmax，全程不依赖 8765；特征重要性/单次贡献用扰动法，反事实对可调连续特征做边界探测+二分，不可行如实返回 `feasible=False`。
3. **SHAP 无硬依赖**: 本机未装 shap（`shap_available=False`，method=perturbation_approx），try-import 保护，装库后自动切 KernelExplainer。
4. **PWA 离线策略**: sw.js 预缓存 App Shell；同源 GET 中 `/api/` 走 networkFirst 回退缓存，静态资源 cache-first/stale-while-revalidate；非 GET 放行。
5. **数据限制**: 真实网络行情/交易日历未接入（E2E 用显式 sources，节假日缺失K线会误报）；下拉刷新逻辑已接通但未逐帧触发；CDP 截图不合成 fixed 层（Tab 栏经 computed display 与 hit-test 验证存在可交互）。
