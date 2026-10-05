# 量化交易系统

基于 Python 的完整量化交易系统：多策略并行、本地 AI 决策模型（Jev）集成、事件驱动回测、实时风控、可视化看板与生产级部署能力。

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 启动服务
python3 web-dashboard/server.py

# 3. 打开看板
# 浏览器访问 http://localhost:8766
```

## 功能列表

| 模块 | 能力 |
|------|------|
| **策略引擎** | 多策略并行（双均线/布林带/动量突破/RSI/MACD/网格），信号自动防未来函数 |
| **回测引擎** | 事件驱动日频回测，交易成本/滑点/印花税，绩效指标，参数扫描 |
| **风险管理** | 止损/止盈/回撤暂停/仓位限制/单日亏损限额/Jev 置信度过滤 |
| **因子引擎** | 因子计算、IC 分析、因子组合、策略排行榜 |
| **ML 决策** | 本地 Jev 模型 HTTP 推理，概率分布输出，审计日志，自动降级 mock |
| **监控告警** | 实时日志、数据质量监控、异常预警、每日报告、Webhook 推送 |
| **数据层** | A股/美股/港股 K线，Parquet 缓存，自动代码标准化 |
| **部署** | Docker 容器化、docker-compose 编排、systemd 服务管理、健康检查探针 |

## 架构图

```
                    ┌─────────────────────────┐
                    │    Web 看板 (FastAPI)    │
                    │    :8766  server.py     │
                    └───────────┬─────────────┘
                                │
          ┌──────────┬──────────┼──────────┬──────────┐
          │          │          │          │          │
    ┌─────▼───┐ ┌───▼────┐ ┌──▼───┐ ┌───▼────┐ ┌──▼─────┐
    │ 策略引擎 │ │ 风控   │ │ 回测 │ │ 监控   │ │ 因子   │
    │strategies│ │ risk  │ │backtest│ │monitor│ │factors │
    └─────┬───┘ └───┬────┘ └──┬───┘ └───┬────┘ └──┬─────┘
          │          │          │          │          │
          └──────────┴────┬─────┴──────────┴──────────┘
                          │
                    ┌─────▼─────┐
                    │  数据层    │
                    │  data/    │
                    │  Parquet  │
                    │  缓存      │
                    └─────┬─────┘
                          │
              ┌───────────┴───────────┐
              │                       │
        ┌─────▼─────┐          ┌──────▼──────┐
        │ QuantDash │          │  Jev 决策   │
        │ 数据源SDK │          │  :8765     │
        └───────────┘          │ 本地ML推理  │
                               └─────────────┘
```

## 目录结构

```
quant_trading_system/
├── web-dashboard/          # FastAPI 服务入口（:8766）
│   └── server.py           #   主服务（看板 + API + WebSocket）
├── config/                 # 配置
│   ├── config.yaml         #   全局配置（策略/风控/数据源/Jev）
│   ├── stock_pool.yaml     #   股票池定义
│   └── __init__.py         #   load_config() + 环境变量覆盖
├── strategies/             # 策略引擎（多策略并行）
├── backtest/               # 回测引擎 + 绩效指标
├── risk/                   # 风险管理（止损/仓位/回撤）
├── factors/                # 因子计算与 IC 分析
├── jev/                    # Jev AI 决策模型集成
├── trading/                # 实盘接口（Broker 抽象 + 模拟盘）
├── monitoring/             # 监控告警 + 数据质量 + 日报
├── data/                   # 数据获取/清洗/缓存
├── security/               # Token 认证
├── persistence/            # 数据持久化与备份
├── optimization/            # 策略参数优化
├── analysis/               # 归因分析
├── scripts/                # 启动/停止/状态脚本
│   ├── start_all.sh
│   ├── stop_all.sh
│   └── status.sh
├── deploy/                 # 生产部署
│   ├── quant-trading.service  # systemd unit
│   └── install_service.sh    # 安装脚本
├── tests/                  # 单元测试（pytest）
├── Dockerfile              # Docker 镜像构建
├── docker-compose.yml      # 容器编排
├── .dockerignore
├── .env.example            # 环境变量模板
├── requirements.txt
├── DEPLOYMENT.md           # 完整部署指南
└── README.md
```

## API 文档

启动服务后访问：

- **Swagger UI**: http://localhost:8766/docs
- **ReDoc**: http://localhost:8766/redoc
- **健康检查**: http://localhost:8766/healthz （极简探针）
- **详细健康**: http://localhost:8766/api/health

## 部署

详见 [DEPLOYMENT.md](./DEPLOYMENT.md)，涵盖：

- 本地启动 / 脚本启动
- Docker 容器化部署（build / run / compose）
- Linux systemd 生产部署
- 环境变量配置与 .env 使用
- 健康检查与负载均衡配置
- 常见问题排查

## 注意事项

- 所有收益计算严格避免未来函数，信号均已延迟执行
- 回测结果不代表未来表现，需进行样本外测试验证
- 实盘前必须用模拟盘充分测试
- 本系统仅供学习研究使用，不构成投资建议
