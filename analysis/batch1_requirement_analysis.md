# 第1批需求深度分析报告

> 任务ID: TASK-20261002-001 | 工作流: quant-trading-v4 | 分析时间: 2026-10-02
> 基于对项目代码的全面探查（server.py 2800+行、HTML 2293行、security/、persistence/、trading/、backtest/ 等模块）

---

## 一、项目现状基线

### 1.1 技术栈与架构

| 维度 | 现状 |
|------|------|
| 后端框架 | FastAPI（单文件 server.py，2800+行），端口 8766 |
| 前端 | 单文件 quant_dashboard_realtime.html（2293行），ECharts 5.5.0，深色主题 |
| 数据库 | SQLite（persistence/database.py），WAL模式，5张表 |
| 认证 | security/auth.py — AuthManager，Bearer Token，三级权限 read/trade/admin |
| 配置 | config/config.yaml + config/__init__.py（load_config，支持环境变量覆盖） |
| 测试 | pytest，33个测试文件，778条全通过 |
| 中间件顺序 | api_stats_middleware → auth_middleware（限流应插在auth之后） |

### 1.2 现有API响应契约

```python
# 成功
{"code": 0, "message": "success", "data": {...}}
# 失败
{"code": <错误码>, "message": "...", "data": None}
```

辅助函数：`ok(data, message)` / `err(code, message, http_status)`

### 1.3 深色主题CSS变量

```css
--bg:#0b0f19; --bg2:#111627; --card:#161b2e; --card2:#1c2238;
--border:#252b45; --text:#e4e8f0; --text2:#8b93a7; --text3:#5c6478;
--up:#00d4aa; --down:#ff4d6a; --gold:#f0b90b; --blue:#4da6ff; --purple:#a78bfa;
```

### 1.4 现有SQLite表结构

| 表名 | 用途 | 关键字段 |
|------|------|----------|
| trades | 成交记录 | timestamp/symbol/side/price/quantity/realized_pnl/account_id |
| jev_decisions | Jev决策审计 | timestamp/symbol/probabilities_json/final_action/executed |
| account_snapshots | 账户快照 | timestamp/account_id/total_asset/cash/positions_json |
| daily_reports | 每日绩效 | date/account_id/daily_return/win_rate/total_pnl |
| walkthroughs | 走查结果 | id/symbol/strategy/snapshots_json/trades_json/meta_json |

### 1.5 回测区域现有Tab结构

```
历史回测分析
├── 绩效对比（active）— 净值曲线 + 指标对比表 + 交易明细
├── 无Jev过滤
├── 有Jev过滤
└── 逐日走查 — 已有基础实现（参数配置 + 快照表格 + 信号对比表）
```

另有独立的"多策略对比"区域（compareSection），已有基础实现（净值曲线叠加 + 8项指标对比表）。

---

## 二、REQ-P0-05 安全审计日志 — 深度分析

### 2.1 需求拆解

| 子项 | 具体内容 | 实现位置 |
|------|----------|----------|
| 新建模块 | security/audit.py，AuditLogger类 | 新文件 |
| SQLite表 | audit_logs新表，9个字段 | persistence/database.py 新增建表+写入+查询 |
| JSONL文件 | logs/audit.jsonl，追加模式 | AuditLogger内部 |
| 哈希链 | 每条记录含前一条hash，可验证完整性 | AuditLogger内部 |
| 操作枚举 | 10种类型 | 模块内Enum |
| 记录时机 | 交易/配置/策略/风控/备份 | 多处集成点 |
| 查询API | 2个端点 | server.py 新增路由 |
| 配置 | config.yaml新增audit节 | config.yaml |

### 2.2 audit_logs表设计

```sql
CREATE TABLE IF NOT EXISTS audit_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,           -- ISO8601
    operator TEXT NOT NULL DEFAULT '', -- token名称或"system"
    action_type TEXT NOT NULL,         -- 10种枚举之一
    target TEXT DEFAULT '',            -- 操作对象（symbol/配置项/策略名）
    params TEXT DEFAULT '{}',          -- JSON字符串，操作参数
    result TEXT DEFAULT '',            -- success/failed + 详情
    ip TEXT DEFAULT '',                -- 请求IP
    request_id TEXT DEFAULT '',        -- 请求追踪ID
    prev_hash TEXT DEFAULT '',         -- 前一条记录的哈希（哈希链）
    hash TEXT NOT NULL                 -- 本条记录的哈希
);
CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_logs(timestamp);
CREATE INDEX IF NOT EXISTS idx_audit_action ON audit_logs(action_type);
CREATE INDEX IF NOT EXISTS idx_audit_operator ON audit_logs(operator);
```

### 2.3 哈希链设计

- **哈希算法**: SHA-256
- **哈希输入**: `timestamp + operator + action_type + target + params + result + ip + request_id + prev_hash`
- **首条记录**: prev_hash = "0" * 64（创世哈希）
- **验证逻辑**: 从头遍历，逐条重新计算hash并与存储的hash对比，同时验证prev_hash指向上一条的hash
- **JSONL同步**: 每条记录同时写入JSONL（含hash字段），SQLite和JSONL互为备份

### 2.4 操作类型枚举与集成点

| 枚举值 | 触发位置 | 记录内容 |
|--------|----------|----------|
| LOGIN | auth_middleware验证通过时 | operator=token名称, ip |
| LOGOUT | （当前无登出端点，预留） | — |
| ORDER_SUBMIT | broker.submit_order() | target=symbol, params={action,quantity,order_type,price} |
| ORDER_CANCEL | broker.cancel_order() | target=order_id |
| POSITION_CLOSE | 平仓操作（broker层） | target=symbol, params={shares,reason} |
| CONFIG_CHANGE | 配置变更（当前无热加载端点，预留） | target=配置路径 |
| STRATEGY_TOGGLE | POST /api/trade_toggle | target=策略名, params={running} |
| RISK_TRIGGER | risk_manager触发拦截时 | target=symbol, params={rule,reason} |
| BACKUP_RESTORE | POST /api/backup/restore | target=备份文件名 |
| API_TOKEN_CREATE | （当前token在config.yaml静态配置，预留） | — |

**关键发现**: 当前系统中，交易操作主要发生在回测引擎（backtest/engine.py）和实时交易器（realtime_trader.py），broker.py的SimulatedBroker.submit_order是核心入口。审计日志应在**broker层**包装，而非在每个调用点散落记录。

### 2.5 查询API设计

```
GET /api/audit/logs?start=&end=&action_type=&operator=&page=1&page_size=50
→ 返回: {total, page, page_size, logs: [...]}

GET /api/audit/verify
→ 返回: {valid: bool, total_records: int, first_hash: str, last_hash: str,
         broken_at: null|int, error: null|str}
```

### 2.6 配置项

```yaml
audit:
  enabled: true
  log_file: "./logs/audit.jsonl"
  retention_days: 90          # 日志保留天数
  hash_algorithm: "sha256"
```

### 2.7 风险与注意事项

1. **性能**: 哈希计算+双写（SQLite+JSONL）在高频交易场景下有延迟，建议异步写入或批量提交
2. **线程安全**: AuditLogger必须线程安全（threading.Lock保护哈希链状态和文件写入）
3. **首次启动**: 数据库中无记录时，创世记录的prev_hash处理
4. **与现有Database类的关系**: audit_logs表应加入Database._init_db()，但AuditLogger可独立持有db引用
5. **operator获取**: 在API层可从request.state.token_info获取operator名称；在broker层需要通过上下文传递

---

## 三、REQ-P0-06 API限流 — 深度分析

### 3.1 需求拆解

| 子项 | 具体内容 | 实现位置 |
|------|----------|----------|
| 新建模块 | security/rate_limit.py，RateLimiter类 | 新文件 |
| 算法 | 令牌桶，按IP或API Token限流 | 模块内 |
| 分级配置 | 4级：read/trade/admin/backtest | config.yaml |
| 中间件 | FastAPI middleware，auth之后执行 | server.py |
| 超限响应 | 429 + Retry-After头 + 统一错误格式 | 中间件内 |
| 白名单 | localhost/127.0.0.1不限流 | 中间件内 |
| 审计集成 | 限流事件计入审计日志（依赖P0-05） | 中间件内调用AuditLogger |
| 状态API | GET /api/rate_limit/status | server.py 新增路由 |

### 3.2 令牌桶算法设计

```python
class TokenBucket:
    capacity: int          # 桶容量（每分钟配额）
    tokens: float          # 当前令牌数
    refill_rate: float     # 每秒补充速率 = capacity/60
    last_refill: float     # 上次补充时间戳

    def consume(self, tokens=1) -> tuple[bool, float]:
        """返回(是否允许, 需等待秒数)"""
```

**限流维度**: 优先按API Token限流（认证通过后），未认证时按IP限流。每个维度+每个分级维护独立的TokenBucket。

### 3.3 分级限流与路径映射

| 分级 | 配额 | 路径匹配规则 |
|------|------|-------------|
| read | 100次/分钟 | 默认所有 /api/* GET 请求 |
| trade | 20次/分钟 | POST /api/trade_toggle, 及未来交易端点 |
| admin | 10次/分钟 | /api/alerts/trigger, /api/daily_reports/export, /api/backup/* |
| backtest | 5次/分钟 | POST /api/backtest, /api/backtest_compare, /api/backtest_portfolio, /api/walkthrough/run, /api/optimize/* |

**路径分类逻辑**: 需在中间件中根据path+method判断分级，建议维护一个路径前缀→分级的映射表，与现有`_required_permission_for_path`类似但独立。

### 3.4 中间件执行顺序

```
请求 → api_stats_middleware → auth_middleware → rate_limit_middleware → 路由
```

**为什么在auth之后**: 
1. 限流维度优先使用token（已认证用户），auth中间件已将token_info挂到request.state
2. 未认证请求按IP限流，不消耗已认证用户的配额
3. 白名单检查在限流之前

### 3.5 超限响应

```python
JSONResponse(
    status_code=429,
    headers={"Retry-After": str(wait_seconds)},
    content={"code": 429, "message": "Too Many Requests", "data": None}
)
```

### 3.6 白名单

```python
_WHITELIST_IPS = {"127.0.0.1", "localhost", "::1"}
# 从 request.client.host 获取IP
```

### 3.7 状态API

```
GET /api/rate_limit/status
→ 返回: {
    "enabled": true,
    "tiers": {"read": 100, "trade": 20, "admin": 10, "backtest": 5},
    "current": {
        "<token_or_ip>": {"read": {"remaining": 95, "reset_in": 12.5}, ...}
    },
    "whitelist": ["127.0.0.1", "localhost"]
}
```

### 3.8 配置项

```yaml
rate_limit:
  enabled: false           # 默认关闭，不影响开发环境
  tiers:
    read: 100              # 次/分钟
    trade: 20
    admin: 10
    backtest: 5
  whitelist:
    - "127.0.0.1"
    - "localhost"
  default_tier: "read"     # 未匹配路径的默认分级
```

### 3.9 依赖关系

**REQ-P0-06 依赖 REQ-P0-05**: 限流事件（429触发）需要计入审计日志（action_type可新增RATE_LIMIT_BLOCKED，或归入现有枚举）。建议P0-05先完成，P0-06在中间件中调用AuditLogger。

### 3.10 风险与注意事项

1. **内存泄漏**: TokenBucket按token/IP创建，长期运行会积累大量桶，需LRU淘汰或定期清理
2. **时钟回拨**: 令牌补充依赖time.time()，时钟回拨可能导致令牌异常，需做保护
3. **WebSocket**: 当前WebSocket端点/ws是否需要限流？需求未明确，建议暂不限制WS
4. **与api_stats的关系**: 限流统计可复用ApiMetricsCollector的数据
5. **多进程**: 当前单进程运行，令牌桶在内存中即可；若未来多进程需改用Redis

---

## 四、REQ-P1-01 回测走查器前端 — 深度分析

### 4.1 现状评估

**已有实现**（WalkthroughModule，HTML第2060-2137行）:
- ✅ 走查参数配置（标的/策略/日期范围）
- ✅ 调用 POST /api/walkthrough/run
- ✅ 快照表格（日期/收盘价/信号/Jev/操作/成交价/持仓/现金/总资产/当日盈亏）
- ✅ 交易事件与信号对比表
- ✅ 基本的Jev过滤/风控拦截颜色标记

**缺失功能**（需求要求但未实现）:
- ❌ 日期选择器（单日照看，范围限制在回测区间内）
- ❌ 上一日/下一日按钮（← →）
- ❌ 跳转到首笔/最后一笔交易按钮
- ❌ 详细信息面板（Jev概率分布、风控检查详情、持仓变化、累计盈亏）
- ❌ 被过滤信号的行级高亮（黄色Jev/红色风控/绿色正常）
- ❌ 当日K线图（小型，标注买卖点）
- ❌ 前端内存缓存（避免重复API调用）
- ❌ use_jev开关（当前硬编码false）

### 4.2 走查API数据结构

**POST /api/walkthrough/run** 请求:
```json
{"symbol": "600519.SH", "strategy": "ma_cross", "start_date": "2025-01-01",
 "end_date": "2025-12-31", "initial_capital": 1000000, "use_jev": false}
```

**响应**:
```json
{"walkthrough_id": "abc123", "summary": {"snapshots_count": 240, "trades_count": 15, ...}}
```

**GET /api/walkthrough/{id}/day?date=YYYY-MM-DD** 响应（单日快照）:
```json
{
  "date": "2025-06-15",
  "close": {"600519.SH": 1680.50},
  "signals": [{
    "symbol": "600519.SH", "signal": "buy", "confidence": 0.75,
    "executed": true, "jev_filtered": false, "risk_blocked": false,
    "jev_decision": {"final_action": "buy", "probabilities": {"buy":0.6,"sell":0.1,"hold":0.3}, "confidence": 0.65, "reason": "..."},
    "action": "buy", "fill_price": 1680.50, "shares": 100, "reason": ""
  }],
  "positions": {"600519.SH": {"shares": 100, "cost_price": 1680.50, "market_value": 168050}},
  "cash": 831950.0, "total_equity": 1000000.0,
  "daily_pnl": 2500.0, "cumulative_pnl": 15000.0
}
```

**注意**: 快照中`close`是dict（key为symbol），`positions`也是dict，前端需正确解析。

### 4.3 前端改造方案

#### 4.3.1 布局重构

将现有"逐日走查"区域从纯表格改为**主控+详情**布局:

```
走查参数栏（标的/策略/日期范围/use_jev开关/运行按钮）
├── 走查控制栏（新增）
│   ├── 日期选择器 <input type="date" min=回测起始 max=回测结束>
│   ├── ← 上一日 / 下一日 → 按钮
│   ├── ⏮ 首笔交易 / 末笔交易 ⏭ 按钮
│   └── 当前位置指示: 第N天 / 共M天
├── 当日详情面板（新增，表格形式）
│   ├── 日期/收盘价
│   ├── 策略信号（buy/sell/hold + 置信度进度条）
│   ├── Jev决策（概率分布柱状 + 最终动作 + 置信度）
│   ├── 风控检查（通过✓/拦截✗ + 原因）
│   ├── 最终操作（买入↑/卖出↓/无操作— + 价格 + 数量）
│   ├── 持仓变化（数量/成本价/市值）
│   └── 资金（现金/总资产/当日盈亏/累计盈亏）
├── 当日K线图（新增，ECharts小型K线，标注买卖点）
├── 全量快照表格（保留，行级高亮）
│   └── Jev过滤行=黄色背景, 风控拦截行=红色背景, 正常执行行=绿色文字
└── 交易事件对比表（保留）
```

#### 4.3.2 前端缓存设计

```javascript
// WalkthroughModule内部
let _cache = {
    walkthrough_id: null,
    dayCache: {},           // {date: snapshot}
    rangeCache: null,       // 全量快照列表
    tradesCache: null,      // 交易列表
    summary: null
};

async function getDay(date) {
    if (_cache.dayCache[date]) return _cache.dayCache[date];
    const data = await api(`/api/walkthrough/${id}/day?date=${date}`);
    _cache.dayCache[date] = data;
    return data;
}
```

#### 4.3.3 K线图数据获取

当前走查API的单日快照不含K线数据。需要:
- **方案A**: 从manager.sims[symbol].klines中取当日附近N根K线（前端无法直接访问，需新API或从range快照中提取收盘价）
- **方案B**: 调用现有 GET /api/klines?symbol=... 获取K线，前端筛选日期范围
- **推荐方案B**: 复用现有/api/klines端点，在走查运行成功后预加载K线数据到缓存

#### 4.3.4 高亮规则

| 状态 | 视觉效果 |
|------|----------|
| 正常执行（executed=true） | 文字颜色 var(--up) 绿色 |
| Jev过滤（jev_filtered=true） | 背景色 rgba(240,185,11,.12) 黄色，文字 var(--gold) |
| 风控拦截（risk_blocked=true） | 背景色 rgba(255,77,106,.12) 红色，文字 var(--down) |
| 无信号 | 默认文字颜色 |

### 4.4 风险与注意事项

1. **HTML文件冲突**: P1-01和P1-02都修改同一个HTML文件，必须串行执行或严格划分修改区域
2. **快照数据字段**: cumulative_pnl字段在walkthrough.py中未明确返回，需确认BacktestEngine的快照是否包含此字段；若缺失需在后端补充
3. **日期导航边界**: 上一日/下一日需基于实际交易日（快照列表中的日期），而非自然日
4. **ECharts实例管理**: 新增K线图需用现有getChart()函数管理，避免重复初始化
5. **use_jev开关**: 当前runWalkthrough硬编码use_jev:false，需增加checkbox

---

## 五、REQ-P1-02 策略对比仪表盘 — 深度分析

### 5.1 现状评估

**已有实现**（CompareModule，HTML第1471-1575行）:
- ✅ 对比参数配置（标的/日期范围/策略多选checkbox）
- ✅ 调用 POST /api/backtest_compare
- ✅ 净值曲线叠加图（含基准）
- ✅ 8项指标对比表（累计收益率/年化收益率/最大回撤/夏普比率/胜率/盈亏比/交易次数/总盈利）
- ✅ 加载状态提示

**缺失功能**（需求要求但未实现）:
- ❌ 指标雷达图（6维归一化）
- ❌ 交易散点图（X=持仓天数, Y=盈亏%）
- ❌ 月度收益热力图
- ❌ 策略相关性矩阵（日收益率相关系数）
- ❌ 9项核心指标（当前8项，缺1项）
- ❌ 进度条（当前只有文字loading）
- ❌ 策略默认全选（当前默认选前2个）

### 5.2 后端API分析

**POST /api/backtest_compare** 已存在，返回:
```json
{
  "symbol": "600519.SH", "start_date": "...", "end_date": "...",
  "benchmark_curve": [["2025-01-02", 1000000], ...],
  "results": [
    {
      "strategy": "ma_cross", "label": "双均线 MA5/MA20",
      "metrics": {"累计收益率": 0.15, "年化收益率": 0.12, "最大回撤": 0.08,
                  "夏普比率": 1.2, "胜率": 0.55, "盈亏比": 1.8,
                  "交易次数": 30, "总盈利": 150000, "总亏损": -80000,
                  "订单成交率": 0.95, "平均持仓时间": 5.2},
      "equity_curve": [["2025-01-02", 1000000], ...],
      "trades": [{"date": "...", "action": "buy", "price": ..., "shares": ..., "pnl": ...}, ...]
    }
  ]
}
```

**关键发现**: API返回的trades列表包含每笔交易的date/action/price/shares/pnl，但**不含持仓天数**。交易散点图需要的"持仓天数"需要前端计算：对每笔sell交易，找到对应的前一笔buy交易，计算日期差。

### 5.3 五图一表详细设计

#### 5.3.1 净值曲线叠加图（已有，增强）

- 保持现有实现
- 增强: 支持更多策略颜色（PALETTE已有5色，需扩展到9色）
- 增强: 图例可点击隐藏/显示

#### 5.3.2 指标雷达图（新增）

**6个维度**: 累计收益、年化收益、夏普比率、最大回撤倒数、胜率、盈亏比

**归一化方法**: 对每个维度，取所有策略中的最大值作为1.0，其他策略按比例缩放。最大回撤取倒数（回撤越小越好）。

```javascript
const dims = ['累计收益率', '年化收益率', '夏普比率', '最大回撤', '胜率', '盈亏比'];
// 最大回撤特殊处理：值越小越好，归一化用 1 - (value/max) 或 1/value
const normalized = results.map(r => {
    return dims.map(d => {
        const vals = results.map(x => x.metrics[d] || 0);
        const max = Math.max(...vals.map(Math.abs));
        let v = r.metrics[d] || 0;
        if (d === '最大回撤') v = max > 0 ? (max - Math.abs(v)) / max : 0;
        else v = max > 0 ? v / max : 0;
        return Math.max(0, Math.min(1, v));
    });
});
```

ECharts radar配置，深色主题适配。

#### 5.3.3 交易散点图（新增）

**数据计算**:
```javascript
// 对每个策略的trades，配对buy/sell计算持仓天数和盈亏%
function calcTradeScatter(trades) {
    const points = [];
    let openTrades = []; // {date, price, shares}
    trades.forEach(t => {
        if (t.action === 'buy') {
            openTrades.push({date: t.date, price: t.price, shares: t.shares});
        } else if (t.action === 'sell' && openTrades.length > 0) {
            const buy = openTrades.shift(); // FIFO配对
            const holdDays = (new Date(t.date) - new Date(buy.date)) / 86400000;
            const pnlPct = t.pnl ? (t.pnl / (buy.price * buy.shares)) * 100 : 0;
            points.push([holdDays, pnlPct]);
        }
    });
    return points;
}
```

ECharts scatter配置，不同策略不同颜色，X轴=持仓天数，Y轴=盈亏%。

#### 5.3.4 月度收益热力图（新增）

**数据计算**: 从equity_curve计算月度收益率
```javascript
// equity_curve: [[date, value], ...]
// 按月分组，取每月最后一个交易日的净值，计算月环比收益率
function calcMonthlyReturns(equityCurve) {
    const monthly = {}; // {"2025-01": {start: 1000000, end: 1050000}}
    equityCurve.forEach(([date, value]) => {
        const month = date.substring(0, 7);
        if (!monthly[month]) monthly[month] = {first: value, last: value};
        monthly[month].last = value;
        if (!monthly[month].firstSet) { monthly[month].first = value; monthly[month].firstSet = true; }
    });
    // 计算月收益率
    return Object.entries(monthly).map(([m, v]) => ({
        month: m, return: ((v.last - v.first) / v.first) * 100
    }));
}
```

**可切换策略**: 下拉选择查看哪个策略的月度热力图。
ECharts heatmap配置，X=月份，Y=年份（如果跨年），颜色=收益率（红涨绿跌或绿涨红跌，按A股习惯红涨绿跌）。

#### 5.3.5 策略相关性矩阵（新增）

**数据计算**: 各策略日收益率的相关系数
```javascript
// 从equity_curve计算日收益率序列，按日期对齐
// 计算Pearson相关系数矩阵
function correlationMatrix(results) {
    // 1. 构建 {date: {strategy: daily_return}} 矩阵
    // 2. 对每对策略计算Pearson相关系数
    // 3. 返回 N×N 矩阵
}
```

ECharts heatmap配置，X/Y轴=策略名，颜色=相关系数（-1到1）。

#### 5.3.6 对比指标表（增强）

从8项扩展到9项，新增"卡尔玛比率"（年化收益/最大回撤）或"平均持仓时间"。
建议新增**"平均持仓时间"**（metrics中已有此字段）。

9项: 累计收益率、年化收益率、最大回撤、夏普比率、胜率、盈亏比、交易次数、总盈利、平均持仓时间。

### 5.4 进度条设计

多策略回测是串行执行的（后端for循环），当前无法获取实时进度。前端方案:
- 显示不确定进度条（indeterminate）+ "正在回测策略 X/N" 文字
- 由于后端不支持进度推送，只能显示总策略数和已完成数（通过分块请求实现，但会增加复杂度）
- **简化方案**: 显示动画进度条 + "对比回测运行中（N个策略），请稍候..."

### 5.5 风险与注意事项

1. **计算复杂度**: 相关性矩阵和月度收益在前端计算，策略数×交易日数较大时可能卡顿，建议策略数上限设为6-8个
2. **ECharts实例**: 5个图表需要妥善管理，切换Tab时resize，隐藏时不渲染
3. **交易配对逻辑**: FIFO配对在加仓/减仓场景下不准确，但回测中通常是全买全卖，可接受
4. **颜色扩展**: PALETTE只有5色，9个策略需要扩展到9色
5. **与P1-01的HTML冲突**: 两个需求都修改quant_dashboard_realtime.html，必须协调

---

## 六、依赖关系与执行顺序

### 6.1 依赖图

```
REQ-P0-05 (审计日志) ◄─── REQ-P0-06 (API限流，限流事件需审计)
     │
     └── 独立后端模块，无前端依赖

REQ-P1-01 (走查器前端) ──┐
                          ├── 同一HTML文件，需串行执行
REQ-P1-02 (策略对比前端) ──┘
```

### 6.2 推荐执行顺序

| 阶段 | 需求 | 原因 |
|------|------|------|
| 第1步 | REQ-P0-05 审计日志 | P0优先级，且P0-06依赖它 |
| 第2步 | REQ-P0-06 API限流 | P0优先级，依赖P0-05的AuditLogger |
| 第3步 | REQ-P1-01 走查器前端 | P1，修改HTML前半部分（walkthrough区域） |
| 第4步 | REQ-P1-02 策略对比前端 | P1，修改HTML后半部分（compare区域） |

**并行可能性**: P0-05和P1-01/P1-02可以并行（后端vs前端，不同文件）。但P0-06必须等P0-05。前端两个必须串行。

### 6.3 server.py协调点

两个后端需求都需要修改server.py:
- P0-05: 新增2个API路由 + broker层审计集成 + 初始化AuditLogger
- P0-06: 新增1个中间件 + 1个API路由 + 初始化RateLimiter

**建议**: P0-05先完成server.py修改，P0-06在其基础上添加。中间件注册顺序需注意：rate_limit_middleware必须在auth_middleware之后定义（FastAPI中间件按注册顺序逆序执行？需确认——实际上FastAPI中间件按代码定义顺序执行，先定义的先执行）。

**FastAPI中间件执行顺序确认**: FastAPI的`@app.middleware("http")`按**注册顺序**执行（先注册的外层，后注册的内层）。当前顺序: api_stats（先注册，最外层）→ auth（后注册，内层）。新增rate_limit应在auth之后注册，即执行顺序为 api_stats → auth → rate_limit → 路由。✅

---

## 七、测试策略

### 7.1 新增测试文件

| 测试文件 | 覆盖内容 | 预估用例数 |
|----------|----------|-----------|
| tests/test_audit.py | AuditLogger写入/哈希链/查询/完整性校验 | 12-15 |
| tests/test_rate_limit.py | 令牌桶/分级限流/429响应/白名单/中间件集成 | 10-12 |

### 7.2 测试模式参考

现有test_auth.py的模式:
- 纯单元测试：直接实例化类，调用方法断言
- 中间件集成测试：最小FastAPI App + TestClient
- 不依赖真实config.yaml，全部构造配置

### 7.3 全量测试目标

当前778条 → 新增约25条 → 目标803条全通过

---

## 八、配置变更汇总

### config.yaml新增内容

```yaml
# 审计日志配置
audit:
  enabled: true
  log_file: "./logs/audit.jsonl"
  retention_days: 90

# API限流配置
rate_limit:
  enabled: false
  tiers:
    read: 100
    trade: 20
    admin: 10
    backtest: 5
  whitelist:
    - "127.0.0.1"
    - "localhost"
  default_tier: "read"
```

---

## 九、工作量评估

| 需求 | 代码行数(估) | 测试用例 | 复杂度 | 风险点 |
|------|-------------|----------|--------|--------|
| REQ-P0-05 审计日志 | 350-450行 | 12-15 | 中 | 哈希链线程安全、broker层集成 |
| REQ-P0-06 API限流 | 200-300行 | 10-12 | 中低 | 令牌桶内存管理、中间件顺序 |
| REQ-P1-01 走查器前端 | 300-400行(HTML/JS) | 0 | 中 | HTML冲突、K线数据获取 |
| REQ-P1-02 策略对比前端 | 350-450行(HTML/JS) | 0 | 中高 | 5个ECharts图、前端计算复杂度 |

---

## 十、验收标准清单

### REQ-P0-05
- [ ] security/audit.py存在，AuditLogger类完整
- [ ] audit_logs表创建成功，9+2字段（含prev_hash/hash）
- [ ] logs/audit.jsonl同步写入，追加模式
- [ ] 哈希链验证通过（故意篡改一条会检测到）
- [ ] 10种操作类型枚举完整
- [ ] broker层交易操作自动记录审计
- [ ] GET /api/audit/logs分页查询正常
- [ ] GET /api/audit/verify完整性校验正常
- [ ] test_audit.py全部通过
- [ ] config.yaml audit节配置生效

### REQ-P0-06
- [ ] security/rate_limit.py存在，RateLimiter类完整
- [ ] 令牌桶算法正确（令牌随时间补充）
- [ ] 4级限流配置生效
- [ ] 超限返回429 + Retry-After头
- [ ] 白名单IP不限流
- [ ] 限流中间件在auth之后执行
- [ ] 限流事件计入审计日志
- [ ] GET /api/rate_limit/status正常
- [ ] test_rate_limit.py全部通过
- [ ] config.yaml rate_limit节配置生效

### REQ-P1-01
- [ ] 走查Tab可正常打开和运行
- [ ] 日期选择器范围限制在回测区间内
- [ ] 上一日/下一日按钮正常导航
- [ ] 首笔/末笔交易跳转正常
- [ ] 详情面板显示所有要求字段
- [ ] Jev过滤行黄色高亮、风控拦截行红色高亮、正常执行绿色
- [ ] 当日K线图显示并标注买卖点
- [ ] 走查结果前端缓存生效
- [ ] 深色主题样式一致
- [ ] 不破坏现有回测功能

### REQ-P1-02
- [ ] 策略对比Tab可正常使用
- [ ] 策略多选checkbox默认全选
- [ ] 净值曲线叠加图正常渲染
- [ ] 雷达图6维归一化正确
- [ ] 交易散点图X/Y轴数据正确
- [ ] 月度收益热力图可切换策略
- [ ] 相关性矩阵热力图正确
- [ ] 9项指标对比表完整
- [ ] 加载进度条显示
- [ ] 深色主题样式一致

---

*分析完成。4项需求均已拆解到可执行粒度，依赖关系明确，可进入开发阶段。*
