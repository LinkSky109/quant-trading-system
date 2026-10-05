# 量化交易系统部署指南

本文档覆盖从开发环境到生产环境的完整部署流程。

---

## 1. 环境要求

| 项目 | 要求 |
|------|------|
| Python | >= 3.9（推荐 3.11） |
| 操作系统 | Linux / macOS（Windows 需 WSL2） |
| 内存 | >= 2 GB（Jev 模型加载时建议 >= 4 GB） |
| 磁盘 | >= 5 GB（含数据缓存与回测输出） |
| 端口 | 8766（量化服务）、8765（Jev 服务，可选） |

### 核心依赖

```bash
pip install -r requirements.txt
```

关键包：`pandas>=2.0`、`numpy>=1.24`、`fastapi`、`uvicorn`、`pyyaml`、`requests`、`pyarrow`、`matplotlib`、`plotly`。

---

## 2. 本地启动

### 方式一：直接启动

```bash
# 仅启动量化服务（端口 8766）
cd quant_trading_system
python3 web-dashboard/server.py
```

访问 http://localhost:8766

### 方式二：一键脚本启动

```bash
# 启动 Jev(8765) + 量化(8766)
bash scripts/start_all.sh

# 查看状态
bash scripts/status.sh

# 停止
bash scripts/stop_all.sh
```

脚本会自动：
- 检查 Python 版本与依赖
- 写入 PID 文件到 `/tmp/quant_server.pid`、`/tmp/jev_server.pid`
- 等待 `/healthz` 端点就绪后再报告成功
- 日志输出到 `/tmp/quant_server.log`、`/tmp/jev_server.log`

---

## 3. Docker 部署

### 构建与运行

```bash
# 构建镜像
docker build -t quant-trading .

# 运行容器
docker run -d \
  --name quant-trading \
  -p 8766:8766 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/output:/app/output \
  -v $(pwd)/config:/app/config:ro \
  -v $(pwd)/logs:/app/logs \
  -e QUANT_PORT=8766 \
  -e TZ=Asia/Shanghai \
  --restart unless-stopped \
  quant-trading
```

### docker-compose 一键编排

```bash
# 仅启动量化服务
docker-compose up -d quant

# 同时启动 Jev 服务（需要 jev profile）
docker-compose --profile jev up -d

# 查看日志
docker-compose logs -f quant

# 停止
docker-compose down
```

### 数据卷说明

| 宿主机路径 | 容器路径 | 用途 |
|-----------|---------|------|
| `./data` | `/app/data` | 行情数据、备份 |
| `./output` | `/app/output` | 回测图表与报告 |
| `./models` | `/app/models` | 模型文件（预留） |
| `./config` | `/app/config` | 配置文件（只读挂载） |
| `./logs` | `/app/logs` | 运行日志 |

---

## 4. systemd 部署（Linux 生产环境）

适用于 Linux 服务器长期运行。

### 安装服务

```bash
# 复制 service 文件到 systemd 目录
sudo cp deploy/quant-trading.service /etc/systemd/system/

# 重载并启动
sudo systemctl daemon-reload
sudo systemctl enable quant-trading
sudo systemctl start quant-trading
```

或使用安装脚本：

```bash
bash deploy/install_service.sh
```

### 常用命令

```bash
# 查看状态
systemctl status quant-trading

# 查看实时日志
journalctl -u quant-trading -f

# 重启
sudo systemctl restart quant-trading

# 停止
sudo systemctl stop quant-trading

# 开机自启（已通过 enable 配置）
sudo systemctl disable quant-trading   # 取消自启
```

> **注意**: `quant-trading.service` 中的 `User=link` 和路径需根据实际服务器环境修改。

---

## 5. 配置说明

### config.yaml 核心项

| 配置项 | 说明 |
|-------|------|
| `data.api_key` | QuantDash 数据源 API Key |
| `jev.base_url` | Jev 决策服务地址（默认 `http://localhost:8765`） |
| `risk.*` | 风控参数（止损/止盈/回撤/仓位） |
| `strategies.*` | 各策略开关与参数 |
| `security.enabled` | 是否开启 Token 认证（默认 false） |

### 环境变量列表

环境变量优先级高于 config.yaml，适合容器化部署。

| 环境变量 | 对应配置 | 说明 |
|---------|---------|------|
| `QUANT_PORT` | （特殊） | 服务端口，存入 `cfg['_env_port']` |
| `QUANT_JEV_URL` | `jev.base_url` | Jev 服务地址 |
| `QUANT_API_KEY` | `data.api_key` | 数据源 API Key |
| `QUANT_LOG_LEVEL` | `logging.level` | 日志级别 |
| `TZ` | （系统） | 时区，容器内设置 |

### .env 使用

```bash
cp .env.example .env
# 编辑 .env 填入实际值
```

---

## 6. 健康检查

系统提供两个健康端点，用途不同：

| 端点 | 认证 | 返回 | 用途 |
|------|------|------|------|
| `GET /healthz` | 公开 | `{"status":"ok"}` | 极简存活探针，Docker HEALTHCHECK / k8s livenessProve / 负载均衡 |
| `GET /api/health` | 公开 | 完整状态 JSON | 深度健康检查：策略列表、Jev 状态、运行时间、账户信息 |

### 负载均衡配置示例

```nginx
# Nginx 健康检查
location /healthz {
    proxy_pass http://quant_backend/healthz;
}
```

```yaml
# Kubernetes livenessProbe
livenessProbe:
  httpGet:
    path: /healthz
    port: 8766
  initialDelaySeconds: 30
  periodSeconds: 30
```

---

## 7. 常见问题

### 端口被占用

```bash
# 查看占用进程
lsof -i :8766

# 停止旧进程
kill $(lsof -ti :8766)
```

### 依赖缺失

```bash
pip install -r requirements.txt
# 如果 numpy/pandas 编译失败，确保 Python >= 3.9
```

### 数据缓存问题

缓存目录在 `./cache/`，Parquet 格式。如遇数据异常：

```bash
rm -rf ./cache/*.parquet   # 清空缓存，下次请求自动重建
```

### 日志排查

```bash
# 实时日志
tail -f /tmp/quant_server.log

# 最近 100 行
tail -100 /tmp/quant_server.log

# systemd 日志
journalctl -u quant-trading -n 200
```

### Jev 服务不可用

量化服务会自动降级为 mock 模式，不影响启动。如需真实 Jev 推理，确保 Jev 服务（端口 8765）已启动且模型加载完成。

```bash
curl http://localhost:8765/api/health | grep model_loaded
```
