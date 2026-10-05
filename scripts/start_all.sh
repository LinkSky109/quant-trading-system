#!/usr/bin/env bash
# 一键启动 Jev 决策服务(8765) + 量化交易服务(8766)
# 用法: bash scripts/start_all.sh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
JEV_DIR="/Users/link/myApp/ai/jev"
JEV_PYTHON="${JEV_DIR}/.venv/bin/python"
JEV_SERVER="${JEV_DIR}/jev_server.py"
QUANT_SERVER="${PROJECT_ROOT}/web-dashboard/server.py"
JEV_LOG="/tmp/jev_server.log"
QUANT_LOG="/tmp/quant_server.log"
JEV_PID_FILE="/tmp/jev_server.pid"
QUANT_PID_FILE="/tmp/quant_server.pid"
JEV_PORT=8765
QUANT_PORT=8766

echo "========================================"
echo "  量化交易系统 - 一键启动"
echo "========================================"

# ---- 环境检查：Python 版本 ----
echo ""
echo "[ENV] 检查 Python 版本..."
PY_VER=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
PY_MAJOR=$(echo "${PY_VER}" | cut -d. -f1)
PY_MINOR=$(echo "${PY_VER}" | cut -d. -f2)
if [ "${PY_MAJOR}" -lt 3 ] || { [ "${PY_MAJOR}" -eq 3 ] && [ "${PY_MINOR}" -lt 9 ]; }; then
    echo "[ERROR] 需要 Python >= 3.9，当前为 ${PY_VER}"
    exit 1
fi
echo "  ✓ Python ${PY_VER} (>= 3.9)"

# ---- 环境检查：关键依赖 ----
echo ""
echo "[ENV] 检查关键 Python 依赖..."
MISSING_DEPS=0
for pkg in pandas numpy yaml fastapi uvicorn; do
    if python3 -c "import ${pkg}" 2>/dev/null; then
        echo "  ✓ ${pkg}"
    else
        echo "  ✗ ${pkg} 未安装"
        MISSING_DEPS=1
    fi
done
if [ "${MISSING_DEPS}" -eq 1 ]; then
    echo "[WARN] 部分依赖缺失，请运行: pip install -r requirements.txt"
fi

# ---- 检查端口占用 ----
check_port() {
    local port=$1
    local name=$2
    if lsof -i ":${port}" -sTCP:LISTEN >/dev/null 2>&1; then
        echo "[WARN] 端口 ${port}(${name}) 已被占用，跳过启动"
        return 1
    fi
    return 0
}

# ---- 日志轮转提示 ----
echo ""
echo "[LOG] 日志文件:"
echo "  Jev:   ${JEV_LOG}"
echo "  Quant: ${QUANT_LOG}"
echo "  提示: 可用 logrotate 管理日志，或定期执行 > ${QUANT_LOG} 清空"

# ---- 1. 启动 Jev 服务 ----
echo ""
echo "[1/2] 启动 Jev 决策服务 (端口 ${JEV_PORT})..."
if check_port "${JEV_PORT}" "Jev"; then
    cd "${JEV_DIR}"
    nohup "${JEV_PYTHON}" "${JEV_SERVER}" > "${JEV_LOG}" 2>&1 &
    JEV_PID=$!
    echo "${JEV_PID}" > "${JEV_PID_FILE}"
    echo "  Jev PID: ${JEV_PID}, 日志: ${JEV_LOG}"
else
    JEV_PID=""
fi

# ---- 等待 Jev 模型加载 ----
echo ""
echo "  等待 Jev 模型加载（约 150 秒）..."
JEV_READY=0
for i in $(seq 1 60); do
    sleep 5
    if curl -s "http://localhost:${JEV_PORT}/api/health" 2>/dev/null | grep -q '"model_loaded":true'; then
        JEV_READY=1
        echo "  ✓ Jev 模型加载完成（等待 $((i * 5)) 秒）"
        break
    fi
    if [ $((i % 6)) -eq 0 ]; then
        echo "  ... 已等待 $((i * 5)) 秒，仍在加载中..."
    fi
done

if [ "${JEV_READY}" -eq 0 ]; then
    echo "[ERROR] Jev 模型加载超时（300秒），请检查 ${JEV_LOG}"
    echo "  量化服务将以 mock 模式启动"
fi

# ---- 2. 启动量化服务 ----
echo ""
echo "[2/2] 启动量化交易服务 (端口 ${QUANT_PORT})..."
if check_port "${QUANT_PORT}" "Quant"; then
    cd "${PROJECT_ROOT}"
    nohup python3 "${QUANT_SERVER}" > "${QUANT_LOG}" 2>&1 &
    QUANT_PID=$!
    echo "${QUANT_PID}" > "${QUANT_PID_FILE}"
    echo "  Quant PID: ${QUANT_PID}, 日志: ${QUANT_LOG}"
    # 等待健康端点响应（最多30秒）
    echo "  等待量化服务就绪..."
    QUANT_READY=0
    for i in $(seq 1 30); do
        sleep 1
        if curl -s "http://localhost:${QUANT_PORT}/healthz" 2>/dev/null | grep -q '"status":"ok"'; then
            QUANT_READY=1
            echo "  ✓ 量化服务已就绪（等待 ${i} 秒）"
            break
        fi
    done
    if [ "${QUANT_READY}" -eq 0 ]; then
        echo "[WARN] 量化服务健康检查超时（30秒），请检查 ${QUANT_LOG}"
    fi
else
    QUANT_PID=""
fi

# ---- 状态汇总 ----
echo ""
echo "========================================"
echo "  启动完成"
echo "========================================"
echo "  Jev 服务:   http://localhost:${JEV_PORT}/api/health"
echo "  量化服务:   http://localhost:${QUANT_PORT}/api/health"
echo "  健康检查:   http://localhost:${QUANT_PORT}/healthz"
echo "  看板页面:   http://localhost:${QUANT_PORT}/"
echo "  Jev日志:    ${JEV_LOG}"
echo "  Quant日志:  ${QUANT_LOG}"
echo "  Jev PID:    ${JEV_PID_FILE}"
echo "  Quant PID:  ${QUANT_PID_FILE}"
echo ""
if [ "${JEV_READY}" -eq 1 ]; then
    echo "  Jev 模式: real（真实模型推理）"
else
    echo "  Jev 模式: mock（Jev未就绪，自动降级）"
fi
echo "========================================"
