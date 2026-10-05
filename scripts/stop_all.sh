#!/usr/bin/env bash
# 一键停止 Jev 决策服务(8765) + 量化交易服务(8766)
# 用法: bash scripts/stop_all.sh
set -euo pipefail

JEV_PORT=8765
QUANT_PORT=8766
JEV_PID_FILE="/tmp/jev_server.pid"
QUANT_PID_FILE="/tmp/quant_server.pid"
GRACEFUL_TIMEOUT=15  # SIGTERM 后最多等待秒数

echo "========================================"
echo "  量化交易系统 - 一键停止"
echo "========================================"

# ---- 优雅停止：先 SIGTERM，等待 GRACEFUL_TIMEOUT 秒，再 SIGKILL ----
stop_service() {
    local pid_file=$1
    local port=$2
    local name=$3

    # 优先从 PID 文件读取
    local pids=""
    if [ -f "${pid_file}" ]; then
        pids=$(cat "${pid_file}" 2>/dev/null || true)
    fi

    # PID 文件不存在或已失效，回退到端口检测
    if [ -z "${pids}" ]; then
        pids=$(lsof -ti ":${port}" -sTCP:LISTEN 2>/dev/null || true)
    fi

    if [ -z "${pids}" ]; then
        echo "[INFO] ${name} (端口 ${port}) 未运行"
        rm -f "${pid_file}" 2>/dev/null || true
        return
    fi

    echo "[STOP] 停止 ${name} (端口 ${port}), PID(s): ${pids}"
    echo "${pids}" | xargs kill 2>/dev/null || true

    # 优雅等待：最多 GRACEFUL_TIMEOUT 秒
    local waited=0
    for i in $(seq 1 ${GRACEFUL_TIMEOUT}); do
        sleep 1
        waited=${i}
        local still_running=""
        still_running=$(echo "${pids}" | xargs ps -p 2>/dev/null | tail -n +1 | grep -v PID || true)
        if [ -z "${still_running}" ]; then
            echo "  ✓ ${name} 已停止（优雅退出，等待 ${waited} 秒）"
            rm -f "${pid_file}" 2>/dev/null || true
            return
        fi
    done

    # 超时，强制 SIGKILL
    echo "[WARN] ${name} 未在 ${GRACEFUL_TIMEOUT} 秒内响应 SIGTERM，发送 SIGKILL..."
    echo "${pids}" | xargs kill -9 2>/dev/null || true
    sleep 1
    rm -f "${pid_file}" 2>/dev/null || true
    echo "  ✓ ${name} 已强制停止"
}

echo ""
stop_service "${QUANT_PID_FILE}" "${QUANT_PORT}" "量化服务"
echo ""
stop_service "${JEV_PID_FILE}" "${JEV_PORT}" "Jev服务"

echo ""
echo "========================================"
echo "  所有服务已停止"
echo "========================================"
