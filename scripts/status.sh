#!/usr/bin/env bash
# 检查 Jev 决策服务(8765) 和量化交易服务(8766) 的运行状态
# 用法: bash scripts/status.sh
set -euo pipefail

JEV_PORT=8765
QUANT_PORT=8766
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "========================================"
echo "  量化交易系统 - 服务状态"
echo "========================================"
echo ""

check_service() {
    local port=$1
    local name=$2
    local health_url=$3

    printf "  %-12s ", "${name}:"

    # 检查端口监听
    local pid
    pid=$(lsof -ti ":${port}" -sTCP:LISTEN 2>/dev/null || true)
    if [ -z "${pid}" ]; then
        echo "✗ 未运行 (端口 ${port} 无监听)"
        return 1
    fi

    # 检查健康端点
    local health
    health=$(curl -s --max-time 5 "${health_url}" 2>/dev/null || true)
    if [ -z "${health}" ]; then
        echo "⚠ 运行中 (PID ${pid}) 但健康端点无响应"
    else
        # 提取状态
        local status
        status=$(echo "${health}" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('status','?'))" 2>/dev/null || echo "?")

        if [ "${name}" = "Jev服务" ]; then
            local model_loaded
            model_loaded=$(echo "${health}" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('model_loaded',False))" 2>/dev/null || echo "False")
            if [ "${model_loaded}" = "True" ]; then
                echo "✓ 运行中 (PID ${pid}, 模型已加载, real模式)"
            else
                echo "⚠ 运行中 (PID ${pid}, 模型加载中...)"
            fi
        else
            local strategies
            strategies=$(echo "${health}" | python3 -c "import sys,json; d=json.load(sys.stdin); data=d.get('data',d); print(','.join(data.get('strategies',[])))" 2>/dev/null || echo "?")
            echo "✓ 运行中 (PID ${pid}, 策略: ${strategies})"
        fi
    fi

    # ---- 进程 CPU/内存 ----
    local pid_first
    pid_first=$(echo "${pid}" | head -1)
    if [ -n "${pid_first}" ]; then
        local cpu_mem
        cpu_mem=$(ps -p "${pid_first}" -o %cpu,%mem,rss 2>/dev/null | tail -1 | xargs || true)
        if [ -n "${cpu_mem}" ]; then
            # shellcheck disable=SC2086
            set -- ${cpu_mem}
            local cpu=$1 mem=$2 rss_kb=$3
            local rss_mb
            rss_mb=$((rss_kb / 1024))
            printf "               CPU: %s%%  MEM: %s%%  RSS: %s MB\n" "${cpu}" "${mem}" "${rss_mb}"
        fi
    fi
    return 0
}

check_service "${JEV_PORT}" "Jev服务" "http://localhost:${JEV_PORT}/api/health"
echo ""
check_service "${QUANT_PORT}" "量化服务" "http://localhost:${QUANT_PORT}/api/health"

# ---- 最近日志 ----
echo ""
echo "----------------------------------------"
echo "  最近日志 (Quant, tail -20):"
echo "----------------------------------------"
if [ -f "/tmp/quant_server.log" ]; then
    tail -20 /tmp/quant_server.log
else
    echo "  (日志文件不存在: /tmp/quant_server.log)"
fi

# ---- 磁盘使用 ----
echo ""
echo "----------------------------------------"
echo "  磁盘使用:"
echo "----------------------------------------"
for dir in data output cache logs; do
    local_path="${PROJECT_ROOT}/${dir}"
    if [ -d "${local_path}" ]; then
        local size
        size=$(du -sh "${local_path}" 2>/dev/null | cut -f1 || echo "?")
        printf "  %-10s %s\n" "${dir}/" "${size}"
    else
        printf "  %-10s (不存在)\n" "${dir}/"
    fi
done

echo ""
echo "========================================"
echo "  快速链接"
echo "  Jev健康:    http://localhost:${JEV_PORT}/api/health"
echo "  量化健康:   http://localhost:${QUANT_PORT}/api/health"
echo "  存活探针:   http://localhost:${QUANT_PORT}/healthz"
echo "  看板:       http://localhost:${QUANT_PORT}/"
echo "========================================"
