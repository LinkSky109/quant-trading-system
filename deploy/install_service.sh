#!/usr/bin/env bash
# 安装 quant-trading systemd 服务
# 用法: bash deploy/install_service.sh
set -euo pipefail

SERVICE_FILE="$(dirname "$0")/quant-trading.service"
TARGET_DIR="/etc/systemd/system"
SERVICE_NAME="quant-trading.service"

echo "安装 ${SERVICE_NAME}..."
sudo cp "${SERVICE_FILE}" "${TARGET_DIR}/"
sudo systemctl daemon-reload
sudo systemctl enable "${SERVICE_NAME}"
sudo systemctl start "${SERVICE_NAME}"
echo "✓ 服务已安装并启动"
echo "  状态: systemctl status ${SERVICE_NAME}"
echo "  日志: journalctl -u ${SERVICE_NAME} -f"
