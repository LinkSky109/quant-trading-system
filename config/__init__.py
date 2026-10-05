"""配置加载模块。"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict

import yaml

logger = logging.getLogger(__name__)


def load_config(config_path: str | None = None) -> Dict[str, Any]:
    """加载 YAML 配置文件。

    Args:
        config_path: 配置文件路径，默认使用项目内 config/config.yaml。

    Returns:
        配置字典。加载后会用 ``QUANT_`` 前缀的环境变量覆盖对应配置项，
        详见 :func:`_apply_env_overrides`。
    """
    if config_path is None:
        config_path = str(Path(__file__).resolve().parent / "config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    _apply_env_overrides(cfg)
    return cfg


# 环境变量到配置路径的映射表：{env_var: (section, key)}
_ENV_OVERRIDE_MAP: Dict[str, tuple[str, str]] = {
    "QUANT_JEV_URL": ("jev", "base_url"),
    "QUANT_API_KEY": ("data", "api_key"),
    "QUANT_LOG_LEVEL": ("logging", "level"),
}


def _apply_env_overrides(cfg: Dict[str, Any]) -> None:
    """用环境变量覆盖配置项（原地修改 ``cfg``）。

    支持的环境变量：

    - ``QUANT_JEV_URL``   -> ``cfg['jev']['base_url']``
    - ``QUANT_API_KEY``   -> ``cfg['data']['api_key']``
    - ``QUANT_LOG_LEVEL`` -> ``cfg['logging']['level']``
    - ``QUANT_PORT``      -> ``cfg['_env_port']``（整型，供 server 启动时读取）

    只有当环境变量存在且非空字符串时才覆盖；目标 section 不存在时跳过。
    """
    for env_var, (section, key) in _ENV_OVERRIDE_MAP.items():
        val = os.environ.get(env_var)
        if val and isinstance(cfg.get(section), dict):
            cfg[section][key] = val
    # QUANT_PORT 特殊处理：存入顶层 _env_port，server 启动时读取
    port = os.environ.get("QUANT_PORT")
    if port:
        try:
            cfg["_env_port"] = int(port)
        except ValueError:
            logger.warning("QUANT_PORT 环境变量值 '%s' 不是合法端口号，已忽略", port)


def get_config_section(cfg: Dict[str, Any], section: str) -> Dict[str, Any]:
    """获取配置的某个子节，不存在时返回空字典。"""
    return cfg.get(section, {})


def load_stock_pool(pool_path: str | None = None) -> Dict[str, Any]:
    """加载股票池配置文件 stock_pool.yaml。

    Args:
        pool_path: 股票池 YAML 路径，默认使用项目内 config/stock_pool.yaml。

    Returns:
        字典，包含:
          - stocks: List[Dict]，每只股票的完整元数据（symbol/name/industry/base_price 等）
          - pool_meta: Dict，股票池元信息（名称/筛选条件/日期等）
          - raw: 原始 YAML 全量内容（含行业分布等附加信息）
    """
    if pool_path is None:
        pool_path = str(Path(__file__).resolve().parent / "stock_pool.yaml")
    with open(pool_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    stocks = data.get("stocks", []) or []
    pool_meta = data.get("pool_meta", {}) or {}
    return {
        "stocks": stocks,
        "pool_meta": pool_meta,
        "raw": data,
    }
