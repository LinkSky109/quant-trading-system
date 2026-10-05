"""配置热加载模块单元测试与 API 端点集成测试。

覆盖：
  - HotReloadManager 初始化 / 可热加载节列表
  - reload_config 成功更新内存中的可热加载节
  - 不可热加载节（data.api_key / jev.base_url）被跳过、保持旧值
  - diff_config 正确识别叶子级变更
  - 回调注册与触发
  - get_current_config 脱敏（api_key/token/password -> "***"）
  - SIGHUP 处理函数触发 reload
  - 校验失败时不更新内存配置（ConfigReloadError）
  - API 端点集成测试（最小 FastAPI App + TestClient）
  - 线程安全：并发 reload 不崩溃
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Dict

import pytest
import yaml
from fastapi import FastAPI
from fastapi.testclient import TestClient

from config.hot_reload import (
    HOT_RELOADABLE_SECTIONS,
    NON_HOT_RELOADABLE_SECTIONS,
    ConfigReloadError,
    HotReloadManager,
)


# ---------------------------------------------------------------------------
# 构造一份能通过 validator 的最小合法配置
# ---------------------------------------------------------------------------

def _base_config() -> Dict[str, Any]:
    """最小可通过 validate_config 的配置（provider=mock 免 api_key 校验）。"""
    return {
        "data": {"provider": "mock", "api_key": "sk_oldkey1234567890abcdef1234567890ab",
                 "cache_dir": "./cache", "cache_ttl_hours": 4},
        "jev": {"base_url": "http://localhost:8765", "timeout": 5.0,
                "confidence_threshold": 0.6, "retry_count": 2, "mock_mode": True},
        "risk": {"single_stop_loss": 0.03, "single_take_profit": 0.08,
                 "max_drawdown_pause": 0.10, "max_position_per_symbol": 0.20,
                 "max_total_position": 0.80, "daily_loss_limit": 0.02},
        "strategies": {"ma_cross": {"label": "双均线", "fast_period": 5,
                                    "slow_period": 20, "enabled": True}},
        "accounts": [{"account_id": "acc_1", "name": "稳健", "initial_capital": 1000000.0,
                      "strategy": "ma_cross", "account_type": "simulated",
                      "seed_positions": [["600519.SH", 0.2]]}],
        "alert": {"webhook_url": "", "cooldown_seconds": 300, "max_history": 500},
        "rate_limit": {"enabled": False, "tiers": {"read": 100}},
        "audit": {"enabled": True, "log_file": "./logs/audit.jsonl"},
        "notification": {"enabled": False},
    }


def _write_config(path: Path, cfg: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True)


@pytest.fixture()
def cfg_file(tmp_path: Path) -> Path:
    p = tmp_path / "config.yaml"
    _write_config(p, _base_config())
    return p


@pytest.fixture()
def manager(cfg_file: Path) -> HotReloadManager:
    cfg = _base_config()
    return HotReloadManager(cfg, config_path=str(cfg_file), register_signal=False)


# ---------------------------------------------------------------------------
# 初始化 / 列表
# ---------------------------------------------------------------------------

class TestInit:
    def test_lists(self):
        assert set(HotReloadManager.get_hot_reloadable_keys()) == set(HOT_RELOADABLE_SECTIONS)
        assert "strategies" in HotReloadManager.get_hot_reloadable_keys()
        assert "risk" in HotReloadManager.get_hot_reloadable_keys()
        assert set(HotReloadManager.get_non_hot_reloadable_keys()) == set(NON_HOT_RELOADABLE_SECTIONS)
        assert "data" in HotReloadManager.get_non_hot_reloadable_keys()

    def test_holds_reference_to_passed_config(self, manager: HotReloadManager):
        # manager 应原地更新传入的 dict（持有同一引用）
        assert manager.get_current_config(sanitize=False)["risk"]["single_stop_loss"] == 0.03


# ---------------------------------------------------------------------------
# reload_config
# ---------------------------------------------------------------------------

class TestReload:
    def test_reload_updates_hot_section(self, manager: HotReloadManager, cfg_file: Path):
        # 修改磁盘上的 risk 节
        cfg = _base_config()
        cfg["risk"]["single_stop_loss"] = 0.05
        cfg["alert"]["cooldown_seconds"] = 999
        _write_config(cfg_file, cfg)

        result = manager.reload_config()
        assert result["reloaded"] is True
        assert result["timestamp"]  # 非空时间戳
        # 内存配置已更新
        assert manager.get_current_config(sanitize=False)["risk"]["single_stop_loss"] == 0.05
        assert manager.get_current_config(sanitize=False)["alert"]["cooldown_seconds"] == 999
        # 变更列表里能找到
        changed_keys = [c["key"] for c in result["changes"]]
        assert "risk.single_stop_loss" in changed_keys
        assert "alert.cooldown_seconds" in changed_keys

    def test_non_hot_section_not_updated(self, manager: HotReloadManager, cfg_file: Path):
        old_api_key = manager.get_current_config(sanitize=False)["data"]["api_key"]
        old_jev_url = manager.get_current_config(sanitize=False)["jev"]["base_url"]
        assert old_api_key == "sk_oldkey1234567890abcdef1234567890ab"

        cfg = _base_config()
        cfg["data"]["api_key"] = "sk_brandnewkey1234567890abcdef1234567890ab"
        cfg["jev"]["base_url"] = "http://other-host:9999"
        _write_config(cfg_file, cfg)

        manager.reload_config()
        cur = manager.get_current_config(sanitize=False)
        # 不可热加载节保持旧值
        assert cur["data"]["api_key"] == old_api_key
        assert cur["jev"]["base_url"] == old_jev_url

    def test_validation_error_keeps_old_config(self, manager: HotReloadManager, cfg_file: Path):
        # 写入非法配置：risk.single_stop_loss 超过上限 0.2 -> 校验错误
        cfg = _base_config()
        cfg["risk"]["single_stop_loss"] = 0.99  # > 0.2 非法
        _write_config(cfg_file, cfg)

        with pytest.raises(ConfigReloadError):
            manager.reload_config()

        # 内存配置保持重载前旧值
        assert manager.get_current_config(sanitize=False)["risk"]["single_stop_loss"] == 0.03

    def test_reload_missing_file_raises(self, tmp_path: Path):
        cfg = _base_config()
        m = HotReloadManager(cfg, config_path=str(tmp_path / "nope.yaml"),
                             register_signal=False)
        with pytest.raises(ConfigReloadError):
            m.reload_config()


# ---------------------------------------------------------------------------
# diff_config
# ---------------------------------------------------------------------------

class TestDiff:
    def test_detects_leaf_change(self):
        old = {"risk": {"single_stop_loss": 0.03, "max_total_position": 0.8}}
        new = {"risk": {"single_stop_loss": 0.05, "max_total_position": 0.8}}
        changes = HotReloadManager.diff_config(old, new)
        assert len(changes) == 1
        c = changes[0]
        assert c["section"] == "risk"
        assert c["key"] == "risk.single_stop_loss"
        assert c["old_value"] == 0.03
        assert c["new_value"] == 0.05

    def test_no_change_returns_empty(self):
        cfg = {"risk": {"a": 1}, "alert": {"b": "x"}}
        assert HotReloadManager.diff_config(cfg, json.loads(json.dumps(cfg))) == []

    def test_detects_added_and_removed_keys(self):
        old = {"risk": {"a": 1}}
        new = {"risk": {"a": 1, "b": 2}}
        changes = HotReloadManager.diff_config(old, new)
        keys = {c["key"]: c for c in changes}
        assert "risk.b" in keys
        assert keys["risk.b"]["old_value"] is None
        assert keys["risk.b"]["new_value"] == 2

    def test_list_treated_as_leaf(self):
        old = {"notification": {"to": ["a@x.com"]}}
        new = {"notification": {"to": ["a@x.com", "b@x.com"]}}
        changes = HotReloadManager.diff_config(old, new)
        assert len(changes) == 1
        assert changes[0]["key"] == "notification.to"


# ---------------------------------------------------------------------------
# 回调
# ---------------------------------------------------------------------------

class TestCallbacks:
    def test_callback_fired_on_reload(self, manager: HotReloadManager, cfg_file: Path):
        received: list = []
        manager.register_callback(lambda changes: received.append(changes))

        cfg = _base_config()
        cfg["risk"]["single_stop_loss"] = 0.07
        _write_config(cfg_file, cfg)

        manager.reload_config()
        assert len(received) == 1
        changed_keys = [c["key"] for c in received[0]]
        assert "risk.single_stop_loss" in changed_keys

    def test_failing_callback_does_not_break_reload(self, manager: HotReloadManager, cfg_file: Path):
        manager.register_callback(lambda changes: (_ for _ in ()).throw(RuntimeError("boom")))
        ok_flag: list = []
        manager.register_callback(lambda changes: ok_flag.append(True))

        cfg = _base_config()
        cfg["risk"]["single_stop_loss"] = 0.09
        _write_config(cfg_file, cfg)
        result = manager.reload_config()  # 不应抛异常
        assert result["reloaded"] is True
        assert ok_flag == [True]


# ---------------------------------------------------------------------------
# 脱敏
# ---------------------------------------------------------------------------

class TestSanitize:
    def test_sensitive_fields_masked(self, tmp_path: Path):
        cfg = _base_config()
        cfg["data"]["api_key"] = "sk_secretkey"
        cfg["notification"] = {"channels": {"email": {"password": "smtp-pass",
                                                      "user": "alert@x.com"}}}
        cfg["alert"] = {"webhook_url": "https://hooks.example.com/TOKEN123"}
        p = tmp_path / "c.yaml"
        _write_config(p, cfg)
        m = HotReloadManager(cfg, config_path=str(p), register_signal=False)

        safe = m.get_current_config(sanitize=True)
        assert safe["data"]["api_key"] == "***"
        assert safe["notification"]["channels"]["email"]["password"] == "***"
        # 非敏感字段保留
        assert safe["notification"]["channels"]["email"]["user"] == "alert@x.com"

    def test_unsanitize_returns_real_values(self, manager: HotReloadManager):
        cur = manager.get_current_config(sanitize=False)
        assert cur["data"]["api_key"] == "sk_oldkey1234567890abcdef1234567890ab"

    def test_sanitize_does_not_mutate_underlying(self, manager: HotReloadManager):
        manager.get_current_config(sanitize=True)
        again = manager.get_current_config(sanitize=False)
        assert again["data"]["api_key"] == "sk_oldkey1234567890abcdef1234567890ab"


# ---------------------------------------------------------------------------
# SIGHUP
# ---------------------------------------------------------------------------

class TestSignal:
    def test_sighup_handler_triggers_reload(self, manager: HotReloadManager, cfg_file: Path):
        cfg = _base_config()
        cfg["alert"]["max_history"] = 777
        _write_config(cfg_file, cfg)
        # 直接调用信号处理函数（不依赖真实信号投递）
        manager._on_sighup(signal_num := 1, frame=None)
        cur = manager.get_current_config(sanitize=False)
        assert cur["alert"]["max_history"] == 777

    def test_signal_registration_can_be_disabled(self, cfg_file: Path):
        m = HotReloadManager(_base_config(), config_path=str(cfg_file),
                             register_signal=False)
        assert m._signal_handler_registered is False


# ---------------------------------------------------------------------------
# 线程安全
# ---------------------------------------------------------------------------

class TestThreadSafety:
    def test_concurrent_reload_no_crash(self, manager: HotReloadManager, cfg_file: Path):
        cfg = _base_config()
        cfg["risk"]["single_stop_loss"] = 0.04
        _write_config(cfg_file, cfg)

        errors: list[Exception] = []

        def worker():
            try:
                for _ in range(5):
                    manager.reload_config()
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        assert manager.get_current_config(sanitize=False)["risk"]["single_stop_loss"] == 0.04


# ---------------------------------------------------------------------------
# API 端点集成测试（最小 FastAPI App，镜像 server.py 的三个端点）
# ---------------------------------------------------------------------------

def build_config_app(hot_reload_manager: HotReloadManager) -> FastAPI:
    """构建最小 FastAPI 应用，端点逻辑与 web-dashboard/server.py 一致。"""
    app = FastAPI()

    def ok(data=None, message="success"):
        return {"code": 0, "message": message, "data": data}

    def err(code, message, http_status=400):
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=http_status,
                            content={"code": code, "message": message, "data": None})

    @app.post("/api/config/reload")
    async def config_reload():
        try:
            result = hot_reload_manager.reload_config()
            return ok(result, message="配置热加载完成")
        except Exception as e:
            return err(50010, f"配置热加载失败: {e}", http_status=400)

    @app.get("/api/config/current")
    async def config_current():
        cfg = hot_reload_manager.get_current_config(sanitize=True)
        return ok({"config": cfg})

    @app.get("/api/config/hot_reloadable")
    async def config_hot_reloadable():
        return ok({
            "keys": hot_reload_manager.get_hot_reloadable_keys(),
            "non_reloadable": hot_reload_manager.get_non_hot_reloadable_keys(),
        })

    return app


class TestConfigEndpoints:
    @pytest.fixture()
    def client(self, cfg_file: Path):
        mgr = HotReloadManager(_base_config(), config_path=str(cfg_file),
                               register_signal=False)
        self._cfg_file = cfg_file
        self._mgr = mgr
        return TestClient(build_config_app(mgr))

    def test_hot_reloadable_endpoint(self, client: TestClient):
        r = client.get("/api/config/hot_reloadable")
        assert r.status_code == 200
        body = r.json()
        assert body["code"] == 0
        assert "risk" in body["data"]["keys"]
        assert "data" in body["data"]["non_reloadable"]

    def test_current_endpoint_sanitized(self, client: TestClient):
        r = client.get("/api/config/current")
        assert r.status_code == 200
        cfg = r.json()["data"]["config"]
        assert cfg["data"]["api_key"] == "***"

    def test_reload_endpoint_applies_change(self, client: TestClient):
        cfg = _base_config()
        cfg["risk"]["single_stop_loss"] = 0.06
        _write_config(self._cfg_file, cfg)

        r = client.post("/api/config/reload")
        assert r.status_code == 200
        body = r.json()
        assert body["code"] == 0
        assert body["data"]["reloaded"] is True
        changed = [c["key"] for c in body["data"]["changes"]]
        assert "risk.single_stop_loss" in changed
        assert body["data"]["timestamp"]

        # 再次查询当前配置确认已生效
        cur = client.get("/api/config/current").json()["data"]["config"]
        assert cur["risk"]["single_stop_loss"] == 0.06

    def test_reload_endpoint_returns_400_on_bad_config(self, client: TestClient):
        cfg = _base_config()
        cfg["risk"]["single_stop_loss"] = 0.99  # 非法
        _write_config(self._cfg_file, cfg)
        r = client.post("/api/config/reload")
        assert r.status_code == 400
        assert r.json()["code"] == 50010
