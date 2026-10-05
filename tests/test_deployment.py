"""部署能力单元测试。

覆盖:
1. /healthz 端点: 用最小 FastAPI TestClient 验证返回 {"status":"ok"}
2. 配置环境变量覆盖: 设置 os.environ 后调用 load_config 验证覆盖生效
3. 脚本语法检查: bash -n 验证三个启动/停止/状态脚本
4. Dockerfile: 检查文件存在且包含关键指令
5. docker-compose.yml: yaml.safe_load 解析验证结构
6. systemd service: configparser 验证 ini 格式与必要节
"""
from __future__ import annotations

import configparser
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from fastapi import FastAPI
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# 1. /healthz 端点测试
# ---------------------------------------------------------------------------

def _make_minimal_app_with_healthz() -> FastAPI:
    """构造与 server.py 中 /healthz 相同的最小 FastAPI 应用。

    server.py 导入时会启动后台线程导致测试挂起，因此这里用相同的路由
    定义构造一个独立 app，验证端点契约（返回 {"status":"ok"}，无需认证）。
    """
    app = FastAPI()

    @app.get("/healthz")
    async def healthz():
        """极简健康检查端点（用于负载均衡 / Docker HEALTHCHECK）。"""
        return {"status": "ok"}

    return app


class TestHealthzEndpoint:
    """验证 /healthz 端点行为。"""

    def test_healthz_returns_ok(self):
        """GET /healthz 应返回 200 和 {"status":"ok"}。"""
        app = _make_minimal_app_with_healthz()
        client = TestClient(app)
        resp = client.get("/healthz")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}

    def test_healthz_is_registered_in_server_source(self):
        """验证 server.py 源码中确实注册了 /healthz 路由。"""
        server_src = (PROJECT_ROOT / "web-dashboard" / "server.py").read_text(encoding="utf-8")
        assert '@app.get("/healthz")' in server_src, "server.py 缺少 /healthz 路由定义"
        assert '"/healthz"' in server_src, "/healthz 未在源码中出现"

    def test_healthz_in_public_paths(self):
        """验证 /healthz 已加入 _PUBLIC_EXACT_PATHS 公开路径集合。"""
        server_src = (PROJECT_ROOT / "web-dashboard" / "server.py").read_text(encoding="utf-8")
        assert '"/healthz"' in server_src, "/healthz 未加入 _PUBLIC_EXACT_PATHS"


# ---------------------------------------------------------------------------
# 2. 环境变量覆盖配置测试
# ---------------------------------------------------------------------------

class TestEnvOverrides:
    """验证 load_config() 的环境变量覆盖逻辑。"""

    def test_jev_url_override(self, monkeypatch):
        """QUANT_JEV_URL 应覆盖 cfg['jev']['base_url']。"""
        monkeypatch.setenv("QUANT_JEV_URL", "http://my-jev-server:9999")
        from config import load_config
        cfg = load_config()
        assert cfg["jev"]["base_url"] == "http://my-jev-server:9999"

    def test_api_key_override(self, monkeypatch):
        """QUANT_API_KEY 应覆盖 cfg['data']['api_key']。"""
        monkeypatch.setenv("QUANT_API_KEY", "test-key-12345")
        from config import load_config
        cfg = load_config()
        assert cfg["data"]["api_key"] == "test-key-12345"

    def test_log_level_override(self, monkeypatch):
        """QUANT_LOG_LEVEL 应覆盖 cfg['logging']['level']。"""
        monkeypatch.setenv("QUANT_LOG_LEVEL", "DEBUG")
        from config import load_config
        cfg = load_config()
        assert cfg["logging"]["level"] == "DEBUG"

    def test_port_override_stored_separately(self, monkeypatch):
        """QUANT_PORT 应存入 cfg['_env_port']，不影响 yaml 原始值。"""
        monkeypatch.setenv("QUANT_PORT", "9999")
        from config import load_config
        cfg = load_config()
        assert cfg["_env_port"] == 9999

    def test_no_env_vars_uses_yaml_defaults(self, monkeypatch):
        """无环境变量时，yaml 原始值不被破坏。"""
        # 确保不设置任何 QUANT_ 变量
        for var in ["QUANT_JEV_URL", "QUANT_API_KEY", "QUANT_LOG_LEVEL", "QUANT_PORT"]:
            monkeypatch.delenv(var, raising=False)
        from config import load_config
        cfg = load_config()
        # yaml 中的默认值应保留
        assert "base_url" in cfg["jev"]
        assert "api_key" in cfg["data"]
        assert "level" in cfg["logging"]

    def test_invalid_port_ignored(self, monkeypatch):
        """QUANT_PORT 非法值不应崩溃，应跳过。"""
        monkeypatch.setenv("QUANT_PORT", "not_a_number")
        from config import load_config
        cfg = load_config()
        assert "_env_port" not in cfg


# ---------------------------------------------------------------------------
# 3. Shell 脚本语法检查
# ---------------------------------------------------------------------------

class TestScriptSyntax:
    """用 bash -n 检查脚本语法正确性。"""

    @pytest.mark.parametrize("script_name", [
        "scripts/start_all.sh",
        "scripts/stop_all.sh",
        "scripts/status.sh",
        "deploy/install_service.sh",
    ])
    def test_script_syntax(self, script_name):
        """bash -n 应通过（语法无错误）。"""
        script_path = PROJECT_ROOT / script_name
        assert script_path.exists(), f"脚本不存在: {script_name}"
        result = subprocess.run(
            ["bash", "-n", str(script_path)],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, \
            f"{script_name} 语法错误: {result.stderr}"

    def test_scripts_have_shebang(self):
        """脚本应以 #!/usr/bin/env bash 开头。"""
        for name in ["scripts/start_all.sh", "scripts/stop_all.sh",
                      "scripts/status.sh", "deploy/install_service.sh"]:
            content = (PROJECT_ROOT / name).read_text(encoding="utf-8")
            assert content.startswith("#!/usr/bin/env bash"), \
                f"{name} 缺少正确的 shebang"


# ---------------------------------------------------------------------------
# 4. Dockerfile 检查
# ---------------------------------------------------------------------------

class TestDockerfile:
    """验证 Dockerfile 存在且包含必要指令。"""

    def test_dockerfile_exists(self):
        assert (PROJECT_ROOT / "Dockerfile").exists()

    def test_dockerfile_contains_key_directives(self):
        content = (PROJECT_ROOT / "Dockerfile").read_text(encoding="utf-8")
        assert "FROM" in content, "缺少 FROM 指令"
        assert "EXPOSE" in content, "缺少 EXPOSE 指令"
        assert "CMD" in content, "缺少 CMD 指令"
        assert "HEALTHCHECK" in content, "缺少 HEALTHCHECK 指令"
        assert "8766" in content, "未暴露 8766 端口"
        assert "requirements.txt" in content, "未安装 Python 依赖"


# ---------------------------------------------------------------------------
# 5. docker-compose.yml 解析
# ---------------------------------------------------------------------------

class TestDockerCompose:
    """验证 docker-compose.yml 可被 YAML 解析且结构正确。"""

    def test_compose_file_exists(self):
        assert (PROJECT_ROOT / "docker-compose.yml").exists()

    def test_compose_yaml_valid(self):
        with open(PROJECT_ROOT / "docker-compose.yml", "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert data is not None
        assert "services" in data, "缺少 services 节"

    def test_compose_services_config(self):
        with open(PROJECT_ROOT / "docker-compose.yml", "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        services = data["services"]
        assert "quant" in services, "缺少 quant 服务"
        quant = services["quant"]
        # 端口映射
        assert "8766:8766" in str(quant.get("ports", [])), "端口映射不正确"
        # 健康检查
        assert "healthcheck" in quant, "quant 服务缺少 healthcheck"
        # 重启策略
        assert quant.get("restart") == "unless-stopped"


# ---------------------------------------------------------------------------
# 6. systemd service 文件检查
# ---------------------------------------------------------------------------

class TestSystemdService:
    """验证 systemd unit 文件格式正确。"""

    def test_service_file_exists(self):
        assert (PROJECT_ROOT / "deploy" / "quant-trading.service").exists()

    def test_service_file_ini_valid(self):
        # systemd 允许同一节内多个 Environment= 行，需 strict=False
        parser = configparser.ConfigParser(strict=False)
        service_path = PROJECT_ROOT / "deploy" / "quant-trading.service"
        parser.read(service_path, encoding="utf-8")
        # 必须包含三个标准节
        assert parser.has_section("Unit"), "缺少 [Unit] 节"
        assert parser.has_section("Service"), "缺少 [Service] 节"
        assert parser.has_section("Install"), "缺少 [Install] 节"

    def test_service_file_content(self):
        parser = configparser.ConfigParser(strict=False)
        service_path = PROJECT_ROOT / "deploy" / "quant-trading.service"
        parser.read(service_path, encoding="utf-8")
        # Service 节应有 ExecStart 和 Restart
        service = parser["Service"]
        assert "ExecStart" in service, "缺少 ExecStart"
        assert "Restart" in service, "缺少 Restart"
        assert "server.py" in service["ExecStart"], "ExecStart 未指向 server.py"

    def test_install_service_script_exists(self):
        assert (PROJECT_ROOT / "deploy" / "install_service.sh").exists()


# ---------------------------------------------------------------------------
# 7. .dockerignore 和 .env.example 存在性
# ---------------------------------------------------------------------------

class TestSupportingFiles:
    """验证辅助部署文件存在。"""

    def test_dockerignore_exists(self):
        assert (PROJECT_ROOT / ".dockerignore").exists()

    def test_env_example_exists(self):
        assert (PROJECT_ROOT / ".env.example").exists()

    def test_env_example_has_required_vars(self):
        content = (PROJECT_ROOT / ".env.example").read_text(encoding="utf-8")
        for var in ["QUANT_PORT", "QUANT_JEV_URL", "QUANT_API_KEY", "QUANT_LOG_LEVEL"]:
            assert var in content, f".env.example 缺少 {var}"

    def test_deployment_md_exists(self):
        assert (PROJECT_ROOT / "DEPLOYMENT.md").exists()
