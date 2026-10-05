"""国际化 i18n 模块测试（REQ-P3-07）。

覆盖：
  - 中/英语言包 key 对齐
  - normalize_lang / Accept-Language 解析（含 q 权重）
  - t() 文案取值与 en 回退
  - 数字 / 货币 / 百分比 / 日期地区格式化
"""
from __future__ import annotations

import sys
from datetime import date, datetime
from pathlib import Path

import pytest

# i18n.py 位于 web-dashboard/，需先加入 path（与路由测试同引导）
WEB_DASHBOARD = Path(__file__).resolve().parent.parent / "web-dashboard"
if str(WEB_DASHBOARD) not in sys.path:
    sys.path.insert(0, str(WEB_DASHBOARD))

from i18n import (
    DEFAULT_LANG,
    LANG_PACKS,
    SUPPORTED_LANGS,
    format_currency,
    format_date,
    format_number,
    format_percent,
    normalize_accept_language,
    normalize_lang,
    t,
)


# ---------------------------------------------------------------------------
# 语言包完整性
# ---------------------------------------------------------------------------


def test_supported_langs():
    assert SUPPORTED_LANGS == ["zh", "en"]
    assert DEFAULT_LANG == "zh"


def test_pack_key_parity():
    zh_keys = set(LANG_PACKS["zh"].keys())
    en_keys = set(LANG_PACKS["en"].keys())
    assert zh_keys == en_keys, f"语言包 key 不对齐: {zh_keys ^ en_keys}"
    assert len(zh_keys) >= 10


def test_no_empty_values():
    for lang, pack in LANG_PACKS.items():
        for k, v in pack.items():
            assert isinstance(v, str) and v.strip(), f"{lang}.{k} 为空"




# ---------------------------------------------------------------------------
# normalize_lang / Accept-Language
# ---------------------------------------------------------------------------


class TestNormalizeLang:
    @pytest.mark.parametrize("raw,expected", [
        ("zh", "zh"), ("en", "en"), ("ZH", "zh"), ("EN", "en"),
        ("zh-CN", "zh"), ("zh-TW", "zh"), ("en-US", "en"), ("en-GB", "en"),
        ("fr", "zh"), ("", "zh"), (None, "zh"), ("  en  ", "en"),
    ])
    def test_mapping(self, raw, expected):
        assert normalize_lang(raw) == expected


class TestAcceptLanguage:
    def test_single(self):
        assert normalize_accept_language("en-US") == "en"

    def test_q_weights(self):
        assert normalize_accept_language("fr-FR;q=0.9,en;q=0.8,zh-CN;q=0.5") == "en"

    def test_zh_preferred(self):
        assert normalize_accept_language("zh-CN,zh;q=0.9,en;q=0.8") == "zh"

    def test_wildcard_falls_back_to_default(self):
        assert normalize_accept_language("*") == DEFAULT_LANG

    def test_none(self):
        assert normalize_accept_language(None) == DEFAULT_LANG

    def test_unsupported_only(self):
        assert normalize_accept_language("fr-FR,fr;q=0.9") == DEFAULT_LANG


# ---------------------------------------------------------------------------
# t() 文案
# ---------------------------------------------------------------------------


class TestT:
    def test_zh(self):
        assert t("zh", "api.success") == LANG_PACKS["zh"]["api.success"]

    def test_en(self):
        assert t("en", "api.success") == LANG_PACKS["en"]["api.success"]

    def test_missing_key_falls_back_to_en(self):
        # en 包没有该 key 时最终返回 key 本身
        assert t("en", "no.such.key") == "no.such.key"
        assert t("zh", "no.such.key") == "no.such.key"

    def test_none_lang_uses_default(self):
        assert t(None, "api.success") == LANG_PACKS[DEFAULT_LANG]["api.success"]

    def test_placeholder_formatting(self):
        text = t("zh", "api.symbol_not_found", symbol="600519.SH")
        assert "600519.SH" in text
        assert "{symbol}" not in text

    def test_placeholder_en(self):
        text = t("en", "api.symbol_not_found", symbol="600519.SH")
        assert "600519.SH" in text
        assert "{symbol}" not in text


# ---------------------------------------------------------------------------
# 地区格式化
# ---------------------------------------------------------------------------


class TestFormatting:
    def test_number_thousands(self):
        assert format_number(1234567.891, "en") == "1,234,567.89"
        assert format_number(1234567.891, "zh") == "1,234,567.89"

    def test_number_precision(self):
        assert format_number(3.14159, "en", precision=4) == "3.1416"

    def test_currency_symbols(self):
        assert format_currency(1234.5, "en").startswith("$")
        assert format_currency(1234.5, "zh").startswith("¥")
        assert "1,234.50" in format_currency(1234.5, "en")

    def test_percent(self):
        assert format_percent(0.1234, "en") == "12.34%"
        assert format_percent(0.05, "zh") == "5.00%"

    def test_date_formats(self):
        d = date(2026, 10, 5)
        assert format_date(d, "zh") == "2026-10-05"
        assert format_date(d, "en") == "Oct 05, 2026"

    def test_date_from_iso_string(self):
        assert format_date("2026-10-05", "zh") == "2026-10-05"
        assert format_date("2026-10-05", "en") == "Oct 05, 2026"

    def test_date_from_datetime(self):
        assert format_date(datetime(2026, 3, 14, 8, 0), "en") == "Mar 14, 2026"

    def test_default_lang_is_zh(self):
        assert format_currency(1.0).startswith("¥")
