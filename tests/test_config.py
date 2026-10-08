"""配置层测试。

配置是所有模块的共同依赖，出错的影响面最大。重点验证：
1. 默认值合理（并发梯度、超时、生成长度）
2. 环境变量能覆盖（Docker 部署全靠这个）
3. 非法环境变量不会让程序崩，而是回退到默认值
"""

from __future__ import annotations

import importlib
import os

import pytest

import config.settings as settings
from config.settings import (
    EngineTarget,
    _env,
    _env_float,
    _env_int,
    default_engines,
)


class TestEngineTarget:
    def test_normalized_base_url_strips_slash(self):
        t = EngineTarget(name="x", base_url="http://h:8000/", model="m")
        assert t.normalized_base_url() == "http://h:8000"

    def test_normalized_base_url_no_slash(self):
        t = EngineTarget(name="x", base_url="http://h:8000", model="m")
        assert t.normalized_base_url() == "http://h:8000"

    def test_normalized_base_url_with_path(self):
        t = EngineTarget(name="x", base_url="http://h/v1/", model="m")
        assert t.normalized_base_url() == "http://h/v1"

    def test_defaults(self):
        t = EngineTarget(name="x", base_url="http://h", model="m")
        assert t.api_key == "EMPTY"
        assert t.quantization == "unknown"
        assert t.note == ""


class TestEnvHelpers:
    def test_env_returns_default(self, monkeypatch):
        monkeypatch.delenv("SOME_MISSING_KEY", raising=False)
        assert _env("SOME_MISSING_KEY", "fallback") == "fallback"

    def test_env_returns_value(self, monkeypatch):
        monkeypatch.setenv("SOME_KEY", "value")
        assert _env("SOME_KEY", "fallback") == "value"

    def test_env_int_parses(self, monkeypatch):
        monkeypatch.setenv("K_INT", "42")
        assert _env_int("K_INT", 1) == 42

    def test_env_int_falls_back_on_garbage(self, monkeypatch):
        """环境变量写成 'abc' 时不能崩，要回退到默认值。"""
        monkeypatch.setenv("K_INT", "abc")
        assert _env_int("K_INT", 7) == 7

    def test_env_int_falls_back_on_empty(self, monkeypatch):
        monkeypatch.setenv("K_INT", "")
        assert _env_int("K_INT", 7) == 7

    def test_env_float_parses(self, monkeypatch):
        monkeypatch.setenv("K_FLOAT", "2.5")
        assert _env_float("K_FLOAT", 1.0) == pytest.approx(2.5)

    def test_env_float_falls_back_on_garbage(self, monkeypatch):
        monkeypatch.setenv("K_FLOAT", "not-a-number")
        assert _env_float("K_FLOAT", 3.5) == pytest.approx(3.5)

    def test_env_int_negative_accepted(self, monkeypatch):
        """负数也要能解析——校验交给上层，helper 只负责类型转换。"""
        monkeypatch.setenv("K_INT", "-5")
        assert _env_int("K_INT", 0) == -5


class TestDefaultEngines:
    def test_both_engines_present(self):
        engines = default_engines()
        assert set(engines) == {"vllm", "ollama"}

    def test_vllm_defaults(self):
        v = default_engines()["vllm"]
        assert v.base_url.startswith("http://localhost:8000")
        assert "Qwen" in v.model

    def test_ollama_defaults(self):
        o = default_engines()["ollama"]
        assert "11434" in o.base_url
        assert o.model

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("VLLM_BASE_URL", "http://gpu-node:9000")
        monkeypatch.setenv("VLLM_MODEL", "custom-model")
        monkeypatch.setenv("VLLM_QUANT", "awq")
        v = default_engines()["vllm"]
        assert v.base_url == "http://gpu-node:9000"
        assert v.model == "custom-model"
        assert v.quantization == "awq"

    def test_ollama_env_override(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_BASE_URL", "http://other:1234")
        monkeypatch.setenv("OLLAMA_QUANT", "q8_0")
        o = default_engines()["ollama"]
        assert o.base_url == "http://other:1234"
        assert o.quantization == "q8_0"

    def test_fresh_dict_each_call(self):
        """必须每次返回新对象，否则一个引擎的覆盖会污染其他调用方。"""
        a = default_engines()
        b = default_engines()
        assert a is not b
        assert a["vllm"] is not b["vllm"]


class TestConfigDefaults:
    def test_bench_defaults_sane(self):
        """默认并发梯度应覆盖单路到高并发，便于画出曲线。"""
        assert settings.BENCH.concurrency_levels == [1, 2, 4, 8, 16, 32]
        assert settings.BENCH.max_tokens > 0
        assert settings.BENCH.timeout > 0
        assert settings.BENCH.temperature == 0.0
        assert settings.BENCH.stream is True

    def test_default_async_safe(self):
        """压测默认必须流式，否则 TTFT/TPOT 测不到。"""
        assert settings.BENCH.stream is True

    def test_warmup_enabled_by_default(self):
        assert settings.BENCH.warmup_requests > 0

    def test_max_failure_rate_threshold(self):
        assert 0 < settings.BENCH.max_failure_rate < 1

    def test_monitor_defaults(self):
        assert settings.MONITOR.interval > 0

    def test_api_defaults(self):
        assert settings.API_PORT > 0
        assert settings.API_HOST
        assert settings.LOG_LEVEL


class TestConcurrencyLevelsParsing:
    def test_valid_list(self, monkeypatch):
        monkeypatch.setenv("BENCH_CONCURRENCY_LEVELS", "1,4,16")
        mod = importlib.reload(settings)
        try:
            assert mod.BENCH.concurrency_levels == [1, 4, 16]
        finally:
            monkeypatch.delenv("BENCH_CONCURRENCY_LEVELS")
            importlib.reload(settings)

    def test_single_level(self, monkeypatch):
        monkeypatch.setenv("BENCH_CONCURRENCY_LEVELS", "8")
        mod = importlib.reload(settings)
        try:
            assert mod.BENCH.concurrency_levels == [8]
        finally:
            monkeypatch.delenv("BENCH_CONCURRENCY_LEVELS")
            importlib.reload(settings)

    def test_invalid_raises_at_import(self, monkeypatch):
        """非法并发梯度应在启动时就报错，而不是压到一半才炸。

        这里明确固定当前行为：int() 转换失败会抛 ValueError。
        这是有意的——配置错误要早暴露，而不是静默用错误的值跑完整轮压测。
        """
        monkeypatch.setenv("BENCH_CONCURRENCY_LEVELS", "1,abc")
        try:
            with pytest.raises(ValueError):
                importlib.reload(settings)
        finally:
            monkeypatch.delenv("BENCH_CONCURRENCY_LEVELS")
            importlib.reload(settings)


class TestEnvOverrides:
    def test_bench_env_overrides(self, monkeypatch):
        monkeypatch.setenv("BENCH_MAX_TOKENS", "512")
        monkeypatch.setenv("BENCH_TIMEOUT", "60")
        monkeypatch.setenv("BENCH_WARMUP", "5")
        monkeypatch.setenv("BENCH_STREAM", "false")
        monkeypatch.setenv("BENCH_TEMPERATURE", "0.7")
        mod = importlib.reload(settings)
        try:
            assert mod.BENCH.max_tokens == 512
            assert mod.BENCH.timeout == pytest.approx(60.0)
            assert mod.BENCH.warmup_requests == 5
            assert mod.BENCH.stream is False
            assert mod.BENCH.temperature == pytest.approx(0.7)
        finally:
            for k in ("BENCH_MAX_TOKENS", "BENCH_TIMEOUT", "BENCH_WARMUP",
                      "BENCH_STREAM", "BENCH_TEMPERATURE"):
                monkeypatch.delenv(k, raising=False)
            importlib.reload(settings)

    def test_stream_flag_parsing(self, monkeypatch):
        monkeypatch.setenv("BENCH_STREAM", "TRUE")
        mod = importlib.reload(settings)
        try:
            assert mod.BENCH.stream is True
        finally:
            monkeypatch.delenv("BENCH_STREAM", raising=False)
            importlib.reload(settings)

    def test_monitor_env_overrides(self, monkeypatch):
        monkeypatch.setenv("MONITOR_INTERVAL", "0.1")
        monkeypatch.setenv("MONITOR_ENABLED", "false")
        mod = importlib.reload(settings)
        try:
            assert mod.MONITOR.interval == pytest.approx(0.1)
            assert mod.MONITOR.enabled is False
        finally:
            for k in ("MONITOR_INTERVAL", "MONITOR_ENABLED"):
                monkeypatch.delenv(k, raising=False)
            importlib.reload(settings)

    def test_api_env_overrides(self, monkeypatch):
        monkeypatch.setenv("API_PORT", "9999")
        monkeypatch.setenv("LOG_LEVEL", "debug")
        mod = importlib.reload(settings)
        try:
            assert mod.API_PORT == 9999
            assert mod.LOG_LEVEL == "DEBUG"  # 统一大写
        finally:
            for k in ("API_PORT", "LOG_LEVEL"):
                monkeypatch.delenv(k, raising=False)
            importlib.reload(settings)


class TestBenchConfigIndependence:
    def test_mutable_defaults_not_shared(self):
        """BenchConfig 的 list 默认值必须每次新建，否则跨实例污染。"""
        from config.settings import BenchConfig

        a = BenchConfig()
        b = BenchConfig()
        a.concurrency_levels.append(64)
        assert 64 not in b.concurrency_levels

    def test_output_dir_is_project_relative(self):
        """默认输出目录要在项目内，不能落到 CWD 或系统临时目录。"""
        from config.settings import BASE_DIR, BenchConfig

        d = BenchConfig().output_dir
        assert str(d).startswith(str(BASE_DIR))
