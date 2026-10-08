"""全局配置。

所有可调参数集中在此，支持环境变量覆盖（便于 Docker / 服务器部署）。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

BASE_DIR = Path(__file__).resolve().parent.parent


def _env(key: str, default: str) -> str:
    return os.getenv(key, default)


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, default))
    except (TypeError, ValueError):
        return default


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, default))
    except (TypeError, ValueError):
        return default


# --------------------------------------------------------------------------
# 引擎接入点
# --------------------------------------------------------------------------
@dataclass
class EngineTarget:
    """一个待评测的推理服务端点。

    vLLM 与 Ollama 都提供 OpenAI 兼容 /v1/chat/completions，
    因此可以用同一套客户端压测，差异只体现在 base_url 与模型名。
    """

    name: str
    base_url: str
    model: str
    api_key: str = "EMPTY"
    # 该引擎声称的量化等级，仅用于报告标注（脚本不做实际探测）
    quantization: str = "unknown"
    # 附加说明，例如机器规格、卡数
    note: str = ""

    def normalized_base_url(self) -> str:
        """去掉尾部斜杠，保证拼接 path 时不会出现双斜杠。"""
        return self.base_url.rstrip("/")


# --------------------------------------------------------------------------
# 压测参数
# --------------------------------------------------------------------------
@dataclass
class BenchConfig:
    """压测运行参数。"""

    # 并发梯度：依次按这些并发数跑，得到「并发 - 吞吐 - 延迟」曲线
    concurrency_levels: List[int] = field(default_factory=lambda: [1, 2, 4, 8, 16, 32])

    # 每个并发档位发起的请求总数（None 表示 = 并发数，即每路请求各跑一次）
    requests_per_level: Optional[int] = None

    # 每档位预热请求数，预热结果不计入统计
    warmup_requests: int = 2

    # 单请求最大生成 token 数
    max_tokens: int = 256

    # 请求超时（秒）
    timeout: float = 120.0

    # 温度：压测取 0 降低随机性，便于复现
    temperature: float = 0.0

    # 是否流式请求。TTFT / TPOT 只有流式才能测准
    stream: bool = True

    # 固定 prompt，保证各档位负载一致
    prompt: str = "请用三句话说明什么是张量并行（Tensor Parallelism）。"

    # 结果落盘目录
    output_dir: Path = BASE_DIR / "data" / "results"

    # 单档位失败请求占比超过该阈值时标记该档位为不稳定
    max_failure_rate: float = 0.2


# --------------------------------------------------------------------------
# GPU 监控
# --------------------------------------------------------------------------
@dataclass
class MonitorConfig:
    """GPU 监控参数。"""

    # 采样间隔（秒）
    interval: float = 0.5

    # 是否启用（无 GPU 环境自动跳过）
    enabled: bool = True


# --------------------------------------------------------------------------
# 默认配置实例
# --------------------------------------------------------------------------
def default_engines() -> Dict[str, EngineTarget]:
    """默认引擎目标，可通过环境变量覆盖。"""
    return {
        "vllm": EngineTarget(
            name="vllm",
            base_url=_env("VLLM_BASE_URL", "http://localhost:8000"),
            model=_env("VLLM_MODEL", "Qwen2.5-72B-Instruct"),
            api_key=_env("VLLM_API_KEY", "EMPTY"),
            quantization=_env("VLLM_QUANT", "unknown"),
        ),
        "ollama": EngineTarget(
            name="ollama",
            base_url=_env("OLLAMA_BASE_URL", "http://localhost:11434"),
            model=_env("OLLAMA_MODEL", "qwen2.5:7b"),
            api_key=_env("OLLAMA_API_KEY", "ollama"),
            quantization=_env("OLLAMA_QUANT", "q4_K_M"),
        ),
    }


BENCH = BenchConfig(
    concurrency_levels=[
        int(x) for x in _env("BENCH_CONCURRENCY_LEVELS", "1,2,4,8,16,32").split(",")
    ],
    max_tokens=_env_int("BENCH_MAX_TOKENS", 256),
    timeout=_env_float("BENCH_TIMEOUT", 120.0),
    temperature=_env_float("BENCH_TEMPERATURE", 0.0),
    stream=_env("BENCH_STREAM", "true").lower() == "true",
    warmup_requests=_env_int("BENCH_WARMUP", 2),
)

MONITOR = MonitorConfig(
    interval=_env_float("MONITOR_INTERVAL", 0.5),
    enabled=_env("MONITOR_ENABLED", "true").lower() == "true",
)

# 服务端口
API_HOST = _env("API_HOST", "0.0.0.0")
API_PORT = _env_int("API_PORT", 8080)

# 日志
LOG_LEVEL = _env("LOG_LEVEL", "INFO").upper()
