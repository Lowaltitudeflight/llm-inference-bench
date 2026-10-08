"""推理引擎统一客户端。

vLLM 与 Ollama 都暴露 OpenAI 兼容的 ``/v1/chat/completions``，
因此可以用同一套客户端接入，差异只体现在 base_url / 模型名 /
是否支持流式 usage 回传。

本模块的两个职责：
1. **单次请求计时**：精确测量 TTFT 与生成阶段耗时，产出 RequestResult。
2. **健康探测**：确认引擎可达、模型已加载，避免对着空服务压测。
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import httpx
from openai import OpenAI

from config.settings import EngineTarget
from metrics.stats import RequestResult


class EngineError(RuntimeError):
    """引擎不可用或返回异常。"""


class UnifiedLLMClient:
    """统一的推理服务客户端。

    Args:
        target: 引擎接入配置。
        timeout: 单请求超时（秒）。
    """

    def __init__(self, target: EngineTarget, timeout: float = 120.0):
        self.target = target
        self.timeout = timeout
        self._client = OpenAI(
            base_url=f"{target.normalized_base_url()}/v1",
            api_key=target.api_key or "EMPTY",
            timeout=timeout,
            # 压测需要重试可控：失败就记为失败，不要静默重试掩盖问题
            max_retries=0,
        )

    # ------------------------------------------------------------------
    # 健康探测
    # ------------------------------------------------------------------
    def health(self) -> Dict[str, Any]:
        """探测引擎可达性与模型可用性。

        Returns:
            {"ok": bool, "models": [...], "error": str|None}
        """
        url = f"{self.target.normalized_base_url()}/v1/models"
        headers = {"Authorization": f"Bearer {self.target.api_key or 'EMPTY'}"}
        try:
            with httpx.Client(timeout=min(self.timeout, 10.0)) as client:
                resp = client.get(url, headers=headers)
            if resp.status_code != 200:
                return {
                    "ok": False,
                    "models": [],
                    "error": f"HTTP {resp.status_code}: {resp.text[:200]}",
                }
            data = resp.json()
            models = [m.get("id") for m in data.get("data", []) if isinstance(m, dict)]
            return {
                "ok": True,
                "models": models,
                "model_loaded": self._model_present(models),
                "error": None,
            }
        except Exception as exc:
            return {"ok": False, "models": [], "error": f"{type(exc).__name__}: {exc}"}

    def _model_present(self, models: List[str]) -> bool:
        """判断目标模型是否在可用列表中。

        引擎的模型名可能带版本后缀（如 Qwen2.5-72B-Instruct-AWQ），
        因此做前缀/包含双向匹配，避免误报未加载。
        """
        if not models:
            return False
        want = self.target.model
        for m in models:
            if m == want or want in m or m in want:
                return True
        return False

    # ------------------------------------------------------------------
    # 请求计时
    # ------------------------------------------------------------------
    def complete(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.0,
        stream: bool = True,
    ) -> RequestResult:
        """发起一次请求并测量耗时。

        流式模式下：
        - 记录**第一个** chunk 到达的时刻作为 TTFT
        - 记录**最后一个** chunk 到达的时刻作为结束
        - 生成阶段耗时 = 结束 - TTFT

        非流式模式下拿不到首 token 时刻，TTFT 留空，
        只有端到端延迟可用。

        Args:
            prompt: 输入提示词。
            max_tokens: 最大生成 token 数。
            temperature: 采样温度，压测建议为 0。
            stream: 是否流式请求。

        Returns:
            RequestResult，失败的请求 success=False 且带 error。
        """
        messages = [{"role": "user", "content": prompt}]
        start = time.perf_counter()

        try:
            if stream:
                return self._complete_stream(messages, max_tokens, temperature, start)
            return self._complete_once(messages, max_tokens, temperature, start)
        except Exception as exc:
            elapsed = time.perf_counter() - start
            return RequestResult(
                success=False,
                total_latency=elapsed,
                error=f"{type(exc).__name__}: {exc}",
            )

    def _complete_stream(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float,
        start: float,
    ) -> RequestResult:
        """流式请求，精确测量 TTFT。"""
        stream = self._client.chat.completions.create(
            model=self.target.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
            stream_options={"include_usage": True},
        )

        first_token_at: Optional[float] = None
        token_count = 0
        usage_tokens: Optional[int] = None
        input_tokens = 0

        for chunk in stream:
            now = time.perf_counter()

            # usage 是独立的一帧，可能不带 choices
            if getattr(chunk, "usage", None):
                usage_tokens = chunk.usage.completion_tokens
                input_tokens = chunk.usage.prompt_tokens or 0

            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta
            content = getattr(delta, "content", None)
            if content:
                if first_token_at is None:
                    first_token_at = now
                token_count += 1

        end = time.perf_counter()

        # 优先用服务端回传的 usage（更准），拿不到则用累计 chunk 数
        output_tokens = usage_tokens if usage_tokens is not None else token_count

        ttft = (first_token_at - start) if first_token_at is not None else None
        generation_time = (end - first_token_at) if first_token_at is not None else None

        return RequestResult(
            success=True,
            total_latency=end - start,
            ttft=ttft,
            generation_time=generation_time,
            output_tokens=output_tokens or 0,
            input_tokens=input_tokens,
        )

    def _complete_once(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float,
        start: float,
    ) -> RequestResult:
        """非流式请求，只有端到端延迟。"""
        resp = self._client.chat.completions.create(
            model=self.target.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=False,
        )
        end = time.perf_counter()

        output_tokens = 0
        input_tokens = 0
        if getattr(resp, "usage", None):
            output_tokens = resp.usage.completion_tokens or 0
            input_tokens = resp.usage.prompt_tokens or 0

        return RequestResult(
            success=True,
            total_latency=end - start,
            ttft=None,
            generation_time=None,
            output_tokens=output_tokens,
            input_tokens=input_tokens,
        )

    def close(self) -> None:
        """释放底层连接。"""
        try:
            self._client.close()
        except Exception:
            pass
