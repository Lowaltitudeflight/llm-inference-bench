"""推理引擎接入层。

vLLM 与 Ollama 都提供 OpenAI 兼容接口，因此统一由
:class:`engines.client.UnifiedLLMClient` 接入。
"""

from engines.client import EngineError, UnifiedLLMClient

__all__ = ["EngineError", "UnifiedLLMClient"]
