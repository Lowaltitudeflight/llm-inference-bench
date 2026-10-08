"""并发压测执行。

:class:`bench.runner.LoadRunner` 是压测主入口：按并发梯度逐档加压，
每路请求使用独立客户端，避免客户端连接池成为瓶颈。
"""

from bench.runner import LoadRunner

__all__ = ["LoadRunner"]
