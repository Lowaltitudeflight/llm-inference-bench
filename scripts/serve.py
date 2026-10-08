#!/usr/bin/env python
"""启动 API 服务。

用法::

    python scripts/serve.py
    python scripts/serve.py --port 9000 --reload
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import API_HOST, API_PORT, LOG_LEVEL  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="启动 llm-inference-bench API 服务")
    parser.add_argument("--host", default=API_HOST, help="监听地址")
    parser.add_argument("--port", type=int, default=API_PORT, help="监听端口")
    parser.add_argument("--reload", action="store_true", help="代码变更自动重载（开发用）")
    parser.add_argument(
        "--log-level", default=LOG_LEVEL.lower(), help="日志级别，默认读 LOG_LEVEL 环境变量"
    )
    args = parser.parse_args()

    import uvicorn

    from api.server import app

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    print(f"\n  llm-inference-bench API  →  http://{args.host}:{args.port}")
    print(f"  交互式文档               →  http://{args.host}:{args.port}/docs")
    print(f"  Prometheus 指标          →  http://{args.host}:{args.port}/metrics\n")

    uvicorn.run(
        "api.server:app" if args.reload else app,
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()
