#!/usr/bin/env python
"""本地假推理服务，用于在没有 GPU / vLLM 的机器上验证完整链路。

模拟一个 OpenAI 兼容的流式推理服务，带**真实的时间特性**：

- 首 token 前有固定延迟（prefill 成本），随 prompt 长度线性增长
- 每生成一个 token 有固定延迟（decode 成本）
- 并发升高时，所有请求的 TTFT 都会因排队而变长（模拟 vLLM 的调度）

最后一点很关键：只有模拟了排队，压测才能跑出「并发上升 → 吞吐饱和、
延迟陡增」的真实曲线，报告里的结论段才有东西可写。

用法::

    python scripts/mock_engine.py --port 8100 --name vllm

然后另开终端::

    python scripts/run_bench.py --engine vllm \\
        --base-url http://localhost:8100 --model mock-72b \\
        --levels 1,2,4,8,16 --report
"""

from __future__ import annotations

import argparse
import json
import math
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

# 当前在处理的请求数。用于让 TTFT 随负载上升——
# 没有这个，并发压测的延迟曲线会是一条直线，看不出瓶颈。
_inflight = 0
_inflight_lock = threading.Lock()


class MockEngineHandler(BaseHTTPRequestHandler):
    """OpenAI 兼容的最小实现。

    支持：
    - GET  /v1/models
    - POST /v1/chat/completions（流式与非流式）
    """

    # ---- 命令行注入的配置（见 serve() 里的赋值）----
    model_name = "mock-model"
    base_ttft = 0.08
    per_token = 0.02
    prefill_per_char = 0.0004
    queue_factor = 0.04
    tokens_default = 64

    protocol_version = "HTTP/1.1"

    # ------------------------------------------------------------------
    def log_message(self, fmt: str, *args: Any) -> None:
        """默认实现会把每个请求打到 stderr，压测时刷屏，静音。"""
        return

    def _model_matches(self, requested: str) -> bool:
        """接受模型名带量化后缀，如 'mock-72b-awq'。

        真实 vLLM 部署时模型名常带 -AWQ / -GPTQ 后缀，
        这里复现同样的宽松匹配，方便演示多量化配置对比。
        """
        if requested == self.model_name:
            return True
        return requested.startswith(self.model_name)

    # ------------------------------------------------------------------
    def _send_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    # ------------------------------------------------------------------
    def do_GET(self) -> None:  # noqa: N802
        if self.path.rstrip("/") == "/v1/models":
            self._send_json(
                {
                    "object": "list",
                    "data": [
                        {"id": self.model_name, "object": "model", "owned_by": "mock"},
                        # 放一个别的模型，验证 health() 的模型匹配逻辑
                        {"id": "another-model", "object": "model", "owned_by": "mock"},
                    ],
                }
            )
            return
        if self.path.rstrip("/") in ("/health", "/"):
            self._send_json({"status": "ok", "model": self.model_name})
            return
        self._send_json({"error": {"message": "not found"}}, status=404)

    # ------------------------------------------------------------------
    def do_POST(self) -> None:  # noqa: N802
        global _inflight

        if self.path.rstrip("/") != "/v1/chat/completions":
            self._send_json({"error": {"message": "not found"}}, status=404)
            return

        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            req = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            self._send_json({"error": {"message": "invalid json"}}, status=400)
            return

        model = req.get("model", "")
        if model and not self._model_matches(model):
            # 让模型名写错的场景能被真实复现
            self._send_json(
                {"error": {"message": f"model '{model}' not found"}}, status=404
            )
            return

        max_tokens = int(req.get("max_tokens") or self.tokens_default)
        prompt_chars = sum(
            len(m.get("content", "")) for m in req.get("messages", [])
        )
        stream = bool(req.get("stream"))

        with _inflight_lock:
            _inflight += 1
            concurrent = _inflight

        # TTFT = 基础 + prompt 预填充 + 排队惩罚
        ttft = (
            self.base_ttft
            + prompt_chars * self.prefill_per_char
            + self.queue_factor * math.log2(max(concurrent, 1))
        )

        try:
            if stream:
                self._stream_response(ttft, max_tokens, prompt_chars, concurrent)
            else:
                self._full_response(ttft, max_tokens, prompt_chars, concurrent)
        finally:
            with _inflight_lock:
                _inflight -= 1

    # ------------------------------------------------------------------
    def _generate(self, n: int) -> str:
        """产出 n 个"词"。内容不重要，长度才重要。"""
        words = ["张量", "并行", "是", "把", "一个", "大", "矩阵", "切", "成", "多块",
                 "分别", "放到", "不同", "GPU", "上", "计算", "再", "汇总", "结果", "。"]
        return " ".join(words[i % len(words)] for i in range(n))

    def _stream_response(
        self, ttft: float, max_tokens: int, prompt_chars: int, concurrent: int
    ) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        # 流式响应没有 Content-Length，用 chunked 更标准；
        # 但为简化，这里直接不设长度并靠连接关闭结束。
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

        # 首 token 之前的等待
        time.sleep(ttft)

        def chunk(payload: dict) -> None:
            data = f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode("utf-8")
            self.wfile.write(f"{len(data):X}\r\n".encode())
            self.wfile.write(data)
            self.wfile.write(b"\r\n")

        # 分 8 次推送，每次若干 token，更接近真实流式
        sent = 0
        per_chunk = max(1, max_tokens // 8)
        while sent < max_tokens:
            n = min(per_chunk, max_tokens - sent)
            if sent > 0:
                time.sleep(self.per_token * n)
            chunk({"choices": [{"index": 0, "delta": {"content": self._generate(n)}}]})
            sent += n

        # usage 帧（stream_options.include_usage）
        chunk(
            {
                "choices": [],
                "usage": {
                    "prompt_tokens": max(1, prompt_chars // 3),
                    "completion_tokens": max_tokens,
                    "total_tokens": max(1, prompt_chars // 3) + max_tokens,
                },
            }
        )
        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()

    def _full_response(
        self, ttft: float, max_tokens: int, prompt_chars: int, concurrent: int
    ) -> None:
        time.sleep(ttft + self.per_token * max_tokens)
        self._send_json(
            {
                "id": "chatcmpl-mock",
                "object": "chat.completion",
                "model": self.model_name,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": self._generate(max_tokens)},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": max(1, prompt_chars // 3),
                    "completion_tokens": max_tokens,
                    "total_tokens": max(1, prompt_chars // 3) + max_tokens,
                },
            }
        )


def serve() -> None:
    parser = argparse.ArgumentParser(description="OpenAI 兼容的假推理服务")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument("--model", default="mock-72b", help="暴露的模型名")
    parser.add_argument("--base-ttft", type=float, default=0.08,
                        help="基础首包延迟（秒）")
    parser.add_argument("--per-token", type=float, default=0.02,
                        help="每 token 生成延迟（秒）")
    parser.add_argument("--prefill-per-char", type=float, default=0.0004,
                        help="每字符 prefill 成本（秒）")
    parser.add_argument("--queue-factor", type=float, default=0.04,
                        help="排队惩罚系数，越大则并发升高时 TTFT 涨得越快")
    parser.add_argument("--tokens-default", type=int, default=64,
                        help="请求未指定 max_tokens 时的默认生成数")
    args = parser.parse_args()

    MockEngineHandler.model_name = args.model
    MockEngineHandler.base_ttft = args.base_ttft
    MockEngineHandler.per_token = args.per_token
    MockEngineHandler.prefill_per_char = args.prefill_per_char
    MockEngineHandler.queue_factor = args.queue_factor
    MockEngineHandler.tokens_default = args.tokens_default

    server = ThreadingHTTPServer((args.host, args.port), MockEngineHandler)
    print(f"假推理服务已启动：http://{args.host}:{args.port}")
    print(f"  模型名        : {args.model}")
    print(f"  基础 TTFT     : {args.base_ttft}s")
    print(f"  每 token      : {args.per_token}s")
    print(f"  排队惩罚系数  : {args.queue_factor}")
    print(f"\n  接入命令：")
    print(f"  python scripts/run_bench.py --engine vllm \\")
    print(f"      --base-url http://{args.host}:{args.port} --model {args.model} --report")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止。")
    finally:
        server.server_close()


if __name__ == "__main__":
    serve()
