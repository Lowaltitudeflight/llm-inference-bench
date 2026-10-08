"""LLM 推理性能基准压测工具包。

统一接入 vLLM / Ollama 等 OpenAI 兼容推理服务，按并发梯度加压，
采集吞吐、TTFT、TPOT、尾延迟与显存曲线，并自动生成对比报告。

主要子模块：

- :mod:`config.settings`  — 集中配置（环境变量可覆盖）
- :mod:`engines.client`   — 统一推理客户端与计时
- :mod:`bench.runner`     — 并发压测执行器
- :mod:`metrics.stats`    — 指标计算（TTFT / TPOT / 吞吐 / 分位数）
- :mod:`metrics.store`    — 结果持久化（JSON / CSV）
- :mod:`monitor.gpu`      — GPU 显存采样
- :mod:`monitor.quant`    — 量化等级横向对比
- :mod:`report`           — Markdown / HTML 报告生成
"""

__version__ = "0.1.0"
