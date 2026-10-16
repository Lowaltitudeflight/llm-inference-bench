# llm-inference-bench

**LLM inference benchmarking and evaluation platform** — cross-engine, cross-quantisation performance comparison for self-hosted LLM serving.

[![Stars](https://img.shields.io/github/stars/Lowaltitudeflight/llm-inference-bench?style=flat&color=yellow)](https://github.com/Lowaltitudeflight/llm-inference-bench/stargazers)
[![Forks](https://img.shields.io/github/forks/Lowaltitudeflight/llm-inference-bench?style=flat&color=blue)](https://github.com/Lowaltitudeflight/llm-inference-bench/network/members)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-313%20passed-brightgreen.svg)](tests/)
[![Version](https://img.shields.io/badge/release-v0.1.0-orange.svg)](https://github.com/Lowaltitudeflight/llm-inference-bench/releases)

**English** ｜ [简体中文](README.zh-CN.md)

**One line:** this project makes *"how do I prove the model actually performs well after deployment"* an engineering problem rather than an argument.

Given a set of inference engines (vLLM / Ollama) and a set of quantisation configurations (FP16 / INT8 / AWQ / GPTQ), it runs an automatic concurrency sweep, collects throughput, time-to-first-token, tail latency and VRAM usage, and produces a comparison report you can hand to a customer — while exposing the metrics live to Prometheus / Grafana.

> **Timeline**: core development 2026.03 – 2026.06; open-source cleanup (docs, containerisation, CI) in 2026.09.
> The project grew out of a real 8-GPU private deployment of Qwen2.5-72B during an internship, focused on eliminating inconsistent benchmark definitions and irreproducible conclusions, and produced a reusable set of metric definitions and evaluation methods.

### At a glance

```bash
# Runs end to end without a GPU — a built-in OpenAI-compatible mock engine
pip install -r requirements.txt
python scripts/mock_engine.py --port 8102 &
export NO_PROXY="127.0.0.1,localhost"
export VLLM_BASE_URL="http://127.0.0.1:8102" VLLM_MODEL="mock-72b"
python scripts/run_bench.py --engine vllm --quant fp16 --levels 1,2,4,8,16,32
# → data/results/<run>/report.html  (self-contained HTML, opens directly on an isolated network)
```

| | |
| --- | --- |
| **Engines** | vLLM, Ollama, and any OpenAI-compatible endpoint |
| **Metrics** | throughput (tokens/s) · TTFT · TPOT · P50/P95/P99 tail latency · peak VRAM |
| **Quantisation** | FP16 / BF16 / INT8 / AWQ / GPTQ comparison + throughput per GiB of VRAM |
| **Output** | Markdown report · self-contained HTML report (inline SVG, no CDN) · Prometheus metrics · Grafana dashboard |
| **Measured** | Benchmarked on **real hardware** (RTX 4060 Laptop, 8 GB): **30.8 tokens/s** peak, saturating at concurrency 2 — see [Measured results](#measured-results) |
| **Verified** | 313 unit tests, ~7 s to run; full pipeline verifiable without a GPU |
| **Deployable** | One-command Docker Compose monitoring stack; works offline and on isolated networks |

### How it differs from "just writing a load-test script"

- **TTFT is measured at the first streamed content chunk** — not when HTTP 200 arrives, which would hide queueing and prefill
- **TPOT = generation time / (output tokens − 1)** — the first token is already counted in TTFT; counting it twice understates decode speed
- **Multi-GPU peak VRAM is summed over time** — taking the max of per-GPU peaks under-reports by 8× on an 8-GPU box
- **Each concurrent request gets its own client** — a shared connection pool turns "concurrency" into "queueing"
- **Reports are self-contained** — customer networks are often physically isolated, and a report depending on external assets opens to a blank page

> The traps behind these definitions are documented in [Benchmarking Methodology](docs/methodology.md), with derivations and a quick-reference table of common pitfalls.

---

## Why you need it

In private deployments the customer does not really care that "you got it running". They care about:

| The customer's question | This tool's answer |
| --- | --- |
| How much concurrency can this handle? | Concurrency sweep curve: throughput rises until the saturation knee |
| Does it feel fast to users? | TTFT — the only meaningful perceived-latency metric for streaming |
| Will tail latency blow up? | P50 / P95 / P99 percentiles; averages hide the tail |
| Is there VRAM headroom for a bigger model? | Full-stage VRAM curve plus peak (summed across GPUs over time) |
| How much VRAM does quantisation save, and what does it cost in speed? | FP16 / INT8 / AWQ comparison matrix plus throughput per GiB |
| How do I keep watching it after go-live? | Prometheus metrics plus a pre-built Grafana dashboard (15 panels) |

---

## Architecture

```
                          ┌──────────────────────────────┐
                          │        CLI / FastAPI         │
                          │  run_bench.py  api/server.py │
                          └───────────────┬──────────────┘
                                          │
                          ┌───────────────▼──────────────┐
                          │      bench/runner.py         │
                          │  concurrency sweep · warmup  │
                          │  pruning · per-request client│
                          │  · wall-clock timing         │
                          └───┬───────────────────┬──────┘
                              │                   │
              ┌───────────────▼──────┐   ┌────────▼─────────────┐
              │  engines/client.py    │   │  monitor/gpu.py      │
              │  UnifiedLLMClient     │   │  GpuMonitor          │
              │  · streaming SSE      │   │  · nvidia-smi poll   │
              │  · TTFT = first chunk │   │  · VRAM curve/peak   │
              │  · prefers server     │   │  · background thread │
              │    usage fields       │   │                      │
              └───────────┬───────────┘   └────────┬─────────────┘
                          │                        │
        ┌─────────────────┴───────┐                │
        │  OpenAI-compatible API  │                │
        │  ┌─────────┐ ┌────────┐ │                │
        │  │  vLLM   │ │ Ollama │ │                │
        │  └─────────┘ └────────┘ │                │
        └─────────────────────────┘                │
                          │                        │
              ┌───────────▼────────────────────────▼─────────┐
              │           metrics/stats.py                   │
              │  RequestResult · StageMetrics                │
              │  percentiles (linear interpolation) ·        │
              │  error classification · failure-rate gate    │
              └───┬──────────────┬──────────────┬────────────┘
                  │              │              │
      ┌───────────▼───┐  ┌───────▼──────┐  ┌────▼─────────────┐
      │ metrics/store │  │ monitor/quant│  │ metrics/prometheus│
      │ JSON+CSV      │  │ quantisation │  │ histograms/gauges │
      │ reproducible  │  │ baseline ID  │  │ → /metrics        │
      └───────┬───────┘  └───────┬──────┘  └────┬─────────────┘
              │                  │              │
      ┌───────▼──────────────────▼──────┐  ┌────▼──────────────┐
      │        report/generator.py      │  │   Prometheus      │
      │  markdown.py  ·  html.py        │  │        ↓          │
      │  svgchart.py (pure inline SVG)  │  │    Grafana        │
      │  self-contained HTML, offline   │  │  15-panel board   │
      └─────────────────────────────────┘  └───────────────────┘
```

---

## Core design decisions

The following are what separates this tool from "a script someone wrote to hammer the endpoint".

### 1. TTFT must be measured on a streaming response

A non-streaming request only yields a total duration, from which first-token latency cannot be recovered. This tool forces `stream=true` and defines TTFT as **the moment the first chunk carrying content arrives** (not the moment HTTP 200 is received). If the server provides a `usage` field it is preferred; otherwise token counts are derived from the chunk stream.

`TPOT` is computed as `generation_time / (output_tokens - 1)` — the minus one exists because waiting for the first token is already counted in TTFT, and counting it twice understates per-token decode speed.

### 2. Each request uses its own client

If all concurrent requests share one HTTP connection pool, what you measure as "concurrency" is really queueing inside the pool. This tool creates an independent client per concurrent request, so the measurement reflects the engine's concurrency capability rather than the HTTP client's queueing behaviour.

### 3. Multi-GPU peak VRAM is summed over time, not maxed per GPU

This is an easy mistake on multi-GPU hosts: 8 GPUs each using 60 GiB is 480 GiB in total. Code that takes "the maximum of the per-GPU peaks" under-reports by 8×.

The implementation aligns samples by timestamp, sums, then takes the maximum (see `monitor/gpu.py::GpuMonitor.curve`), and a dedicated regression test guards it (`test_multi_gpu_peak_not_underreported`).

### 4. Reports are self-contained HTML

Charts are inline SVG — no CDN, no `<script>`. The reason is practical: customer networks are often isolated, and a report that depends on external resources opens to a blank page.

### 5. Quantisation comparison identifies the baseline automatically

Anything in `BASELINE_QUANTS = {fp16, bf16, float16, bfloat16, none, unknown}` is treated as the baseline. Conclusions state facts only ("awq peaks at 1874.1 tokens/s, 2.0% below the best") and do not invent a winner — in a mock environment `awq` genuinely should be slightly slower than `fp16`.

---

## Quick start

### Setup

```bash
python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install -r requirements.txt      # runtime dependencies
pip install -r requirements-dev.txt  # dev/test dependencies
```

### Full pipeline in one minute (no GPU, no real model)

A built-in OpenAI-compatible **mock engine** with realistic queueing-delay simulation lets you verify the whole pipeline without a GPU.

```bash
# Terminal 1: start the mock engine (listening on 8102)
python scripts/mock_engine.py --port 8102

# Terminal 2: point base_url at the mock, then benchmark
export NO_PROXY="127.0.0.1,localhost" no_proxy="127.0.0.1,localhost"
export VLLM_BASE_URL="http://127.0.0.1:8102" VLLM_MODEL="mock-72b"
python scripts/run_bench.py --engine vllm --quant fp16 \
    --levels 1,2,4,8,16,32 --max-tokens 256 --output-dir data/results/demo
```

> On Windows Git Bash you must set `NO_PROXY`, or local requests may be hijacked by a system proxy and fail with `502 upstream connect failed`.

### Benchmarking real hardware

```bash
# Real vLLM
python scripts/run_bench.py \
    --base-url http://127.0.0.1:8000/v1 \
    --model Qwen2.5-72B-Instruct \
    --quant fp16 \
    --levels 1,2,4,8,16,32,64 \
    --max-tokens 512 \
    --output-dir data/results/qwen72b

# Real Ollama (quantisation level is part of the model name)
python scripts/run_bench.py \
    --engine ollama --base-url http://127.0.0.1:11434 \
    --model qwen2.5:7b --quant q4_K_M
```

### Quantisation comparison

```bash
# Run FP16 and AWQ over the same concurrency levels into one run directory
python scripts/run_bench.py --run-id quant-72b --output-dir data/results \
    --base-url http://127.0.0.1:8000/v1 --model Qwen2.5-72B-Instruct --quant fp16
python scripts/run_bench.py --run-id quant-72b --output-dir data/results \
    --base-url http://127.0.0.1:8000/v1 --model Qwen2.5-72B-AWQ --quant awq

# A quantisation comparison section is generated automatically
```

### Re-generating a report from existing results

```bash
python scripts/run_bench.py --report-only --from-results data/results/demo
```

### Serving mode

```bash
python scripts/serve.py --host 0.0.0.0 --port 8090
```

Because a full sweep takes several minutes, the API is designed around an **asynchronous job model**:

| Method | Path | Description |
| --- | --- | --- |
| GET | `/` | Service info and endpoint list |
| GET | `/health` | Liveness probe |
| GET | `/engines` | List configured engines |
| POST | `/engines/check` | Probe engine reachability and whether the model is loaded |
| POST | `/bench` | Submit a benchmark job, **returns `job_id` immediately** |
| GET | `/jobs` | Job list |
| GET | `/jobs/{id}` | Poll progress (`progress` 0→1) |
| GET | `/jobs/{id}/result` | Fetch result JSON |
| GET | `/jobs/{id}/report` | Fetch the HTML report |
| DELETE | `/jobs/{id}` | Delete a job record |
| GET | `/reports` | List historical reports |
| GET | `/reports/{run_id}/{filename}` | Fetch a report file by run |
| POST | `/compare` | Compare multiple results within one run |
| GET | `/metrics` | Prometheus scrape endpoint |
| GET | `/metrics/snapshot` | Metric snapshot (JSON, for debugging) |

> ⚠️ **Before exposing this service beyond localhost:** the endpoints ship without authentication, and `base_url` in a `/bench` request body is caller-controlled — that is an SSRF primitive against your internal network. Put it behind an authenticating gateway and restrict permitted target hosts.

### Monitoring stack

```bash
cd deploy && docker compose up -d
# Prometheus  → http://localhost:9090
# Grafana     → http://localhost:3000  (admin/admin; anonymous read-only)
```

The Grafana data source and dashboard are provisioned automatically; the dashboard JSON is generated by `scripts/gen_grafana_dashboard.py` (15 panels, including engine template variables).

---

## Project structure

```
llm-inference-bench/
├── config/settings.py        # EngineTarget / BenchConfig / MonitorConfig
├── engines/client.py         # Unified OpenAI-compatible client (streaming TTFT capture)
├── bench/runner.py           # Concurrency sweep, warmup pruning, VRAM window alignment
├── metrics/
│   ├── stats.py              # RequestResult / StageMetrics / percentiles / error classification
│   ├── store.py              # JSON+CSV persistence, deserialisation, collision-safe filenames
│   └── prometheus.py         # Histograms/gauges/info, custom LLM latency buckets
├── monitor/
│   ├── gpu.py                # nvidia-smi polling, VRAM curve (time-summed peak)
│   └── quant.py              # Quantisation grouping, baseline detection, comparison matrix
├── report/
│   ├── generator.py          # Report orchestration entry point
│   ├── markdown.py           # 7-section Markdown report
│   ├── html.py               # Self-contained HTML report
│   ├── svgchart.py           # Pure inline SVG line/bar charts (incl. log axes)
│   └── fmt.py                # Adaptive unit formatting
├── api/server.py             # FastAPI service (async job model)
├── scripts/
│   ├── run_bench.py          # CLI entry point
│   ├── serve.py              # Service launcher
│   ├── mock_engine.py        # OpenAI-compatible mock engine (verify without a GPU)
│   └── gen_grafana_dashboard.py
├── deploy/                   # Dockerfile / docker-compose / Prometheus / Grafana
├── tests/                    # 313 unit tests
└── data/results/             # Benchmark artefacts (JSON / CSV / reports)
```

---

## Report format

The report has 7 sections:

1. **Overview** — configuration under test, concurrency levels, headline conclusions
2. **Test environment** — Python / platform / GPU model and driver, model and quantisation under test
3. **Throughput** — tokens/s per concurrency level, with a throughput-vs-concurrency curve
4. **Latency** — TTFT / TPOT / P50 / P95 / P99 tables, with a latency-vs-concurrency curve
5. **VRAM usage** — full-stage VRAM curve and peak, broken down per concurrency level
6. **Quantisation comparison** — baseline detection, relative throughput, throughput per GiB
7. **Conclusions and recommendations** — saturation knee, stability warnings, selection advice

Both `report.md` and `report.html` are produced.

---

## Measured results

> The run below used **real hardware and a real model**, not the mock: an 8 GiB consumer GPU running
> Ollama with `qwen2.5:7b` (Q4_K_M). The raw artefacts are **published in this repository** so you can
> verify them, or recompute the report yourself:
>
> - [`report.md`](data/results/run-20261009-103334/report.md) — full 7-section report (readable on GitHub)
> - [`report.html`](data/results/run-20261009-103334/report.html) — self-contained chart version (inline SVG, no external assets)
> - [`ollama__qwen2.5_7b__q4_k_m.json`](data/results/run-20261009-103334/ollama__qwen2.5_7b__q4_k_m.json) — raw results, feed it to `--report-only` to regenerate
>
> ⓘ The `result_stem` field inside that JSON still uses the pre-fix format
> (`ollama__qwen2.5:7b__q4_k_m`, **with a colon**), while the actual filename was produced by the
> fixed `_safe_slug`. This is not edited data — it is the scene of the "colon triggers an NTFS
> alternate data stream" bug. See the Fixed section of [`CHANGELOG.md`](CHANGELOG.md).

![Real RTX 4060 Laptop benchmark: throughput saturates at concurrency 8 while time-to-first-token keeps degrading](docs/images/real-gpu-benchmark-charts.png)

> Read it in one glance: **throughput flattens at concurrency 8 (30.8 tokens/s), while TTFT climbs from 169 ms to 30.8 s.**

**Environment and parameters**

| Item | Value |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU, 8.0 GiB |
| Engine / model | Ollama / `qwen2.5:7b` (Q4_K_M) |
| Concurrency levels | 1 → 32 (1, 2, 4, 8, 16, 32) |
| Per-request generation | 256 tokens, temperature 0, streaming, 2 warmup runs |
| Scale | 63 requests, **0 failures**, 222.94 s total |

**Throughput and latency**

| Concurrency | Throughput (tokens/s) | Speedup | Mean TTFT | Mean TPOT | End-to-end P99 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 26.9 | 1.00× | 168.81 ms | 21.64 ms | 2159.52 ms |
| 2 | 27.3 | 1.02× | 1335.32 ms | 23.93 ms | 4607.57 ms |
| 4 | 30.4 | 1.13× | 3636.49 ms | 23.44 ms | 8871.39 ms |
| **8** | **30.8 (peak)** | 1.14× | 7804.78 ms | 22.96 ms | 17178.87 ms |
| 16 | 30.0 | 1.11× | 15979.12 ms | 21.56 ms | 32414.35 ms |
| 32 | 29.3 (declining) | 1.09× | **30839.10 ms** | 21.69 ms | **62798.84 ms** |

Peak VRAM **5.6 / 8.0 GiB (70.3%)**; average GPU utilisation 71.9% across the run (305 samples).

**Three conclusions**

1. **It saturates at concurrency 2.** Going from concurrency 1 to peak throughput buys only a
   **1.14×** speedup, after which adding load reduces throughput. The capacity ceiling of a 7B
   Q4_K_M model on an 8 GiB card is very low — if you plan capacity as "concurrency = number of
   users", the third user is already queuing.
2. **Past the saturation point, more load only adds latency.** From concurrency 1 to 32, TTFT
   degrades from 168.81 ms to **30839.10 ms (182×)**, and end-to-end P99 rises from 2.2 s to
   62.8 s. This is direct evidence that "reporting only peak throughput yields the wrong capacity
   conclusion", and the reason this project sweeps concurrency instead of measuring a single point.
3. **TPOT barely moves with concurrency (21.6 – 23.9 ms).** Per-token decode speed is stable,
   which means the bottleneck is request queueing and scheduling, not saturated compute. To
   raise throughput you have to work on scheduling, not swap in a faster decode kernel.

> ⚠️ **Do not confuse this with the demo:** the bundled [`data/results/demo/`](data/results/demo/)
> uses the built-in **mock engine** (latency is simulated) and exists only to verify the pipeline
> with no GPU. It **cannot be used to infer any real hardware performance**. The table above is
> currently the only real-hardware dataset in the repository.

---

## Tests

```bash
python -m pytest tests -q
```

313 tests, ~7 seconds, covering:

| Test file | Cases | Focus |
| --- | --- | --- |
| `test_stats.py` | 43 | Percentile interpolation, TPOT boundary (single token), error classification |
| `test_quant.py` | 46 | Baseline detection, ratio computation, conclusion wording |
| `test_gpu_monitor.py` | 34 | Multi-GPU time-summed peak, graceful degradation without nvidia-smi |
| `test_runner.py` | 21 | Concurrency sweep, warmup pruning, failure-rate gate |
| `test_store.py` | 40 | Write/read-back consistency, filename collision safety (incl. Windows reserved characters) |
| `test_report.py` | 61 | Markdown/HTML sections, SVG with no external references |
| `test_api.py` | 35 | Endpoint contracts, async job lifecycle |
| `test_config.py` | 33 | Environment variable overrides, URL normalisation |

Writing the tests and running the real-hardware benchmark surfaced and fixed 6 genuine defects
(see the Fixed section of [`CHANGELOG.md`](CHANGELOG.md)):

1. Multi-GPU peak VRAM took the per-GPU maximum → now summed over timestamps (was under-reporting 8× on 8 GPUs)
2. `prometheus_client`'s `Info.info()` replaces the whole label set when called per field → now passes one complete dict
3. Result filenames used only the engine name, so FP16 and AWQ overwrote each other → now `<engine>__<model>__<quant>`
4. A run where every request failed still exited 0 (CI would pass incorrectly) → now returns 2 and prints failure details
5. The error sample list had no cap → limited to 5 entries, preventing results files from ballooning
6. `_safe_slug` allowed `:`, so `qwen2.5:7b` triggered an NTFS alternate data stream → the result file was written into the ADS and only a 0-byte file remained in the directory (found while running Ollama on real hardware)

---

## Relationship to similar tools

This is not intended to replace them; the positioning differs:

| Tool | Positioning | How this project differs |
| --- | --- | --- |
| `vllm bench` / `vllm benchmark_serving` | Official vLLM benchmark scripts | vLLM only; this project compares multiple engines and produces a customer-ready report |
| `locust` / `wrk` / `ab` | General HTTP load testing | No LLM awareness: no TTFT / TPOT / token throughput, and no way to align VRAM curves |
| `genai-perf` (NVIDIA) | LLM inference profiling | More focused on deep single-model profiling; this project focuses on **multi-config comparison and quantisation decisions** |
| `evalscope` (Alibaba) | Model capability evaluation | Evaluates *how well the model answers*; this evaluates *how fast inference runs* — complementary |

The scenario this project is built for is **private deployment delivery**: given several engine and
quantisation combinations, produce the conclusion "which one should the customer run", plus the
material to put in the acceptance report.

---

## Contributing

Issues and pull requests are welcome. Before you start, read [`CONTRIBUTING.md`](CONTRIBUTING.md) —
it lists 5 **metric definitions you should not assume away**, and you need to read them before
touching any metric-related logic, or you will very likely get it wrong.

```bash
# Quick start, no GPU required
pip install -r requirements.txt -r requirements-dev.txt
python -m pytest tests -q          # 313 passed in ~7s
python scripts/mock_engine.py --port 8102 &
```

If you find a metric definition that is wrong, a misleading statement in a report, or a
counter-intuitive result on your own hardware, please open an issue — **this kind of feedback is
more valuable than a new feature.**

---

## Known limitations

- **The mock engine simulates latency** and exists only to verify the pipeline; it cannot be used
  to infer real hardware performance (the bundled `data/results/demo/` is such a case; real data is
  in [Measured results](#measured-results)).
- VRAM monitoring depends on `nvidia-smi`; on non-NVIDIA platforms it silently degrades to "no VRAM data".
- `/compare` currently only compares results **within a single run directory**; cross-run comparison requires merging result files first.
- Prometheus histograms are cumulative and keep accumulating across benchmark runs; only a service restart resets them.
- Multi-machine distributed load generation is not implemented; single-process benchmarking may be GIL-limited at very high concurrency (>256).

---

## License

MIT, see [`LICENSE`](LICENSE).
