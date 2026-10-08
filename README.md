# llm-inference-bench

大模型推理部署与压测评测平台 —— 面向**多引擎、多量化等级**的推理性能横向评测工具。

一句话定位：**把「部署完大模型之后怎么证明它跑得好」这件事工程化。**
给定若干推理引擎（vLLM / Ollama）和若干量化配置（FP16 / INT8 / AWQ / GPTQ），
自动做并发梯度压测，采集吞吐、首包延迟、尾延迟、显存占用，产出一份可直接交付客户的
对比报告，并把指标实时暴露给 Prometheus / Grafana。

---

## 为什么需要它

私有化交付场景里，客户真正关心的不是「你部署起来了」，而是：

| 客户的疑问 | 本工具的答案 |
| --- | --- |
| 这套配置能扛多少并发？ | 并发梯度压测曲线：吞吐随并发增长直到饱和拐点 |
| 用户体感快不快？ | TTFT（首包延迟）—— 流式场景下唯一有意义的体感指标 |
| 延迟长尾会不会爆？ | P50 / P95 / P99 分位统计，只看均值会掩盖长尾 |
| 显存够不够跑更大模型？ | 全阶段显存占用曲线 + 峰值（多卡按时序求和） |
| 换量化能省多少显存、掉多少性能？ | FP16 / INT8 / AWQ 横向对比矩阵 + 换算每 GiB 显存吞吐 |
| 部署完怎么持续看？ | Prometheus 指标 + 预置 Grafana 面板（15 个图） |

---

## 架构

```
                          ┌──────────────────────────────┐
                          │        CLI / FastAPI         │
                          │  run_bench.py  api/server.py │
                          └───────────────┬──────────────┘
                                          │
                          ┌───────────────▼──────────────┐
                          │      bench/runner.py         │
                          │  并发梯度扫描 · 预热剔除 ·    │
                          │  每请求独立客户端 · 墙钟计时   │
                          └───┬───────────────────┬──────┘
                              │                   │
              ┌───────────────▼──────┐   ┌────────▼─────────────┐
              │  engines/client.py    │   │  monitor/gpu.py      │
              │  UnifiedLLMClient     │   │  GpuMonitor          │
              │  · 流式 SSE 解析       │   │  · nvidia-smi 轮询    │
              │  · TTFT=首个内容块     │   │  · 显存曲线/峰值      │
              │  · 优先采信服务端 usage│   │  · 后台线程采样       │
              └───────────┬───────────┘   └────────┬─────────────┘
                          │                        │
        ┌─────────────────┴───────┐                │
        │   OpenAI 兼容 HTTP API   │                │
        │  ┌─────────┐ ┌────────┐ │                │
        │  │  vLLM   │ │ Ollama │ │                │
        │  └─────────┘ └────────┘ │                │
        └─────────────────────────┘                │
                          │                        │
              ┌───────────▼────────────────────────▼─────────┐
              │           metrics/stats.py                   │
              │  RequestResult · StageMetrics                │
              │  分位数(线性插值) · 错误分类 · 失败率门禁      │
              └───┬──────────────┬──────────────┬────────────┘
                  │              │              │
      ┌───────────▼───┐  ┌───────▼──────┐  ┌────▼─────────────┐
      │ metrics/store │  │ monitor/quant│  │ metrics/prometheus│
      │ JSON+CSV 落盘  │  │ 量化横向对比  │  │ 直方图/仪表/Info  │
      │ 结果可复现     │  │ 基线自动识别  │  │ → /metrics        │
      └───────┬───────┘  └───────┬──────┘  └────┬─────────────┘
              │                  │              │
      ┌───────▼──────────────────▼──────┐  ┌────▼──────────────┐
      │        report/generator.py      │  │   Prometheus      │
      │  markdown.py  ·  html.py        │  │        ↓          │
      │  svgchart.py（纯内联 SVG 图表）  │  │    Grafana        │
      │  自包含 HTML，离线可用，无 CDN    │  │  15 面板预置仪表盘 │
      └─────────────────────────────────┘  └───────────────────┘
```

---

## 核心设计决策

以下几点是这个工具区别于「随手写个脚本压一压」的地方。

### 1. TTFT 必须在流式下测量

非流式请求只能拿到总耗时，首包延迟无从得知。本工具强制走 `stream=true`，
并把 TTFT 定义为**第一个携带内容的分片到达的时刻**（而非收到 HTTP 200 的时刻）。
服务端提供的 `usage` 字段若存在则优先采信，否则按分片计数自行统计。

`TPOT` 的算法是 `generation_time / (output_tokens - 1)` —— 减去 1 是因为
首个 token 的等待时间已经计入 TTFT，重复计算会低估单 token 生成速度。

### 2. 每个请求使用独立客户端

压测时若所有并发请求复用同一个 HTTP 连接池，测出来的「并发」实际上是在
连接池队列里排队。本工具为每个并发请求创建独立客户端，确保测量的是
推理引擎的并发能力，而不是 HTTP 客户端的排队能力。

### 3. 显存峰值按时序求和，而非取单卡最大值

多卡场景下这是个容易写错的地方：8 张卡各用 60 GiB，真实占用是 480 GiB。
如果代码写成「取各卡峰值的最大者」，结果会少报 8 倍。

实现上按时间戳对齐后求和再取最大值（见 `monitor/gpu.py::GpuMonitor.curve`），
并有专门的回归测试守着（`test_multi_gpu_peak_not_underreported`）。

### 4. 报告是自包含 HTML

内联 SVG 画图、无 CDN、无 `<script>`。原因很实际：客户内网多为隔离环境，
一个依赖外部资源的报告打开就是白屏。

### 5. 量化对比自动识别基线

`BASELINE_QUANTS = {fp16, bf16, float16, bfloat16, none, unknown}` 中出现即视为基线。
对比结论只陈述事实（"awq 峰值 1874.1 tokens/s，比最优低 2.0%"），
不编造胜负 —— 在 mock 环境里 awq 本来就该略慢于 fp16。

---

## 快速开始

### 环境准备

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt     # 运行依赖
pip install -r requirements-dev.txt # 开发/测试依赖
```

### 一分钟跑通（不需要 GPU 和真实模型）

内置了一个 OpenAI 兼容的 **mock 引擎**，带真实感的排队延迟模拟，
用于在无 GPU 环境下验证整条链路。

```bash
# 终端 1：启动 mock 引擎（监听 8102）
python scripts/mock_engine.py --port 8102

# 终端 2：先把 base_url 指向 mock，再跑压测
export NO_PROXY="127.0.0.1,localhost" no_proxy="127.0.0.1,localhost"
export VLLM_BASE_URL="http://127.0.0.1:8102" VLLM_MODEL="mock-72b"
python scripts/run_bench.py --engine vllm --quant fp16 \
    --levels 1,2,4,8,16,32 --max-tokens 256 --output-dir data/results/demo
```

> Windows Git Bash 下必须设置 `NO_PROXY`，否则本地请求可能被系统代理劫持，
> 表现为 `502 upstream connect failed`。

### 打真机

```bash
# 真实 vLLM
python scripts/run_bench.py \
    --base-url http://127.0.0.1:8000/v1 \
    --model Qwen2.5-72B-Instruct \
    --quant fp16 \
    --levels 1,2,4,8,16,32,64 \
    --max-tokens 512 \
    --output-dir data/results/qwen72b

# 真实 Ollama（量化等级写模型名里）
python scripts/run_bench.py \
    --engine ollama --base-url http://127.0.0.1:11434 \
    --model qwen2.5:7b --quant q4_K_M
```

### 量化横向对比

```bash
# 用同一批并发档位分别跑 FP16 / AWQ，产物落在同一 run 目录下
python scripts/run_bench.py --run-id quant-72b --output-dir data/results \
    --base-url http://127.0.0.1:8000/v1 --model Qwen2.5-72B-Instruct --quant fp16
python scripts/run_bench.py --run-id quant-72b --output-dir data/results \
    --base-url http://127.0.0.1:8000/v1 --model Qwen2.5-72B-AWQ --quant awq

# 报告里会自动生成量化对比章节
```

### 从已有结果重新出报告

```bash
python scripts/run_bench.py --report-only --from-results data/results/demo
```

### 服务化

```bash
python scripts/serve.py --host 0.0.0.0 --port 8090
```

因为一次完整扫描要跑几分钟，接口设计成**异步任务模型**：

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/` | 服务信息与端点清单 |
| GET | `/health` | 存活探针 |
| GET | `/engines` | 列出已配置引擎 |
| POST | `/engines/check` | 探测引擎可达性与模型是否已加载 |
| POST | `/bench` | 提交压测任务，**立即返回 job_id** |
| GET | `/jobs` | 任务列表 |
| GET | `/jobs/{id}` | 轮询进度（`progress` 0→1） |
| GET | `/jobs/{id}/result` | 取结果 JSON |
| GET | `/jobs/{id}/report` | 取 HTML 报告 |
| DELETE | `/jobs/{id}` | 删除任务记录 |
| GET | `/reports` | 列出历史报告 |
| GET | `/reports/{run_id}/{filename}` | 按 run 取报告文件 |
| POST | `/compare` | 对同一 run 下多份结果做量化对比 |
| GET | `/metrics` | Prometheus 抓取端点 |
| GET | `/metrics/snapshot` | 指标快照（JSON，用于调试） |

### 监控栈

```bash
cd deploy && docker compose up -d
# Prometheus  → http://localhost:9090
# Grafana     → http://localhost:3000  (admin/admin，匿名只读)
```

Grafana 数据源与仪表盘由 provisioning 自动装载，仪表盘 JSON 由
`scripts/gen_grafana_dashboard.py` 生成（15 面板，含引擎模板变量）。

---

## 项目结构

```
llm-inference-bench/
├── config/settings.py        # EngineTarget / BenchConfig / MonitorConfig
├── engines/client.py         # 统一 OpenAI 兼容客户端（流式 TTFT 采集）
├── bench/runner.py           # 并发梯度扫描、预热剔除、显存窗口对齐
├── metrics/
│   ├── stats.py              # RequestResult / StageMetrics / 分位数 / 错误分类
│   ├── store.py              # JSON+CSV 落盘、结果反序列化、文件命名防碰撞
│   └── prometheus.py         # 直方图/仪表/Info，定制 LLM 延迟桶
├── monitor/
│   ├── gpu.py                # nvidia-smi 轮询、显存曲线（时序求和峰值）
│   └── quant.py              # 量化分组、基线识别、对比矩阵、结论生成
├── report/
│   ├── generator.py          # 报告编排入口
│   ├── markdown.py           # 7 章节 Markdown 报告
│   ├── html.py               # 自包含 HTML 报告
│   ├── svgchart.py           # 纯内联 SVG 折线图/柱状图（含对数轴）
│   └── fmt.py                # 自适应单位格式化
├── api/server.py             # FastAPI 服务（异步任务模型）
├── scripts/
│   ├── run_bench.py          # CLI 入口
│   ├── serve.py              # 服务启动器
│   ├── mock_engine.py        # OpenAI 兼容 mock 引擎（无 GPU 也能验证）
│   └── gen_grafana_dashboard.py
├── deploy/                   # Dockerfile / docker-compose / Prometheus / Grafana
├── tests/                    # 302 个单元测试
└── data/results/             # 压测产物（JSON / CSV / 报告）
```

---

## 报告示例

报告分 7 个章节：

1. **概览** —— 被测配置、并发档位、结论速览
2. **测试环境** —— Python / 平台 / GPU 型号驱动、被测模型与量化等级
3. **吞吐能力** —— 各并发档位 tokens/s，含吞吐-并发曲线
4. **延迟特性** —— TTFT / TPOT / P50 / P95 / P99 表格，含延迟-并发曲线
5. **显存占用** —— 全阶段显存曲线与峰值，按并发档位拆表
6. **量化横向对比** —— 基线识别、吞吐相对比、每 GiB 显存吞吐
7. **结论与建议** —— 饱和拐点、稳定性提示、选型建议

产物同时输出 `report.md` 与 `report.html`。

---

## 测试

```bash
python -m pytest tests -q
```

302 个测试，约 12 秒跑完，覆盖：

| 测试文件 | 用例数 | 重点 |
| --- | --- | --- |
| `test_stats.py` | 43 | 分位数插值、TPOT 边界（单 token）、错误分类 |
| `test_quant.py` | 46 | 基线识别、比值计算、结论措辞 |
| `test_gpu_monitor.py` | 34 | 多卡时序求和峰值、nvidia-smi 缺失降级 |
| `test_runner.py` | 21 | 并发扫描、预热剔除、失败率门禁 |
| `test_store.py` | 29 | 落盘/回读一致性、文件名防碰撞 |
| `test_report.py` | 61 | Markdown/HTML 章节、SVG 无外部引用 |
| `test_api.py` | 35 | 端点契约、异步任务生命周期 |
| `test_config.py` | 33 | 环境变量覆盖、URL 归一化 |

写测试的过程中发现并修掉了 4 个真实缺陷（详见 `CHANGELOG.md`）。

---

## 已知限制

- **mock 引擎的延迟是模拟的**，只用于验证链路，不能用来推断真实硬件性能。
- 显存监控依赖 `nvidia-smi`，非 NVIDIA 平台会静默降级为「无显存数据」。
- `/compare` 目前只能在**同一个 run 目录**内对比，跨 run 对比需先合并结果文件。
- Prometheus 直方图是累计值，跨多次压测会持续累加，重启服务才归零。
- 未实现多机分布式压测；单进程压测在极高并发（>256）下可能受 GIL 限制。

---

## License

MIT，见 `LICENSE`。
