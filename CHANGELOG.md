# Changelog

本项目遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/) 与
[语义化版本](https://semver.org/lang/zh-CN/)。

## [0.1.0] - 2026-10-08

首个可用版本：从零搭出「多引擎 + 多量化」的大模型推理压测与评测平台。

### Added

**配置层**
- `EngineTarget`：支持 vLLM / Ollama / 任意 OpenAI 兼容端点，含量化等级与备注字段
- `BenchConfig`：并发梯度、每档请求数、预热数、`max_tokens`、超时、采样温度等全部可配
- `MonitorConfig`：显存采样间隔与开关
- 全部配置支持环境变量覆盖（`_env` / `_env_int` / `_env_float`）
- `EngineTarget.normalized_base_url()` 统一处理 `/v1` 后缀有无两种写法

**引擎适配层**
- `UnifiedLLMClient`：vLLM 与 Ollama 共用一套 OpenAI 兼容协议，无需各写一套
- `health()` 探测可达性 + 模型是否已加载；模型名匹配支持双向前缀（`qwen2.5` ↔ `qwen2.5:7b`）
- 流式解析：TTFT 取**首个携带内容的分片**到达时刻，而非 HTTP 响应头到达时刻
- 服务端返回 `usage` 时优先采信，否则按分片自行统计 token 数
- `stream_options={"include_usage": True}` 以拿到准确 token 统计

**压测执行层**
- 并发梯度扫描：`concurrency_levels` 逐档压测，档间结果独立统计
- **每请求独立客户端**，避免连接池排队污染并发测量
- 预热请求自动剔除，不进入统计
- 墙钟计时（`time.perf_counter`），不受系统时钟调整影响
- 失败率门禁：`max_failure_rate` 超限的阶段标记为不稳定并单独提示
- 自动启停显存监控，并按阶段时间窗口切分显存曲线

**指标统计层**
- `RequestResult`：单请求的 TTFT / TPOT / 吞吐 / 延迟 / token 数 / 错误
- `StageMetrics`：单档位的分位数、均值、标准差、失败率、错误分类分布
- 分位数采用线性插值法（与 NumPy 默认行为一致）
- 错误分类：timeout / connection / oom / rate_limit / server_error / bad_request / other
- `MAX_ERROR_SAMPLES = 5`：只保留 5 条错误样本，防止全失败档位把结果文件撑爆

**显存监控层**
- `GpuMonitor`：后台线程轮询 `nvidia-smi`，主流程零阻塞
- `nvidia-smi` 不存在时静默降级，不抛异常
- 显存曲线按时间戳对齐，**峰值取「同一时刻各卡之和」的最大值**
- 支持上下文管理器（`with GpuMonitor(...) as m:`）

**量化评测层**
- `QuantGroup` / `pick_baseline()`：自动识别 FP16/BF16 为基线
- `compare()`：生成对比矩阵，含吞吐相对比、TTFT/TPOT 变化百分比
- `throughput_per_gib`：每 GiB 显存产出吞吐，用于量化性价比判断
- `_build_conclusion()`：只陈述可验证的事实，不编造胜负

**报告层**
- Markdown 报告（7 章节）与 HTML 报告双产出
- `svgchart.py`：纯手写内联 SVG 折线图/柱状图，含对数横轴（范围 ≥ 8 倍时自动切换）
- **自包含 HTML**：无 CDN、无 `<script>`、无外部字体，隔离内网可直接打开
- `fmt.py`：自适应单位（ns/µs/ms/s、MiB/GiB）

**服务层**
- FastAPI 服务，异步任务模型（`POST /bench` 立即返回 `job_id`，轮询 `GET /jobs/{id}`）
- 任务注册表带锁保护，支持进度上报
- `/compare` 对同一 run 下多份结果做量化对比
- `/metrics` 暴露 Prometheus 指标，`/metrics/snapshot` 提供 JSON 调试视图

**监控栈**
- Prometheus 指标：请求计数、延迟/ TTFT / TPOT 直方图、吞吐仪表、失败率、显存、功耗
- **定制 LLM 延迟分桶**（默认桶对 LLM 场景过粗，P99 会落在边界上）
- `scripts/gen_grafana_dashboard.py` 生成 15 面板 Grafana 仪表盘 JSON
- Docker Compose 一键起 Prometheus + Grafana，数据源与面板自动 provisioning

**工程配套**
- 302 个单元测试，约 12 秒跑完
- `scripts/mock_engine.py`：OpenAI 兼容 mock 引擎，带排队延迟模拟，无 GPU 也能验证全链路
- `pytest.ini` / `requirements.txt` / `requirements-dev.txt` / `.env.example` / `.gitignore`

### Fixed

以下是写测试过程中发现的**真实缺陷**，均为本版本内修复。

- **多卡显存峰值被严重低估**
  `GpuMonitor.curve()` 原先取「各卡各自峰值的最大值」。8 卡各占 60 GiB 时，
  真实占用 480 GiB，而原实现只报 60 GiB，少报 8 倍。
  改为按时间戳对齐后求和再取峰值，并补充回归测试
  `test_multi_gpu_peak_not_underreported`。

- **Prometheus 运行元信息只保留了最后一个字段**
  `set_run_info()` 原先对每个字段单独调用一次 `Info.info({key: value})`。
  但 `prometheus_client` 的 `Info.info()` 每次调用会**整体替换**标签集合，
  导致逐字段调用后只剩最后一个字段。改为一次性传入完整字典。

- **多配置压测结果互相覆盖**
  结果文件名原先只是 `<engine>.json`，同一引擎跑 FP16 和 AWQ 会互相覆盖，
  量化对比直接失效。引入 `result_stem(target)`，文件名变为
  `vllm__Qwen2.5-72B-Instruct__fp16.json`，量化等级为 unknown 时省略该段。

- **全部请求失败的运行以退出码 0 静默结束**
  CLI 在所有配置都失败时，报告里满屏 "N/A"，但进程却返回成功，
  CI 里会当成通过了。现在 `run_one()` 会打印带首个错误、健康探测详情、
  模型名提示的醒目失败块，`main()` 在无任何有效数据时返回退出码 `2`。

### Changed

- 错误样本列表由无上限改为最多保留 5 条（避免结果文件膨胀数十倍）
- 测试套件用 `os.utime()` 控制文件修改时间来排序结果，
  替换原先的 `time.sleep()`，套件耗时由 66 秒降至 19 秒

### Notes

- 本版本未实现多机分布式压测；单进程在极高并发（>256）下可能受 GIL 限制
- `/compare` 目前仅支持同一 run 目录内的结果对比
- Prometheus 直方图为累计值，跨压测持续累加，重启服务才归零
