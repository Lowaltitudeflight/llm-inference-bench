# 部署与运维

面向私有化交付现场。包含依赖准备、容器化部署、监控栈接线、隔离内网注意事项。

---

## 一、环境要求

| 组件 | 要求 | 说明 |
| --- | --- | --- |
| Python | ≥ 3.10 | 使用了 `X \| None` 类型语法与 `dataclass` |
| OS | Linux / Windows | 显存监控仅 NVIDIA 平台有效 |
| GPU | NVIDIA + 可用 `nvidia-smi` | 无 GPU 时静默降级，压测仍可跑 |
| 磁盘 | ≥ 1 GB | 结果 JSON + 报告产物 |
| 内存 | ≥ 2 GB | 压测客户端本身占用很小 |

**压测客户端不要和推理引擎挤在同一张卡上。** 客户端的 CPU 开销
（JSON 解析、SSE 处理）在并发 64+ 时不可忽略，同卡会互相干扰。

---

## 二、裸机部署

### 2.1 准备环境

```bash
git clone <repo-url> llm-inference-bench
cd llm-inference-bench

python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate

pip install -r requirements.txt
pip install -r requirements-dev.txt  # 仅开发/测试需要
```

### 2.2 配置

复制 `.env.example` 为 `.env` 并按现场情况修改：

```bash
cp .env.example .env
```

关键变量：

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `VLLM_BASE_URL` | `http://127.0.0.1:8000/v1` | vLLM 的 OpenAI 兼容端点 |
| `VLLM_MODEL` | — | 模型名，需与引擎实际加载的一致 |
| `OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | Ollama 端点 |
| `OLLAMA_MODEL` | `qwen2.5:7b` | Ollama 模型名（含量化标签） |
| `BENCH_CONCURRENCY_LEVELS` | `1,2,4,8,16,32` | 并发梯度 |
| `BENCH_REQUESTS_PER_LEVEL` | — | 每档请求数 |
| `BENCH_MAX_TOKENS` | `256` | 输出上限，**横向对比时必须一致** |
| `BENCH_TIMEOUT` | `120` | 单请求超时（秒） |
| `BENCH_MAX_FAILURE_RATE` | `0.2` | 超过则标记该档位不稳定 |
| `API_HOST` / `API_PORT` | `0.0.0.0` / `8090` | 服务监听 |
| `GPU_SAMPLE_INTERVAL` | — | 显存采样间隔（秒），建议 ≤ 0.5 |
| `LOG_LEVEL` | `INFO` | 日志级别 |

### 2.3 验证

```bash
# 不需要 GPU：起 mock 引擎
python scripts/mock_engine.py --port 8102 &

export NO_PROXY="127.0.0.1,localhost" no_proxy="127.0.0.1,localhost"
export VLLM_BASE_URL="http://127.0.0.1:8102" VLLM_MODEL="mock-72b"

python scripts/run_bench.py --engine vllm --levels 1,4 --max-tokens 32 \
    --output-dir data/results/selftest
```

看到报告生成即环境就绪。之后把 `VLLM_BASE_URL` 换回真实引擎即可。

---

## 三、容器化部署（监控栈）

`deploy/` 下提供完整编排。

```bash
cd deploy
docker compose up -d
docker compose ps
```

启动后：

| 服务 | 地址 | 凭据 |
| --- | --- | --- |
| Prometheus | http://localhost:9090 | 无 |
| Grafana | http://localhost:3000 | `admin` / `admin`（匿名只读） |

### 3.1 接线说明

```
llm-inference-bench 服务  ──/metrics──▶  Prometheus  ──▶  Grafana
   (api/server.py)                        :9090           :3000
```

- `deploy/prometheus.yml` 已配置抓取目标
- Grafana 数据源与仪表盘通过 `provisioning/` 自动装载，无需手工导入
- 仪表盘 JSON 由 `scripts/gen_grafana_dashboard.py` 生成（15 个面板）

### 3.2 让容器内的 Prometheus 抓到宿主机服务

容器内访问宿主机需用 `host.docker.internal`（Docker Desktop）
或宿主机的桥接网关 IP（Linux）：

```yaml
# deploy/prometheus.yml
scrape_configs:
  - job_name: llmbench
    static_configs:
      - targets: ["host.docker.internal:8090"]   # Linux 下换成网关 IP
```

Linux 上同时需要在 compose 里为 prometheus 服务加：

```yaml
extra_hosts:
  - "host.docker.internal:host-gateway"
```

### 3.3 重新生成 Grafana 面板

改完指标或想调整面板布局后：

```bash
python scripts/gen_grafana_dashboard.py
# → deploy/grafana/dashboards/llm-inference-bench.json
docker compose restart grafana
```

---

## 四、服务化运行

### 4.1 直接启动

```bash
python scripts/serve.py --host 0.0.0.0 --port 8090
```

开发时可加 `--reload` 热重载。

### 4.2 systemd 托管

```ini
# /etc/systemd/system/llmbench.service
[Unit]
Description=LLM Inference Bench API
After=network.target

[Service]
Type=simple
User=bench
WorkingDirectory=/opt/llm-inference-bench
Environment="NO_PROXY=127.0.0.1,localhost"
Environment="VLLM_BASE_URL=http://127.0.0.1:8000/v1"
Environment="VLLM_MODEL=Qwen2.5-72B-Instruct"
ExecStart=/opt/llm-inference-bench/.venv/bin/python scripts/serve.py --port 8090
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now llmbench
sudo systemctl status llmbench
```

### 4.3 健康检查

```bash
curl -s http://127.0.0.1:8090/health
# {"status":"ok","time":"...","jobs_total":0}
```

K8s liveness/readiness 探针可直接打 `/health`。

---

## 五、API 使用示例

因为一次完整扫描要几分钟，接口是**异步任务模型**。

```bash
# 1. 探测引擎
curl -s -X POST http://127.0.0.1:8090/engines/check \
     -H "Content-Type: application/json" \
     -d '{"engines":["vllm"]}'

# 2. 提交压测任务 —— 立即返回 job_id
curl -s -X POST http://127.0.0.1:8090/bench \
     -H "Content-Type: application/json" \
     -d '{
       "engines": ["vllm"],
       "concurrency_levels": [1, 4, 16],
       "max_tokens": 256,
       "warmup_requests": 2,
       "enable_monitor": true,
       "quantization": "fp16"
     }'
# → {"job_id":"3b93e8d4e814","run_id":"run-20261008-131759","status":"pending","poll":"/jobs/3b93e8d4e814"}

# 3. 轮询进度
curl -s http://127.0.0.1:8090/jobs/3b93e8d4e814

# 4. 取结果与报告
curl -s http://127.0.0.1:8090/jobs/3b93e8d4e814/result
curl -s http://127.0.0.1:8090/jobs/3b93e8d4e814/report -o report.html

# 5. 量化横向对比（同一 run 目录下的多份结果）
curl -s -X POST http://127.0.0.1:8090/compare \
     -H "Content-Type: application/json" \
     -d '{"run_id":"quant-72b","output_dir":"data/results"}'
```

交互式 API 文档：`http://<host>:8090/docs`

---

## 六、隔离内网注意事项

私有化交付现场常为物理隔离环境，以下几点必须提前准备：

### 6.1 离线安装依赖

在**有网机器**上打包 wheel：

```bash
pip download -r requirements.txt -d wheels/ \
    --platform manylinux2014_x86_64 --python-version 311 \
    --only-binary=:all:
```

拷进内网后：

```bash
pip install --no-index --find-links=wheels/ -r requirements.txt
```

### 6.2 报告无需外部资源

生成的 HTML 报告是**自包含**的：内联 SVG、无 CDN、无 `<script>`、无外部字体。
可直接拷到任意机器打开，包括完全隔离的环境。

这一点在交付时很关键 —— 带外部依赖的报告在内网打开就是白屏。

### 6.3 镜像离线导入

```bash
# 有网机器
docker save prom/prometheus:latest grafana/grafana:latest -o monitor-stack.tar
# 内网机器
docker load -i monitor-stack.tar
```

---

## 七、产物与数据管理

### 7.1 目录结构

```
data/results/
├── run-20261008-131759/                    # 每次运行一个目录
│   ├── vllm__Qwen2.5-72B-Instruct__fp16.json   # 完整结果
│   ├── vllm__Qwen2.5-72B-Instruct__fp16.csv    # 便于 Excel 分析
│   ├── report.md
│   └── report.html
└── quant-72b/                              # 多次运行可指定同一 run-id 以便对比
    ├── vllm__Qwen2.5-72B-Instruct__fp16.json
    └── vllm__Qwen2.5-72B-AWQ__awq.json
```

文件名格式：`<引擎>__<模型>__<量化>.json`（量化为 unknown 时省略该段）。
**这个命名规则是为了防止多配置压测互相覆盖。**

### 7.2 清理策略

结果文件体积很小（单次扫描约 10~30 KB），但仍建议：

```bash
# 列出所有运行
ls -lt data/results/

# 删除 30 天前的运行目录
find data/results -maxdepth 1 -type d -name "run-*" -mtime +30 -exec rm -rf {} +
```

**报告可以从结果重新生成**，所以清理时可以放心保留 JSON、删掉报告：

```bash
# 之后随时重建
python scripts/run_bench.py --report-only --from-results data/results/run-20261008-131759
```

### 7.3 结果归档建议

交付时建议按「客户 / 机型 / 模型」组织归档，并保留环境元信息
（结果 JSON 里已含 `environment` 字段：Python 版本、平台、GPU 型号与驱动）。
同一个客户复测时，环境信息可用于判断结果是否可比。

---

## 八、故障排查

| 症状 | 原因 | 处理 |
| --- | --- | --- |
| `502 upstream connect failed` | 系统代理劫持 localhost | 设 `NO_PROXY=127.0.0.1,localhost` |
| 全部请求失败，退出码非 0 | 引擎不可达 / 模型名不符 | 先跑 `--check`，确认模型名写法 |
| `model not found` | 模型名与引擎加载的不一致 | `--check` 会列出引擎实际加载的模型 |
| 显存数据全为空 | 无 `nvidia-smi` 或非 NVIDIA | 正常降级；确认容器内有 GPU 直通 |
| 容器内 `nvidia-smi` 不可用 | 未加 `--gpus all` | compose 中补 GPU 配置 |
| Grafana 面板无数据 | Prometheus 抓不到目标 | 检查 `prometheus.yml` 的 targets 与网络可达性 |
| Prometheus P99 长期不变 | 直方图累计，混入历史数据 | 重启服务以归零 |
| 端口被占用 | 上次进程未退出 | `netstat -ano \| grep :8090` 后按 PID 结束 |
| 吞吐明显低于预期 | 客户端与引擎同卡 / 客户端被其他任务抢占 | 分离压测客户端 |
| 报告 HTML 白屏 | 被改成了引用外部资源 | 保持自包含（测试有断言守着） |

### 8.1 查看详细日志

```bash
LOG_LEVEL=DEBUG python scripts/run_bench.py --engine vllm --levels 1 --verbose
```

---

## 九、容量规划建议

压测结果的用法不是「看最大吞吐」，而是：

1. **找饱和拐点** —— 吞吐曲线上的峰值点，这是理论容量上限
2. **打折留余量** —— 生产建议跑在拐点负载的 **60~70%**，留出突发余量
3. **看尾延迟是否可接受** —— 在选定的并发下，P99 是否满足 SLA
4. **若拐点处 P99 已超标** —— 说明瓶颈不在并发数，而在单请求生成速度
   （考虑量化、张量并行、投机解码等手段）

示例读数：

> 吞吐在并发 32 达到峰值 1911 tokens/s，并发 64 时回落 —— 拐点在 32。
> 生产建议并发控制在 19~22（约 60~70%）。
> 该点 TTFT 均值约 93 ms，P99 约 958 ms，需确认 P99 是否满足业务 SLA。

`P99 ≈ 958 ms` 明显高于 TTFT 均值，说明存在排队长尾 ——
如果业务对尾延迟敏感，应进一步降低并发或增大 batch 调优空间。

---

## 十、升级与版本

```bash
git pull
pip install -r requirements.txt
python -m pytest tests -q        # 升级后先跑测试
```

改动内容见 `CHANGELOG.md`。涉及指标口径变更时会显式标注，
因为这会使得新旧结果不可直接比较。
