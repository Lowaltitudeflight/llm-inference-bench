# 贡献指南

感谢有兴趣参与改进。这个项目的核心场景是**私有化交付现场的性能验证**，
所以判据很明确：改动应该让「测得更准」「报告更能说明问题」「部署更容易」三者之一变好。

## 开发环境

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt -r requirements-dev.txt

python -m pytest tests -q          # 应当 302 passed
```

无需 GPU 也能完成全部开发与测试 —— `scripts/mock_engine.py` 提供了带排队
延迟模拟的 OpenAI 兼容假引擎。

## 提交规范

采用 [Conventional Commits](https://www.conventionalcommits.org/zh-hans/)：

```
feat(monitor): 支持按 MIG 实例拆分显存统计
fix(report): 修正单 token 响应时 TPOT 除零
docs(readme): 补充 Ollama 量化模型名对照表
test(stats): 增加分位数奇偶长度用例
refactor(bench): 抽出阶段计时器
```

类型：`feat` / `fix` / `docs` / `test` / `refactor` / `perf` / `build` / `chore`

**一次提交只做一件事。** 不要把「修 bug + 顺手重排版」混在一起 ——
这个项目的 `CHANGELOG` 会逐条列出真实缺陷，提交粒度太粗会让溯源变得困难。

## 改动时的注意事项

### 指标口径不能想当然

以下几条是这个项目踩过的坑，改代码前请先读一遍：

1. **TTFT 必须基于流式首个内容分片**。收到 HTTP 200 不代表模型开始出字了，
   中间还隔着排队和 prefill。任何「用总耗时近似首包」的改动都是错的。

2. **TPOT 除以 `output_tokens - 1`**。第一个 token 的等待时间已计入 TTFT，
   重复计算会系统性低估单 token 生成速度。

3. **多卡显存峰值必须按时序求和**。取各卡峰值的最大值在 8 卡场景下会少报 8 倍。
   `test_multi_gpu_peak_not_underreported` 专门守这条。

4. **每个并发请求要独立客户端**。复用连接池会把「并发」变成「排队」，
   测出来的是 HTTP 客户端的队列深度，不是推理引擎的并发能力。

5. **健康探测与模型名匹配要宽松**。真实环境里模型名写法五花八门
   （`qwen2.5` vs `qwen2.5:7b` vs `/models/Qwen2.5-7B-Instruct`），
   严格相等匹配会把正常部署判成失败。现有实现是双向前缀匹配。

### Prometheus 相关

- 直方图分桶不要用默认值。默认桶对 LLM 的 TTFT/TPOT 量级过粗，
  P99 会落在桶边界上，分位数估计严重失真。改动分桶需同步更新
  `deploy/grafana/dashboards/*.json` 里的查询。
- `Info.info()` 每次调用**整体替换**标签集合。写多个字段必须一次性传入完整字典。

### 报告相关

- HTML 报告必须保持**自包含**：不允许出现外部 URL、CDN、`<script>`。
  客户内网常为物理隔离环境，任何外部依赖都会导致白屏。
  测试里有断言守着这一点。
- 图表用 `report/svgchart.py` 手写 SVG，不要引入绘图库 —— 引入了就不再自包含了。
- 报告措辞只陈述可验证的事实。不要为了让结论「好看」而挑数据。

### 测试

- 新增行为请配套测试。当前覆盖 302 个用例，请保持全绿。
- 测试不要依赖真实网络、真实 GPU、真实模型。
  需要 HTTP 交互时参考 `tests/test_runner.py` 的 `MockTransport` 写法。
- 不要用 `time.sleep()` 来制造顺序，用 `os.utime()` 控制文件修改时间。

## 提 PR 前

```bash
python -m pytest tests -q                    # 单测全绿
python scripts/mock_engine.py --port 8102 &  # 起 mock 引擎
python scripts/run_bench.py --engine vllm --base-url http://127.0.0.1:8102 \
    --model mock-72b --levels 1,4 --max-tokens 32 --output-dir /tmp/smoke
```

确认压测能跑通、报告能生成。如果改了报告或指标逻辑，
请把新生成的报告截图或片段贴进 PR 描述 —— 这类改动看代码不如看产物直观。

> Windows Git Bash 下记得 `export NO_PROXY="127.0.0.1,localhost"`，
> 否则本地请求可能被系统代理拦成 `502 upstream connect failed`。

## 不接受的改动

- 伪造或美化压测数据（包括「选择性展示最优档位」这类隐性粉饰）
- 引入会破坏 HTML 报告自包含性的依赖
- 为通过测试而修改测试断言，而非修代码

## License

贡献的代码将以 MIT 协议发布，见 `LICENSE`。
