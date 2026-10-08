# 文档索引

| 文档 | 内容 | 适合谁看 |
| --- | --- | --- |
| [README](../README.md) | 项目定位、架构图、核心设计决策、快速开始 | 第一次接触项目 |
| [压测方法论](methodology.md) | TTFT/TPOT/吞吐/分位数的精确定义与推导、曲线解读、常见坑点速查 | 想知道「这些数是怎么算出来的」 |
| [部署手册](deployment.md) | 裸机 / systemd / 容器化部署、隔离内网注意事项、容量规划、故障排查 | 要在真实环境落地 |
| [CHANGELOG](../CHANGELOG.md) | 逐版本变更记录，含每个修复的缺陷说明 | 追踪版本变化 |
| [CONTRIBUTING](../CONTRIBUTING.md) | 贡献流程与 5 条指标口径纪律 | 准备提 PR |

---

## 按问题找文档

**「TTFT 是怎么测的？为什么不能直接用响应时间？」**
→ [methodology.md · 关键指标的精确定义](methodology.md#二关键指标的精确定义)

**「多卡显存占用为什么不能各卡取最大值？」**
→ [methodology.md · 显存测量](methodology.md#七显存测量) / [README · 核心设计决策 3](../README.md#3-显存峰值按时序求和而非取单卡最大值)

**「报告能不能在隔离内网打开？」**
→ [deployment.md · 隔离内网注意事项](deployment.md#六隔离内网注意事项)

**「结果曲线怎么解读？多少并发是安全的？」**
→ [methodology.md · 并发梯度扫描](methodology.md#三并发梯度扫描) / [deployment.md · 容量规划建议](deployment.md#九容量规划建议)

**「Prometheus 的 P99 为什么一直不变？」**
→ [methodology.md · Prometheus 指标口径](methodology.md#八prometheus-指标口径)

**「Windows 下为什么请求报 502？」**
→ [methodology.md · 测量环境要求](methodology.md#52-代理问题windows-常见) / [deployment.md · 故障排查](deployment.md#八故障排查)

**「怎么选量化等级？」**
→ [methodology.md · 量化等级横向对比](methodology.md#六量化等级横向对比)

---

## 代码导航

按数据流读代码，比按目录结构读更快：

```
1. config/settings.py        ← 先看配置模型（EngineTarget / BenchConfig）
2. engines/client.py         ← 再看怎么发请求、怎么测 TTFT
3. bench/runner.py           ← 然后看并发扫描怎么调度
4. metrics/stats.py          ← 核心：单请求 → 阶段统计的聚合逻辑
5. monitor/gpu.py            ← 与压测并行的显存采样
6. monitor/quant.py          ← 多配置如何横向对比
7. report/generator.py       ← 最后看报告怎么装配（→ markdown.py / html.py）
```

想快速理解指标口径，直接看 `metrics/stats.py` 里 `RequestResult.tpot` 与
`percentile()` 的实现，以及 `monitor/gpu.py::GpuMonitor.curve()` 的峰值算法——
这三处是最容易出错、也最值得读的地方。

---

## 想看效果但不想装环境

`data/results/demo/` 里有一组用 mock 引擎跑出的示例产物：

- `report.html` —— 自包含 HTML 报告（下载后直接双击打开，无需联网）
- `report.md` —— Markdown 版报告
- `vllm__mock-72b__fp16.json` / `vllm__mock-72b-awq__awq.json` —— 原始结果数据
