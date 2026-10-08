#!/usr/bin/env python
"""压测命令行入口。

用法::

    # 单引擎
    python scripts/run_bench.py --engine vllm

    # 多引擎对比 + 自动出报告
    python scripts/run_bench.py --engine vllm,ollama --report

    # 自定义并发梯度与生成长度
    python scripts/run_bench.py --engine vllm --levels 1,4,16,64 --max-tokens 128

    # 只做健康检查，不压测
    python scripts/run_bench.py --engine vllm --check

    # 基于已有结果重新出报告（不重新压测）
    python scripts/run_bench.py --from-results data/results/run-20260518-143002 --report-only
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# 允许以 `python scripts/run_bench.py` 直接运行
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench.runner import LoadRunner  # noqa: E402
from config.settings import BENCH, LOG_LEVEL, MONITOR, EngineTarget, default_engines  # noqa: E402
from engines.client import UnifiedLLMClient  # noqa: E402
from metrics.store import (  # noqa: E402
    build_result,
    load_result,
    new_run_id,
    save_result,
)
from monitor.quant import QuantGroup, compare  # noqa: E402
from report import generate_reports  # noqa: E402

logger = logging.getLogger("run_bench")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    engines = default_engines()
    parser = argparse.ArgumentParser(
        description="LLM 推理服务并发压测与多引擎对比",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--engine",
        default="vllm",
        help=f"被测引擎，逗号分隔。可选：{', '.join(engines)}",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="覆盖被测服务的接入地址（仅单引擎时有效）",
    )
    parser.add_argument("--model", default=None, help="覆盖模型名（仅单引擎时有效）")
    parser.add_argument(
        "--quant",
        default=None,
        help="覆盖量化等级标注，用于报告分组，如 fp16 / int8 / awq",
    )
    parser.add_argument(
        "--levels",
        default=None,
        help="并发梯度，逗号分隔，如 1,2,4,8,16,32",
    )
    parser.add_argument("--max-tokens", type=int, default=None, help="单请求最大生成 token 数")
    parser.add_argument("--timeout", type=float, default=None, help="单请求超时（秒）")
    parser.add_argument("--warmup", type=int, default=None, help="每档位预热请求数")
    parser.add_argument(
        "--no-monitor", action="store_true", help="关闭 GPU 显存监控"
    )
    parser.add_argument(
        "--no-stream", action="store_true", help="使用非流式请求（将拿不到 TTFT/TPOT）"
    )
    parser.add_argument("--report", action="store_true", help="压测结束后生成报告")
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="不压测，直接基于 --from-results 指向的目录重新生成报告",
    )
    parser.add_argument(
        "--from-results",
        default=None,
        help="已有结果目录（配合 --report-only）",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="结果输出目录，默认 data/results",
    )
    parser.add_argument(
        "--run-id", default=None, help="指定 run id，默认按时间自动生成"
    )
    parser.add_argument(
        "--check", action="store_true", help="只做健康检查与模型可用性探测"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="输出调试日志")
    return parser.parse_args(argv)


def setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else getattr(logging, LOG_LEVEL, logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def resolve_targets(args: argparse.Namespace) -> list[EngineTarget]:
    """根据命令行参数解析出待测目标列表。"""
    available = default_engines()
    names = [n.strip() for n in args.engine.split(",") if n.strip()]
    if not names:
        raise SystemExit("--engine 不能为空")

    unknown = [n for n in names if n not in available]
    if unknown:
        raise SystemExit(
            f"未知引擎：{', '.join(unknown)}。可选：{', '.join(available)}"
        )

    targets = [available[n] for n in names]

    # 覆盖参数只对单引擎有意义：多引擎时两者会串味
    if len(targets) == 1:
        t = targets[0]
        if args.base_url:
            t.base_url = args.base_url
        if args.model:
            t.model = args.model
        if args.quant:
            t.quantization = args.quant
    elif args.base_url or args.model or args.quant:
        logger.warning("多引擎模式下 --base-url / --model / --quant 被忽略")

    return targets


def apply_overrides(args: argparse.Namespace) -> None:
    """把命令行参数写回全局配置。"""
    if args.levels:
        try:
            BENCH.concurrency_levels = [
                int(x) for x in args.levels.split(",") if x.strip()
            ]
        except ValueError as exc:
            raise SystemExit(f"--levels 格式错误：{exc}") from exc
    if args.max_tokens is not None:
        BENCH.max_tokens = args.max_tokens
    if args.timeout is not None:
        BENCH.timeout = args.timeout
    if args.warmup is not None:
        BENCH.warmup_requests = args.warmup
    if args.no_stream:
        BENCH.stream = False
    if args.output_dir:
        BENCH.output_dir = Path(args.output_dir)
    if args.no_monitor:
        MONITOR.enabled = False


def do_check(targets: list[EngineTarget]) -> int:
    """健康检查。返回非零表示有引擎不可用。"""
    print("\n=== 引擎健康检查 ===\n")
    bad = 0
    for t in targets:
        client = UnifiedLLMClient(t, timeout=min(BENCH.timeout, 10.0))
        try:
            health = client.health()
        finally:
            client.close()

        if health.get("ok"):
            loaded = health.get("model_loaded")
            flag = "OK " if loaded else "WARN"
            if not loaded:
                bad += 1
            print(f"[{flag}] {t.name:8s} {t.normalized_base_url()}")
            print(f"       目标模型: {t.model}")
            print(f"       可用模型: {', '.join(health.get('models') or []) or '（空）'}")
            if not loaded:
                print(f"       ⚠️ 目标模型不在可用列表中，压测大概率返回 404")
        else:
            bad += 1
            print(f"[FAIL] {t.name:8s} {t.normalized_base_url()}")
            print(f"       错误: {health.get('error')}")
    print()
    return 1 if bad else 0


def run_one(
    target: EngineTarget, run_id: str, output_dir: Path
) -> dict:
    """对单个目标跑一轮完整压测并落盘，返回结果字典。"""
    print(f"\n{'=' * 62}")
    print(f"  压测引擎：{target.name}  |  模型：{target.model}")
    print(f"  接入地址：{target.normalized_base_url()}  |  量化标注：{target.quantization}")
    print(f"  并发梯度：{BENCH.concurrency_levels}  |  最大生成：{BENCH.max_tokens}")
    print(f"{'=' * 62}\n")

    runner = LoadRunner(target, BENCH, monitor_config=MONITOR if MONITOR.enabled else None)
    health = runner.health()
    if not health.get("ok"):
        print(f"  ⚠️ 健康检查未通过：{health.get('error')}")
        print("  仍会尝试压测，但失败请求会被记为 error。\n")

    import time

    started = time.perf_counter()

    def on_stage(stage) -> None:
        """每档完成即打印一行进度，长跑时能看见动静。"""
        tp = f"{stage.throughput_output:7.1f}" if stage.throughput_output else "      -"
        ttft = f"{stage.ttft_mean * 1000:7.1f}" if stage.ttft_mean else "      -"
        p99 = f"{stage.latency_p99 * 1000:8.1f}" if stage.latency_p99 else "       -"
        flag = "!" if stage.unstable else " "
        print(
            f"  {flag} 并发 {stage.concurrency:>3}  |  "
            f"成功 {stage.success_requests:>3}/{stage.total_requests:<3}  |  "
            f"吞吐 {tp} tokens/s  |  TTFT {ttft} ms  |  P99 {p99} ms"
        )

    stages = runner.run_sweep(on_stage=on_stage)
    duration = time.perf_counter() - started

    vram_curve = runner.vram_curve()
    result = build_result(
        target=target,
        bench=BENCH,
        stages=stages,
        vram_curve=vram_curve,
        health=health,
        run_id=run_id,
        duration=duration,
    )

    # 全档位零成功 = 压测实质失败，必须显式报警。
    # 否则调用方（CI / 人工）会拿到一份「成功生成但全是 N/A」的报告，
    # 误以为测试通过。
    total_ok = sum(s.success_requests for s in stages)
    if total_ok == 0:
        first_error = next((s.errors[0] for s in stages if s.errors), None)
        print(f"\n  ❌ 全部 {sum(s.total_requests for s in stages)} 个请求均失败。")
        if first_error:
            print(f"     首个错误：{first_error}")
        if not health.get("ok"):
            print(f"     健康检查失败：{health.get('error')}")
        elif not health.get("model_loaded"):
            print(
                f"     ⚠️ 目标模型 `{target.model}` 不在服务的可用模型列表中："
                f"{', '.join(health.get('models') or []) or '（空）'}"
            )
            print("     请检查 --model 是否与引擎实际加载的模型名一致。")
        print("     结果仍会落盘，但报告中的数据不可用。\n")

    paths = save_result(result, output_dir, run_id)
    print(f"\n  用时 {duration:.1f}s，结果已保存：")
    print(f"    JSON: {paths['json']}")
    print(f"    CSV : {paths['csv']}")

    if vram_curve.get("available"):
        print(f"  显存峰值：{vram_curve.get('peak_memory_used_mib')} MiB"
              f"（{vram_curve.get('gpu_count')} 张卡）")
    return result


def report_only(source_dir: Path, output_dir: Path) -> int:
    """基于已有结果重新生成报告。"""
    if not source_dir.exists():
        raise SystemExit(f"结果目录不存在：{source_dir}")

    json_files = sorted(source_dir.glob("*.json"))
    if not json_files:
        raise SystemExit(f"目录下没有结果 JSON：{source_dir}")

    results = [load_result(p) for p in json_files]
    groups = [
        QuantGroup(
            label=f"{r.get('model')} @ {r.get('quantization')}",
            engine=r.get("engine", "?"),
            model=r.get("model", "?"),
            quantization=r.get("quantization", "unknown"),
            stages=[],
            vram_peak_mib=r.get("vram_peak_mib"),
            extra={"is_baseline": r.get("quantization") in {"fp16", "bf16"}},
        )
        for r in results
    ]
    from monitor.quant import _stage_from_dict

    for g, r in zip(groups, results):
        g.stages = [_stage_from_dict(s) for s in r.get("stages", [])]

    matrix = compare(groups) if len(groups) > 1 else None
    paths = generate_reports(
        results,
        output_dir,
        quant_matrix=matrix,
    )
    print(f"\n报告已生成：\n  Markdown: {paths['markdown']}\n  HTML    : {paths['html']}")
    return 0


def build_quant_matrix(results: list[dict]) -> dict | None:
    """从多份压测结果构造量化对比矩阵。

    只有当各结果代表的配置**真的可比较**时才有意义：
    同引擎或同模型家族、且量化等级不全相同。
    """
    if len(results) < 2:
        return None

    quants = {r.get("quantization", "unknown") for r in results}
    if len(quants) < 2:
        logger.info("所有配置量化等级相同（%s），跳过量化的横向对比", quants.pop())
        return None

    from monitor.quant import _stage_from_dict

    groups = [
        QuantGroup(
            label=f"{r.get('model')} @ {r.get('quantization')}",
            engine=r.get("engine", "?"),
            model=r.get("model", "?"),
            quantization=r.get("quantization", "unknown"),
            stages=[_stage_from_dict(s) for s in r.get("stages", [])],
            vram_peak_mib=r.get("vram_peak_mib"),
            extra={"is_baseline": r.get("quantization", "").lower() in {"fp16", "bf16"}},
        )
        for r in results
    ]
    return compare(groups)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    setup_logging(args.verbose)
    apply_overrides(args)

    output_dir = Path(BENCH.output_dir)

    # 仅重新出报告
    if args.report_only:
        if not args.from_results:
            raise SystemExit("--report-only 需要配合 --from-results <目录>")
        return report_only(Path(args.from_results), Path(args.from_results))

    targets = resolve_targets(args)

    if args.check:
        return do_check(targets)

    run_id = args.run_id or new_run_id()
    print(f"\n压测批次 run_id = {run_id}")

    results = []
    for target in targets:
        try:
            results.append(run_one(target, run_id, output_dir))
        except KeyboardInterrupt:
            print("\n已被用户中断。")
            return 130
        except Exception as exc:  # 单个引擎失败不应中断整个批次
            logger.error("引擎 %s 压测失败：%s", target.name, exc, exc_info=args.verbose)
            print(f"\n  ❌ 引擎 {target.name} 压测出错：{exc}")

    if not results:
        print("\n没有任何成功的压测结果。")
        return 1

    # 汇总
    print(f"\n{'=' * 62}\n  汇总\n{'=' * 62}")
    usable = 0
    for r in results:
        s = r.get("summary") or {}
        tp = s.get("best_throughput")
        if tp:
            usable += 1
            tp_s = f"{tp:.1f} tokens/s @ 并发 {s.get('best_throughput_concurrency')}"
        else:
            tp_s = "无数据（全部请求失败）"
        print(f"  {r.get('engine'):8s} {r.get('quantization'):8s} 峰值吞吐 {tp_s}")

    if args.report:
        matrix = build_quant_matrix(results)
        if matrix:
            print(f"\n  量化对比：{matrix.get('conclusion')}")
        paths = generate_reports(results, output_dir / run_id, quant_matrix=matrix)
        print(f"\n报告已生成：")
        print(f"    Markdown: {paths['markdown']}")
        print(f"    HTML    : {paths['html']}")

    # 全部配置都没跑出数据 -> 非零退出，让 CI / 脚本能感知失败
    return 0 if usable else 2


if __name__ == "__main__":
    raise SystemExit(main())
