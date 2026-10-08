"""GPU 监控测试。

核心要求：**监控模块在任何环境下都不能抛异常**。压测跑了几分钟，
不能因为 nvidia-smi 挂了把整个压测带崩。所以这里重点测降级路径。

标记 ``@pytest.mark.gpu`` 的测试只在真有 NVIDIA 卡时有意义。
"""

from __future__ import annotations

import pytest

from monitor.gpu import (
    GpuMonitor,
    GpuSample,
    _to_float,
    nvidia_smi_available,
    query_gpus,
)


class TestToFloat:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("1234", 1234.0),
            ("1234.5", 1234.5),
            (" 42 ", 42.0),
            ("N/A", None),
            ("[N/A]", None),
            ("[Not Supported]", None),
            ("", None),
            ("   ", None),
            ("unknown", None),
            ("81920 MiB", None),  # 带单位无法解析：返回 None 而不是抛错
        ],
    )
    def test_parsing(self, text, expected):
        assert _to_float(text) == expected


class TestGpuSample:
    def test_memory_percent(self):
        s = GpuSample(index=0, memory_total_mib=81920.0, memory_used_mib=40960.0)
        assert s.memory_used_percent == pytest.approx(50.0)

    def test_memory_percent_zero_total_is_none(self):
        """总量为 0 时不能除零。"""
        s = GpuSample(index=0, memory_total_mib=0.0, memory_used_mib=100.0)
        assert s.memory_used_percent is None

    def test_memory_percent_missing_total_is_none(self):
        s = GpuSample(index=0, memory_used_mib=100.0)
        assert s.memory_used_percent is None

    def test_to_dict_includes_percent(self):
        s = GpuSample(index=1, name="RTX 4090", memory_total_mib=24576.0, memory_used_mib=12288.0)
        d = s.to_dict()
        assert d["index"] == 1
        assert d["memory_used_percent"] == pytest.approx(50.0)


class TestQueryGpus:
    def test_never_raises(self):
        """无论有无 GPU、有无 nvidia-smi，都不能抛异常。"""
        result = query_gpus()
        assert isinstance(result, dict)
        assert "available" in result
        assert "gpus" in result
        assert "message" in result
        assert isinstance(result["gpus"], list)

    def test_unavailable_shape_is_consistent(self):
        """不可用时也必须返回完整结构，调用方无需判空。"""
        result = query_gpus()
        if not result["available"]:
            assert result["gpus"] == []
            assert isinstance(result["message"], str)
            assert result["message"]

    @pytest.mark.gpu
    def test_real_gpu_query(self):
        """有卡时应能解析出至少一张卡及其显存容量。"""
        result = query_gpus()
        assert result["available"] is True
        assert len(result["gpus"]) >= 1
        gpu = result["gpus"][0]
        assert gpu.index >= 0
        assert gpu.name
        assert gpu.memory_total_mib is not None
        assert gpu.memory_total_mib > 0


class TestNvidiaSmiAvailable:
    def test_returns_bool(self):
        """必须是 bool，不能是 None 或异常。"""
        assert isinstance(nvidia_smi_available(), bool)

    def test_consistent_with_query(self):
        """两个探测口径要一致，否则启动判定会和实际采样打架。"""
        if nvidia_smi_available():
            assert query_gpus()["available"] is True


class TestGpuMonitorDegradation:
    def test_disabled_monitor_does_not_start_thread(self):
        mon = GpuMonitor(interval=0.1, enabled=False)
        mon.start()
        assert mon.available is False
        assert mon.message == "监控已关闭"
        assert mon._thread is None
        mon.stop()

    def test_curve_on_disabled_monitor(self):
        """关闭监控时 curve() 返回 available=False 的空结构。"""
        mon = GpuMonitor(enabled=False)
        mon.start()
        curve = mon.curve()
        assert curve["available"] is False
        assert curve["gpus"] == []
        mon.stop()

    def test_summary_on_disabled_monitor(self):
        mon = GpuMonitor(enabled=False)
        mon.start()
        assert mon.summary() == {"available": False}
        mon.stop()

    def test_start_stop_is_idempotent(self):
        """重复 stop 不应报错。"""
        mon = GpuMonitor(enabled=False)
        mon.start()
        mon.stop()
        mon.stop()
        assert mon._thread is None

    def test_context_manager(self):
        with GpuMonitor(enabled=False) as mon:
            assert mon.available is False
        assert mon._thread is None

    def test_interval_is_clamped(self):
        """过小的采样间隔会打爆 CPU，应被夹到下限。"""
        mon = GpuMonitor(interval=0.0, enabled=False)
        assert mon.interval >= 0.05


class TestCurveAssembly:
    def _monitor_with(self, samples) -> GpuMonitor:
        """注入伪造样本，绕过真实 nvidia-smi。"""
        mon = GpuMonitor(interval=0.5, enabled=False)
        mon.available = True
        mon.message = "test"
        mon._samples = samples
        return mon

    def test_groups_by_gpu_index(self):
        samples = [
            GpuSample(index=0, name="A", memory_total_mib=81920, memory_used_mib=1000, util_percent=10, t=0.0),
            GpuSample(index=1, name="B", memory_total_mib=81920, memory_used_mib=2000, util_percent=20, t=0.0),
            GpuSample(index=0, name="A", memory_total_mib=81920, memory_used_mib=40000, util_percent=90, t=0.5),
            GpuSample(index=1, name="B", memory_total_mib=81920, memory_used_mib=5000, util_percent=30, t=0.5),
        ]
        curve = self._monitor_with(samples).curve()

        assert curve["available"] is True
        assert curve["gpu_count"] == 2
        by_index = {g["index"]: g for g in curve["gpus"]}
        assert by_index[0]["peak_memory_used_mib"] == 40000
        assert by_index[1]["peak_memory_used_mib"] == 5000
        assert by_index[0]["mean_util_percent"] == pytest.approx(50.0)
        assert by_index[0]["sample_count"] == 2

    def test_global_peak_is_sum_at_same_timestamp(self):
        """全局显存峰值是「同一时刻各卡之和」的最大值，不是各卡峰值相加。"""
        samples = [
            GpuSample(index=0, memory_total_mib=81920, memory_used_mib=10000, t=0.0),
            GpuSample(index=1, memory_total_mib=81920, memory_used_mib=10000, t=0.0),
            GpuSample(index=0, memory_total_mib=81920, memory_used_mib=30000, t=0.5),
            GpuSample(index=1, memory_total_mib=81920, memory_used_mib=5000, t=0.5),
        ]
        curve = self._monitor_with(samples).curve()
        # t=0 合计 20000；t=0.5 合计 35000 -> 全局峰值取 t=0.5 的合计
        assert curve["peak_memory_used_mib"] == 35000
        # 各卡自身峰值仍单独记录
        by_index = {g["index"]: g for g in curve["gpus"]}
        assert by_index[0]["peak_memory_used_mib"] == 30000
        assert by_index[1]["peak_memory_used_mib"] == 10000

    def test_multi_gpu_peak_not_underreported(self):
        """8 卡各用 60000 MiB 时，全局峰值必须是 480000 而非 60000。

        这是真实存在的坑：若按「各卡峰值的最大值」计算，多卡场景会
        把总占用低估 N 倍，报告里的成本结论就完全错了。
        """
        samples = [
            GpuSample(index=i, memory_total_mib=81920, memory_used_mib=60000, t=1.0)
            for i in range(8)
        ]
        curve = self._monitor_with(samples).curve()
        assert curve["peak_memory_used_mib"] == 480000

    def test_total_capacity_and_percent(self):
        samples = [
            GpuSample(index=0, memory_total_mib=81920, memory_used_mib=40960, t=0.0),
            GpuSample(index=1, memory_total_mib=81920, memory_used_mib=40960, t=0.0),
        ]
        curve = self._monitor_with(samples).curve()
        assert curve["total_memory_mib"] == 163840
        # 81920 / 163840 = 50%
        assert curve["peak_memory_used_percent"] == pytest.approx(50.0)

    def test_series_sorted_by_time(self):
        samples = [
            GpuSample(index=0, memory_total_mib=1000, memory_used_mib=300, t=2.0),
            GpuSample(index=0, memory_total_mib=1000, memory_used_mib=100, t=0.0),
            GpuSample(index=0, memory_total_mib=1000, memory_used_mib=200, t=1.0),
        ]
        curve = self._monitor_with(samples).curve()
        times = [row["t"] for row in curve["gpus"][0]["series"]]
        assert times == [0.0, 1.0, 2.0]

    def test_empty_samples_unavailable(self):
        curve = self._monitor_with([]).curve()
        assert curve["available"] is False
        assert curve["gpus"] == []

    def test_summary_shape(self):
        samples = [GpuSample(index=0, name="A", memory_total_mib=1000, memory_used_mib=500, t=0.0)]
        summary = self._monitor_with(samples).summary()
        assert summary["available"] is True
        assert summary["gpu_count"] == 1
        assert summary["peak_memory_used_mib"] == 500
        assert summary["per_gpu_peak_mib"] == {0: 500}

    def test_empty_memory_samples(self):
        """样本里没有显存字段时不能崩，峰值应为 None。"""
        samples = [GpuSample(index=0, name="A", util_percent=90.0, t=0.0)]
        curve = self._monitor_with(samples).curve()
        assert curve["peak_memory_used_mib"] is None
        assert curve["gpus"][0]["peak_memory_used_mib"] is None
        assert curve["gpus"][0]["mean_util_percent"] == pytest.approx(90.0)

    def test_samples_within_stage_peak(self):
        """模拟 runner 里的「档位内峰值」计算。"""
        mon = self._monitor_with(
            [
                GpuSample(index=0, memory_used_mib=100, t=0.0),
                GpuSample(index=1, memory_used_mib=100, t=0.0),
                GpuSample(index=0, memory_used_mib=900, t=1.0),
                GpuSample(index=1, memory_used_mib=900, t=1.0),
            ]
        )
        batch = mon.samples[2:]
        by_t: dict[float, float] = {}
        for s in batch:
            by_t[s.t] = by_t.get(s.t, 0.0) + (s.memory_used_mib or 0)
        assert max(by_t.values()) == 1800
