from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/benchmark_infographicvqa_decar_fit_runtime.py"


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "benchmark_infographicvqa_decar_fit_runtime", SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Device:
    def __init__(self, value: str) -> None:
        self.value = value


class _Cuda:
    def __init__(self) -> None:
        self.devices: list[_Device] = []
        self.empty_cache_calls = 0
        self.init_calls = 0
        self.initialized = False

    def init(self) -> None:
        self.init_calls += 1
        self.initialized = True

    def empty_cache(self) -> None:
        self.empty_cache_calls += 1

    def _record(self, device: _Device) -> None:
        assert self.initialized
        assert isinstance(device, _Device)
        assert device.value == "cuda:0"
        self.devices.append(device)

    def set_device(self, device: _Device) -> None:
        self._record(device)

    def synchronize(self, device: _Device) -> None:
        self._record(device)

    def reset_peak_memory_stats(self, device: _Device) -> None:
        self._record(device)

    def max_memory_allocated(self, device: _Device) -> int:
        self._record(device)
        return 123

    def get_device_name(self, device: _Device) -> str:
        self._record(device)
        return "NVIDIA H800"


class _Torch:
    def __init__(self) -> None:
        self.cuda = _Cuda()

    @staticmethod
    def device(value: str) -> _Device:
        return _Device(value)


def test_cuda_runtime_apis_receive_canonical_torch_device() -> None:
    module = _load_script()
    torch = _Torch()

    seconds, peak = module._timed_fit(torch, "cuda:0", lambda: object())
    accelerator = torch.cuda.get_device_name(module._runtime_device(torch, "cuda:0"))

    assert seconds >= 0.0
    assert peak == 123
    assert accelerator == "NVIDIA H800"
    assert torch.cuda.empty_cache_calls == 2
    assert torch.cuda.init_calls == 1
    assert len(torch.cuda.devices) == 6


def test_cpu_runtime_device_remains_cpu_string() -> None:
    module = _load_script()
    torch: Any = _Torch()

    assert module._runtime_device(torch, "cpu") == "cpu"
