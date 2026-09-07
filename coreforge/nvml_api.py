"""Мониторинг через NVML (nvml.dll из драйвера NVIDIA)."""

from __future__ import annotations

import ctypes
from ctypes import POINTER, c_char, c_int, c_uint, c_ulonglong, create_string_buffer
from dataclasses import dataclass


NVML_SUCCESS = 0
NVML_TEMPERATURE_GPU = 0
NVML_CLOCK_GRAPHICS = 0
NVML_CLOCK_SM = 1
NVML_CLOCK_MEM = 2


class NvmlError(RuntimeError):
    pass


class NvmlUtilization(ctypes.Structure):
    _fields_ = [("gpu", c_uint), ("memory", c_uint)]


class NvmlMemory(ctypes.Structure):
    _fields_ = [("total", c_ulonglong), ("free", c_ulonglong), ("used", c_ulonglong)]


@dataclass
class GpuSample:
    temp: int = 0
    temp_hotspot: int = 0
    temp_vram: int = 0
    power_w: float = 0.0
    gpu_util: int = 0
    mem_util: int = 0
    vram_used: int = 0
    vram_total: int = 0
    clock_graphics: int = 0
    clock_sm: int = 0
    clock_mem: int = 0
    voltage_v: float = 0.0
    fan: int = 0
    ok: bool = False
    error: str = ""


class Nvml:
    def __init__(self) -> None:
        self.lib: ctypes.WinDLL | None = None
        self.handle = ctypes.c_void_p()
        self.available = False
        self.error = ""
        self._extra = None
        try:
            self.lib = self._load()
            self._bind()
            if self.lib.nvmlInit_v2() != NVML_SUCCESS:
                raise NvmlError("nvmlInit_v2")
            if self.lib.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(self.handle)) != NVML_SUCCESS:
                raise NvmlError("nvmlDeviceGetHandleByIndex")
            self.available = True
        except Exception as exc:
            self.available = False
            self.error = str(exc)
            return
        try:
            from coreforge.nvapi_temps import ExtraTemps

            self._extra = ExtraTemps(self.lib, self.handle)
        except Exception:
            self._extra = None

    def _load(self) -> ctypes.WinDLL:
        for path in (
            "nvml.dll",
            r"C:\Windows\System32\nvml.dll",
            r"C:\Program Files\NVIDIA Corporation\NVSMI\nvml.dll",
        ):
            try:
                return ctypes.WinDLL(path)
            except OSError:
                continue
        raise NvmlError("nvml.dll не найдена")

    def _bind(self) -> None:
        L = self.lib
        assert L is not None
        L.nvmlInit_v2.restype = c_int
        L.nvmlShutdown.restype = c_int
        L.nvmlDeviceGetHandleByIndex_v2.argtypes = [c_uint, POINTER(ctypes.c_void_p)]
        L.nvmlDeviceGetHandleByIndex_v2.restype = c_int
        L.nvmlDeviceGetName.argtypes = [ctypes.c_void_p, POINTER(c_char), c_uint]
        L.nvmlDeviceGetName.restype = c_int
        L.nvmlDeviceGetTemperature.argtypes = [ctypes.c_void_p, c_uint, POINTER(c_uint)]
        L.nvmlDeviceGetTemperature.restype = c_int
        L.nvmlDeviceGetPowerUsage.argtypes = [ctypes.c_void_p, POINTER(c_uint)]
        L.nvmlDeviceGetPowerUsage.restype = c_int
        L.nvmlDeviceGetUtilizationRates.argtypes = [ctypes.c_void_p, POINTER(NvmlUtilization)]
        L.nvmlDeviceGetUtilizationRates.restype = c_int
        L.nvmlDeviceGetMemoryInfo.argtypes = [ctypes.c_void_p, POINTER(NvmlMemory)]
        L.nvmlDeviceGetMemoryInfo.restype = c_int
        L.nvmlDeviceGetClockInfo.argtypes = [ctypes.c_void_p, c_uint, POINTER(c_uint)]
        L.nvmlDeviceGetClockInfo.restype = c_int
        L.nvmlDeviceGetFanSpeed.argtypes = [ctypes.c_void_p, POINTER(c_uint)]
        L.nvmlDeviceGetFanSpeed.restype = c_int

    def name(self) -> str:
        if not self.available or not self.lib:
            return ""
        buf = create_string_buffer(96)
        if self.lib.nvmlDeviceGetName(self.handle, buf, 96) == NVML_SUCCESS:
            return buf.value.decode("utf-8", errors="replace")
        return ""

    def sample(self) -> GpuSample:
        out = GpuSample()
        if not self.available or not self.lib:
            out.error = self.error or "NVML недоступен"
            return out
        temp = c_uint()
        if self.lib.nvmlDeviceGetTemperature(self.handle, NVML_TEMPERATURE_GPU, ctypes.byref(temp)) == 0:
            out.temp = int(temp.value)
        power = c_uint()
        if self.lib.nvmlDeviceGetPowerUsage(self.handle, ctypes.byref(power)) == 0:
            out.power_w = power.value / 1000.0
        util = NvmlUtilization()
        if self.lib.nvmlDeviceGetUtilizationRates(self.handle, ctypes.byref(util)) == 0:
            out.gpu_util = int(util.gpu)
            out.mem_util = int(util.memory)
        mem = NvmlMemory()
        if self.lib.nvmlDeviceGetMemoryInfo(self.handle, ctypes.byref(mem)) == 0:
            out.vram_used = int(mem.used)
            out.vram_total = int(mem.total)
        clk = c_uint()
        if self.lib.nvmlDeviceGetClockInfo(self.handle, NVML_CLOCK_GRAPHICS, ctypes.byref(clk)) == 0:
            out.clock_graphics = int(clk.value)
        if self.lib.nvmlDeviceGetClockInfo(self.handle, NVML_CLOCK_SM, ctypes.byref(clk)) == 0:
            out.clock_sm = int(clk.value)
        if self.lib.nvmlDeviceGetClockInfo(self.handle, NVML_CLOCK_MEM, ctypes.byref(clk)) == 0:
            out.clock_mem = int(clk.value)
        fan = c_uint()
        if self.lib.nvmlDeviceGetFanSpeed(self.handle, ctypes.byref(fan)) == 0:
            out.fan = int(fan.value)
        if self._extra is not None:
            hot, mem = self._extra.nvapi_pair()
            if not hot or not mem:
                sh, sm = self._extra.smi_pair()
                hot = hot or sh
                mem = mem or sm
            out.temp_hotspot = hot
            out.temp_vram = mem or self._extra.nvml_memory()
            out.voltage_v = self._extra.voltage_v()
        out.ok = True
        return out

    def close(self) -> None:
        if self.available and self.lib:
            self.lib.nvmlShutdown()
            self.available = False
