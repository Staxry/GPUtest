"""Hotspot и температура памяти через NvAPI (как GPU-Z / LHM) и NVML field 82."""

from __future__ import annotations

import ctypes
from ctypes import CFUNCTYPE, POINTER, c_int, c_uint, c_void_p


NVAPI_OK = 0
NVAPI_INIT = 0x0150E828
NVAPI_ENUM_GPUS = 0xE5AC921F
NVAPI_THERMAL_SENSORS = 0x65FE3AAD
NVML_FI_DEV_MEMORY_TEMP = 82


class NvThermalSensors(ctypes.Structure):
    _pack_ = 8
    _fields_ = [
        ("Version", c_uint),
        ("Mask", c_uint),
        ("Reserved", c_int * 8),
        ("Temperatures", c_int * 32),
    ]


class NvmlFieldValue(ctypes.Structure):
    _fields_ = [
        ("fieldId", c_uint),
        ("scopeId", c_uint),
        ("timestamp", ctypes.c_int64),
        ("latencyUsec", c_uint),
        ("valueType", c_uint),
        ("nvmlReturn", c_uint),
        ("value", ctypes.c_uint64),
    ]


def _make_version(size: int, ver: int) -> int:
    return (ver << 16) | size


class ExtraTemps:
    def __init__(self, nvml_lib=None, nvml_handle=None) -> None:
        self._nvml = nvml_lib
        self._nvml_handle = nvml_handle
        self._gpu = None
        self._mask = 0
        self._sensors = None
        self._enum = None
        self._voltage = None
        self.ok = False
        try:
            self._init_nvapi()
        except Exception:
            self.ok = False

    def _init_nvapi(self) -> None:
        dll = ctypes.WinDLL("nvapi64.dll")
        query = dll.nvapi_QueryInterface
        query.restype = c_void_p
        query.argtypes = [c_uint]
        init_ptr = query(NVAPI_INIT)
        if not init_ptr:
            return
        init = CFUNCTYPE(c_int)(init_ptr)
        if init() != NVAPI_OK:
            return
        enum_ptr = query(NVAPI_ENUM_GPUS)
        sens_ptr = query(NVAPI_THERMAL_SENSORS)
        if not enum_ptr or not sens_ptr:
            return
        self._enum = CFUNCTYPE(c_int, POINTER(c_void_p), POINTER(c_int))(enum_ptr)
        self._sensors = CFUNCTYPE(c_int, c_void_p, POINTER(NvThermalSensors))(sens_ptr)
        handles = (c_void_p * 64)()
        count = c_int()
        if self._enum(handles, ctypes.byref(count)) != NVAPI_OK or count.value < 1:
            return
        self._gpu = handles[0]
        self._mask = self._probe_mask()
        self.ok = self._mask > 0
        volt_ptr = query(0xC16C7E2C)
        if volt_ptr:
            self._voltage = CFUNCTYPE(c_int, c_void_p, c_void_p)(volt_ptr)

    def _probe_mask(self) -> int:
        found = 0
        for bit in range(32):
            mask = 1 << bit
            if self._read(mask) is None:
                break
            found = mask * 2 - 1
        return found

    def _read(self, mask: int) -> NvThermalSensors | None:
        if not self._sensors or not self._gpu:
            return None
        data = NvThermalSensors()
        data.Version = _make_version(ctypes.sizeof(NvThermalSensors), 2)
        data.Mask = mask
        if self._sensors(self._gpu, ctypes.byref(data)) != NVAPI_OK:
            return None
        return data

    def nvapi_pair(self) -> tuple[int, int]:
        """(hotspot, vram) в °C, 0 если нет данных. RTX 40xx: idx 1 и 7, значение / 256."""
        if not self.ok:
            return 0, 0
        data = self._read(self._mask)
        if data is None:
            return 0, 0
        vals = [data.Temperatures[i] / 256.0 for i in range(32)]
        hot = vals[1] if vals[1] > 0 else 0.0
        mem = vals[7] if vals[7] > 0 else 0.0
        if mem <= 0:
            for idx in (9, 2, 8, 6):
                if vals[idx] > 20:
                    mem = vals[idx]
                    break
        return int(round(hot)), int(round(mem))

    def smi_pair(self) -> tuple[int, int]:
        import subprocess
        import time

        now = time.monotonic()
        cached = getattr(self, "_smi_cache", None)
        if cached and now - cached[0] < 1.2:
            return cached[1], cached[2]
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "-q", "-d", "TEMPERATURE"],
                text=True,
                encoding="utf-8",
                errors="replace",
                creationflags=0x08000000,
                timeout=2.5,
            )
        except Exception:
            return 0, 0
        hot = mem = 0
        for line in out.splitlines():
            if "Hot Spot" in line and ":" in line:
                hot = _parse_c(line)
            elif "Memory Current Temp" in line and ":" in line:
                mem = _parse_c(line)
        self._smi_cache = (now, hot, mem)
        return hot, mem

    def voltage_v(self) -> float:
        if not self._voltage or not self._gpu:
            return 0.0
        buf = (c_uint * 48)()
        buf[0] = _make_version(ctypes.sizeof(buf), 1)
        if self._voltage(self._gpu, ctypes.byref(buf)) != NVAPI_OK:
            return 0.0
        for raw in buf[2:16]:
            val = int(raw)
            if 400_000 <= val <= 2_000_000:
                return val / 1_000_000.0
            if 400 <= val <= 2000:
                return val / 1000.0
        return 0.0

    def nvml_memory(self) -> int:
        if not self._nvml or not self._nvml_handle:
            return 0
        try:
            if not hasattr(self._nvml, "nvmlDeviceGetFieldValues"):
                self._nvml.nvmlDeviceGetFieldValues.argtypes = [c_void_p, c_int, POINTER(NvmlFieldValue)]
                self._nvml.nvmlDeviceGetFieldValues.restype = c_int
            field = NvmlFieldValue()
            field.fieldId = NVML_FI_DEV_MEMORY_TEMP
            rc = self._nvml.nvmlDeviceGetFieldValues(self._nvml_handle, 1, ctypes.byref(field))
            if rc != 0 or field.nvmlReturn != 0:
                return 0
            raw = int(field.value)
            if 10 < raw < 130:
                return raw
            # иногда значение лежит в младших 32 битах другого типа
            lo = raw & 0xFFFFFFFF
            if 10 < lo < 130:
                return lo
        except Exception:
            return 0
        return 0


def _parse_c(line: str) -> int:
    tail = line.split(":")[-1].strip().split()[0]
    try:
        return int(float(tail))
    except ValueError:
        return 0
