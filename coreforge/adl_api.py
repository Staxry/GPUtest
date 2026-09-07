"""Мониторинг AMD через ADL (atiadlxx.dll из драйвера Adrenalin)."""

from __future__ import annotations

import ctypes
from ctypes import CFUNCTYPE, POINTER, c_int, c_void_p

from coreforge.nvml_api import GpuSample


ADL_OK = 0
PMLOG_GFXCLK = 1
PMLOG_MEMCLK = 2
PMLOG_FAN_RPM = 8
PMLOG_FAN_PCT = 9
PMLOG_TEMP_EDGE = 10
PMLOG_TEMP_MEM = 11
PMLOG_TEMP_HOTSPOT = 16
PMLOG_GFX_POWER = 29
PMLOG_GFX_ACTIVITY = 23


class ADLSingleSensor(ctypes.Structure):
    _fields_ = [("supported", c_int), ("value", c_int)]


class ADLPMLogDataOutput(ctypes.Structure):
    _fields_ = [("size", c_int), ("sensors", ADLSingleSensor * 256)]


class AdapterInfoX2(ctypes.Structure):
    _fields_ = [
        ("iSize", c_int),
        ("iAdapterIndex", c_int),
        ("strUDID", ctypes.c_char * 256),
        ("iBusNumber", c_int),
        ("iDeviceNumber", c_int),
        ("iFunctionNumber", c_int),
        ("iVendorID", c_int),
        ("strAdapterName", ctypes.c_char * 256),
        ("strDisplayName", ctypes.c_char * 256),
        ("iPresent", c_int),
        ("iExist", c_int),
        ("strDriverPath", ctypes.c_char * 256),
        ("strDriverPathExt", ctypes.c_char * 256),
        ("strPNPString", ctypes.c_char * 256),
        ("iOSDisplayIndex", c_int),
        ("iInfoMask", c_int),
        ("iInfoValue", c_int),
    ]


MALLOC = CFUNCTYPE(c_void_p, ctypes.c_int)


class Adl:
    def __init__(self) -> None:
        self.lib = None
        self.ctx = c_void_p()
        self.adapter = 0
        self.available = False
        self.error = ""
        self._name = ""
        self._keep: list = []
        try:
            self.lib = ctypes.WinDLL("atiadlxx.dll")
            self._bind()

            def _malloc(size: int) -> int:
                buf = (ctypes.c_ubyte * max(int(size), 1))()
                self._keep.append(buf)
                return ctypes.addressof(buf)

            self._cb = MALLOC(_malloc)
            if self.lib.ADL2_Main_Control_Create(self._cb, 1, ctypes.byref(self.ctx)) != ADL_OK:
                raise RuntimeError("ADL2_Main_Control_Create")
            n = c_int()
            if self.lib.ADL2_Adapter_NumberOfAdapters_Get(self.ctx, ctypes.byref(n)) != ADL_OK or n.value < 1:
                raise RuntimeError("нет адаптеров ADL")
            infos = (AdapterInfoX2 * n.value)()
            infos[0].iSize = ctypes.sizeof(AdapterInfoX2)
            if hasattr(self.lib, "ADL2_Adapter_AdapterInfo_Get"):
                self.lib.ADL2_Adapter_AdapterInfo_Get(self.ctx, infos, ctypes.sizeof(infos))
            for i in range(n.value):
                if infos[i].iPresent:
                    self.adapter = infos[i].iAdapterIndex
                    self._name = infos[i].strAdapterName.decode("utf-8", "replace")
                    break
            self.available = True
        except Exception as exc:
            self.available = False
            self.error = str(exc)

    def _bind(self) -> None:
        L = self.lib
        L.ADL2_Main_Control_Create.argtypes = [MALLOC, c_int, POINTER(c_void_p)]
        L.ADL2_Main_Control_Create.restype = c_int
        L.ADL2_Main_Control_Destroy.argtypes = [c_void_p]
        L.ADL2_Main_Control_Destroy.restype = c_int
        L.ADL2_Adapter_NumberOfAdapters_Get.argtypes = [c_void_p, POINTER(c_int)]
        L.ADL2_Adapter_NumberOfAdapters_Get.restype = c_int
        if hasattr(L, "ADL2_New_QueryPMLogData_Get"):
            L.ADL2_New_QueryPMLogData_Get.argtypes = [c_void_p, c_int, POINTER(ADLPMLogDataOutput)]
            L.ADL2_New_QueryPMLogData_Get.restype = c_int

    def name(self) -> str:
        return self._name

    def sample(self) -> GpuSample:
        out = GpuSample()
        if not self.available or not self.lib:
            out.error = self.error or "ADL недоступен"
            return out
        data = ADLPMLogDataOutput()
        data.size = ctypes.sizeof(ADLPMLogDataOutput)
        fn = getattr(self.lib, "ADL2_New_QueryPMLogData_Get", None)
        if fn is None or fn(self.ctx, self.adapter, ctypes.byref(data)) != ADL_OK:
            out.error = "PMLog недоступен"
            return out

        def sen(idx: int) -> int:
            if 0 <= idx < 256 and data.sensors[idx].supported:
                return int(data.sensors[idx].value)
            return 0

        out.temp = sen(PMLOG_TEMP_EDGE)
        out.temp_hotspot = sen(PMLOG_TEMP_HOTSPOT)
        out.temp_vram = sen(PMLOG_TEMP_MEM)
        out.power_w = float(sen(PMLOG_GFX_POWER))
        out.gpu_util = sen(PMLOG_GFX_ACTIVITY)
        out.clock_graphics = sen(PMLOG_GFXCLK)
        out.clock_sm = out.clock_graphics
        out.clock_mem = sen(PMLOG_MEMCLK)
        out.fan = sen(PMLOG_FAN_PCT) or sen(PMLOG_FAN_RPM)
        out.ok = bool(out.temp or out.power_w or out.gpu_util)
        return out

    def close(self) -> None:
        if self.available and self.lib:
            try:
                self.lib.ADL2_Main_Control_Destroy(self.ctx)
            except Exception:
                pass
            self.available = False
