"""Определение вендора GPU: NVIDIA / AMD / Intel через DLL и Vulkan."""

from __future__ import annotations

import ctypes
from ctypes import POINTER, byref, c_char_p, c_int, c_uint32, c_void_p
from dataclasses import dataclass, field


VENDOR_NVIDIA = 0x10DE
VENDOR_AMD = 0x1002
VENDOR_INTEL = 0x8086


def _dll(*names: str) -> bool:
    for name in names:
        try:
            ctypes.WinDLL(name)
            return True
        except OSError:
            continue
    return False


@dataclass
class GpuCaps:
    vendor: str = "unknown"
    vendor_id: int = 0
    name: str = "GPU"
    vram_total: int = 0
    has_cuda: bool = False
    has_nvml: bool = False
    has_nvenc: bool = False
    has_nvdec: bool = False
    has_tensor: bool = False
    has_vulkan: bool = False
    has_ray_query: bool = False
    has_adl: bool = False
    has_opengl: bool = True
    extras: dict = field(default_factory=dict)

    @property
    def is_nvidia(self) -> bool:
        return self.vendor == "nvidia"

    @property
    def is_amd(self) -> bool:
        return self.vendor == "amd"


def _vulkan_probe() -> dict | None:
    try:
        lib = ctypes.WinDLL("vulkan-1.dll")
    except OSError:
        return None
    lib.vkCreateInstance.argtypes = [c_void_p, c_void_p, POINTER(c_void_p)]
    lib.vkCreateInstance.restype = c_int
    lib.vkDestroyInstance.argtypes = [c_void_p, c_void_p]
    lib.vkEnumeratePhysicalDevices.argtypes = [c_void_p, POINTER(c_uint32), c_void_p]
    lib.vkEnumeratePhysicalDevices.restype = c_int
    lib.vkGetPhysicalDeviceProperties.argtypes = [c_void_p, c_void_p]
    lib.vkGetPhysicalDeviceMemoryProperties.argtypes = [c_void_p, c_void_p]
    lib.vkEnumerateDeviceExtensionProperties.argtypes = [c_void_p, c_char_p, POINTER(c_uint32), c_void_p]
    lib.vkEnumerateDeviceExtensionProperties.restype = c_int

    class App(ctypes.Structure):
        _fields_ = [
            ("sType", c_uint32), ("pNext", c_void_p), ("pApp", c_char_p),
            ("appVer", c_uint32), ("pEng", c_char_p), ("engVer", c_uint32), ("api", c_uint32),
        ]

    class Info(ctypes.Structure):
        _fields_ = [
            ("sType", c_uint32), ("pNext", c_void_p), ("flags", c_uint32),
            ("pApp", POINTER(App)), ("nLayer", c_uint32), ("ppLayer", POINTER(c_char_p)),
            ("nExt", c_uint32), ("ppExt", POINTER(c_char_p)),
        ]

    class Props(ctypes.Structure):
        _fields_ = [
            ("api", c_uint32), ("driver", c_uint32), ("vendor", c_uint32), ("device", c_uint32),
            ("dtype", c_uint32), ("name", ctypes.c_char * 256), ("uuid", ctypes.c_ubyte * 16),
            ("pad", ctypes.c_ubyte * 1024),
        ]

    class Heap(ctypes.Structure):
        _fields_ = [("size", ctypes.c_uint64), ("flags", c_uint32)]

    class Type(ctypes.Structure):
        _fields_ = [("prop", c_uint32), ("heap", c_uint32)]

    class Mem(ctypes.Structure):
        _fields_ = [("nType", c_uint32), ("types", Type * 32), ("nHeap", c_uint32), ("heaps", Heap * 16)]

    class Ext(ctypes.Structure):
        _fields_ = [("name", ctypes.c_char * 256), ("ver", c_uint32)]

    app = App(0, None, b"CoreForge", 1, b"cf", 1, (1 << 22) | (2 << 12))
    info = Info(1, None, 0, ctypes.pointer(app), 0, None, 0, None)
    inst = c_void_p()
    if lib.vkCreateInstance(ctypes.pointer(info), None, byref(inst)) != 0:
        return None
    try:
        n = c_uint32()
        if lib.vkEnumeratePhysicalDevices(inst, byref(n), None) != 0 or n.value < 1:
            return None
        arr = (c_void_p * n.value)()
        lib.vkEnumeratePhysicalDevices(inst, byref(n), arr)
        best = None
        for i in range(n.value):
            p = Props()
            lib.vkGetPhysicalDeviceProperties(arr[i], byref(p))
            mem = Mem()
            lib.vkGetPhysicalDeviceMemoryProperties(arr[i], byref(mem))
            vram = 0
            for h in range(mem.nHeap):
                if mem.heaps[h].flags & 1:
                    vram += mem.heaps[h].size
            ne = c_uint32()
            lib.vkEnumerateDeviceExtensionProperties(arr[i], None, byref(ne), None)
            exts = (Ext * max(1, ne.value))()
            lib.vkEnumerateDeviceExtensionProperties(arr[i], None, byref(ne), exts)
            names = {exts[j].name.decode("ascii", "ignore") for j in range(ne.value)}
            rec = {
                "vendor_id": int(p.vendor),
                "name": p.name.decode("utf-8", "replace").strip() or "Vulkan GPU",
                "vram_total": int(vram),
                "has_ray_query": "VK_KHR_ray_query" in names,
                "has_as": "VK_KHR_acceleration_structure" in names,
            }
            if p.vendor in (VENDOR_NVIDIA, VENDOR_AMD) or best is None:
                best = rec
                if p.vendor in (VENDOR_NVIDIA, VENDOR_AMD):
                    break
        return best
    finally:
        lib.vkDestroyInstance(inst, None)


def detect() -> GpuCaps:
    caps = GpuCaps()
    vk = None
    try:
        vk = _vulkan_probe()
    except Exception:
        vk = None
    if vk:
        caps.has_vulkan = True
        caps.name = vk["name"]
        caps.vendor_id = vk["vendor_id"]
        caps.vram_total = vk["vram_total"]
        caps.has_ray_query = bool(vk["has_ray_query"] and vk["has_as"])
        if vk["vendor_id"] == VENDOR_NVIDIA:
            caps.vendor = "nvidia"
        elif vk["vendor_id"] == VENDOR_AMD:
            caps.vendor = "amd"
        elif vk["vendor_id"] == VENDOR_INTEL:
            caps.vendor = "intel"

    caps.has_cuda = _dll("nvcuda.dll")
    caps.has_nvml = _dll("nvml.dll")
    caps.has_nvenc = _dll("nvEncodeAPI64.dll")
    caps.has_nvdec = _dll("nvcuvid.dll")
    caps.has_adl = _dll("atiadlxx.dll")
    caps.has_opengl = _dll("opengl32.dll")

    if caps.vendor == "unknown":
        if caps.has_cuda or caps.has_nvml:
            caps.vendor = "nvidia"
        elif caps.has_adl or _dll("amdhip64.dll", "amdocl64.dll"):
            caps.vendor = "amd"

    if caps.is_nvidia:
        caps.has_tensor = True
    return caps
