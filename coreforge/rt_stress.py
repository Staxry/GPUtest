"""Нагрузка RT-ядер через Vulkan ray query + TLAS."""

from __future__ import annotations

import ctypes
import struct
import time
from ctypes import POINTER, byref, c_char_p, c_float, c_int, c_uint32, c_uint64, c_void_p
from typing import Callable

from coreforge.rt_spirv import make_rt_spv

VK_SUCCESS = 0
VK_QUEUE_COMPUTE_BIT = 2
VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT = 2
VK_MEMORY_PROPERTY_HOST_COHERENT_BIT = 4
VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT = 1
VK_BUFFER_USAGE_ACCEL_IN = 0x00080000
VK_BUFFER_USAGE_ACCEL_STORE = 0x00100000
VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS = 0x00020000
VK_BUFFER_USAGE_STORAGE = 0x00000020
VK_BUFFER_USAGE_UNIFORM = 0x00000010
VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT = 2
VK_SHADER_STAGE_COMPUTE = 0x20
VK_DESCRIPTOR_TYPE_AS = 1000150000
VK_DESCRIPTOR_TYPE_STORAGE_IMAGE = 3
VK_DESCRIPTOR_TYPE_UNIFORM = 6
VK_IMAGE_LAYOUT_GENERAL = 8
VK_FORMAT_RGBA8 = 37
VK_FORMAT_RGB32F = 106
VK_GEOMETRY_TYPE_TRIANGLES = 0
VK_GEOMETRY_OPAQUE = 1
VK_AS_TYPE_TLAS = 0
VK_AS_TYPE_BLAS = 1
VK_BUILD_PREFER_FAST_TRACE = 4
VK_BUILD_TYPE_DEVICE = 1
VK_INDEX_UINT32 = 1
VK_PIPELINE_COMPUTE = 1
VK_TRUE = 1
VK_CMD_POOL_RESET = 0x00000002
VK_CMD_POOL_TRANSIENT = 0x00000001
VK_PIPELINE_STAGE_TOP = 0x00000001
VK_PIPELINE_STAGE_COMPUTE = 0x00000800
VK_PIPELINE_STAGE_AS_BUILD = 0x02000000
VK_ACCESS_AS_READ = 0x00200000
VK_ACCESS_AS_WRITE = 0x00400000
VK_ACCESS_SHADER_WRITE = 0x00000040
VK_IMAGE_LAYOUT_UNDEFINED = 0
VK_IMAGE_ASPECT_COLOR = 1

API_1_2 = (1 << 22) | (2 << 12)


class VkExtent3D(ctypes.Structure):
    _fields_ = [("width", c_uint32), ("height", c_uint32), ("depth", c_uint32)]


class RayStress:
    def __init__(self, width, height, intensity, stop_event, on_mrays, on_log=None) -> None:
        self.width = max(640, width)
        self.height = max(360, height)
        self.intensity = max(10, min(100, intensity))
        self.stop_event = stop_event
        self.on_mrays = on_mrays
        self.on_log = on_log or (lambda _t: None)
        self._alive = True

    def request_close(self) -> None:
        self._alive = False

    def run(self) -> None:
        vk = None
        try:
            vk = _VulkanRT(self.width, self.height, self.intensity / 100.0, self.on_log)
            self.on_log(f"RT-ядра: Vulkan ray query {self.width}x{self.height}")
            t0 = time.perf_counter()
            last = t0
            rays = 0
            bounces = 1
            batches = max(16, int(20 + self.intensity / 2))
            while self._alive and not self.stop_event.is_set():
                vk.dispatch(time.perf_counter() - t0, batches)
                rays += self.width * self.height * bounces * batches
                now = time.perf_counter()
                if now - last >= 0.4:
                    self.on_mrays(rays / max(now - last, 1e-4) / 1e6)
                    rays = 0
                    last = now
        except Exception as exc:
            self.on_log(f"RT-ядра: {exc}")
            self.on_mrays(0.0)
            while self._alive and not self.stop_event.is_set():
                time.sleep(0.25)
        finally:
            if vk:
                vk.close()


class _VulkanRT:
    def __init__(self, width: int, height: int, work: float, log) -> None:
        self.w = width
        self.h = height
        self.work = work
        self.log = log
        self.lib = ctypes.WinDLL("vulkan-1.dll")
        self._keep: list = []
        self._bind_base()
        self._instance = c_void_p()
        self._check(self.lib.vkCreateInstance(self._inst_info(), None, byref(self._instance)), "vkCreateInstance")
        self._phys = self._pick_gpu()
        self._qfam = self._queue_family()
        self._device = c_void_p()
        self._check(self.lib.vkCreateDevice(self._phys, self._dev_info(), None, byref(self._device)), "vkCreateDevice")
        self._queue = c_void_p()
        self.lib.vkGetDeviceQueue(self._device, self._qfam, 0, byref(self._queue))
        self._load_khr()
        self._pool = self._cmd_pool()
        self._cmd = self._alloc_cmd()
        verts, ntri = _cube_mesh()
        self._vbuf, self._vmem, vaddr = self._device_buffer(verts, VK_BUFFER_USAGE_ACCEL_IN | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS)
        if not vaddr:
            raise RuntimeError("адрес вершин = 0")
        geom = self._tri_geom(vaddr, ntri)
        self._blas, self._blas_buf = self._build_as(geom, ntri, VK_AS_TYPE_BLAS)
        grid = 10
        inst = self._instances(grid)
        self._ibuf, self._imem, iaddr = self._device_buffer(inst, VK_BUFFER_USAGE_ACCEL_IN | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS)
        if not iaddr:
            raise RuntimeError("адрес инстансов = 0")
        ninst = grid * grid * grid
        tgeom = self._inst_geom(iaddr, ninst)
        try:
            self._tlas, self._tlas_buf = self._build_as(tgeom, ninst, VK_AS_TYPE_TLAS)
        except OSError as exc:
            raise RuntimeError(f"сборка TLAS: {exc}") from exc
        self._image, self._imem_img, self._view = self._storage_image()
        self._transition_image()
        self._ubo, self._ubomem, _ = self._host_buffer(b"\x00" * 16, VK_BUFFER_USAGE_UNIFORM)
        self._pipe, self._layout, self._dset = self._pipeline()
        self._write_descriptors()

    def _bind_base(self) -> None:
        L = self.lib
        L.vkCreateInstance.argtypes = [c_void_p, c_void_p, POINTER(c_void_p)]
        L.vkCreateInstance.restype = c_int
        L.vkDestroyInstance.argtypes = [c_void_p, c_void_p]
        L.vkEnumeratePhysicalDevices.argtypes = [c_void_p, POINTER(c_uint32), c_void_p]
        L.vkEnumeratePhysicalDevices.restype = c_int
        L.vkGetPhysicalDeviceProperties.argtypes = [c_void_p, c_void_p]
        L.vkGetPhysicalDeviceQueueFamilyProperties.argtypes = [c_void_p, POINTER(c_uint32), c_void_p]
        L.vkGetPhysicalDeviceMemoryProperties.argtypes = [c_void_p, c_void_p]
        L.vkCreateDevice.argtypes = [c_void_p, c_void_p, c_void_p, POINTER(c_void_p)]
        L.vkCreateDevice.restype = c_int
        L.vkDestroyDevice.argtypes = [c_void_p, c_void_p]
        L.vkGetDeviceQueue.argtypes = [c_void_p, c_uint32, c_uint32, POINTER(c_void_p)]
        L.vkCreateCommandPool.argtypes = [c_void_p, c_void_p, c_void_p, POINTER(c_void_p)]
        L.vkCreateCommandPool.restype = c_int
        L.vkAllocateCommandBuffers.argtypes = [c_void_p, c_void_p, POINTER(c_void_p)]
        L.vkAllocateCommandBuffers.restype = c_int
        L.vkBeginCommandBuffer.argtypes = [c_void_p, c_void_p]
        L.vkBeginCommandBuffer.restype = c_int
        L.vkEndCommandBuffer.argtypes = [c_void_p]
        L.vkEndCommandBuffer.restype = c_int
        L.vkQueueSubmit.argtypes = [c_void_p, c_uint32, c_void_p, c_void_p]
        L.vkQueueSubmit.restype = c_int
        L.vkQueueWaitIdle.argtypes = [c_void_p]
        L.vkQueueWaitIdle.restype = c_int
        L.vkCreateBuffer.argtypes = [c_void_p, c_void_p, c_void_p, POINTER(c_void_p)]
        L.vkCreateBuffer.restype = c_int
        L.vkGetBufferMemoryRequirements.argtypes = [c_void_p, c_void_p, c_void_p]
        L.vkAllocateMemory.argtypes = [c_void_p, c_void_p, c_void_p, POINTER(c_void_p)]
        L.vkAllocateMemory.restype = c_int
        L.vkBindBufferMemory.argtypes = [c_void_p, c_void_p, c_void_p, c_uint64]
        L.vkBindBufferMemory.restype = c_int
        L.vkMapMemory.argtypes = [c_void_p, c_void_p, c_uint64, c_uint64, c_uint32, POINTER(c_void_p)]
        L.vkMapMemory.restype = c_int
        L.vkUnmapMemory.argtypes = [c_void_p, c_void_p]
        L.vkCreateShaderModule.argtypes = [c_void_p, c_void_p, c_void_p, POINTER(c_void_p)]
        L.vkCreateShaderModule.restype = c_int
        L.vkCreateDescriptorSetLayout.argtypes = [c_void_p, c_void_p, c_void_p, POINTER(c_void_p)]
        L.vkCreateDescriptorSetLayout.restype = c_int
        L.vkCreatePipelineLayout.argtypes = [c_void_p, c_void_p, c_void_p, POINTER(c_void_p)]
        L.vkCreatePipelineLayout.restype = c_int
        L.vkCreateComputePipelines.argtypes = [c_void_p, c_void_p, c_uint32, c_void_p, c_void_p, POINTER(c_void_p)]
        L.vkCreateComputePipelines.restype = c_int
        L.vkCreateDescriptorPool.argtypes = [c_void_p, c_void_p, c_void_p, POINTER(c_void_p)]
        L.vkCreateDescriptorPool.restype = c_int
        L.vkAllocateDescriptorSets.argtypes = [c_void_p, c_void_p, POINTER(c_void_p)]
        L.vkAllocateDescriptorSets.restype = c_int
        L.vkUpdateDescriptorSets.argtypes = [c_void_p, c_uint32, c_void_p, c_uint32, c_void_p]
        L.vkCreateImage.argtypes = [c_void_p, c_void_p, c_void_p, POINTER(c_void_p)]
        L.vkCreateImage.restype = c_int
        L.vkGetImageMemoryRequirements.argtypes = [c_void_p, c_void_p, c_void_p]
        L.vkBindImageMemory.argtypes = [c_void_p, c_void_p, c_void_p, c_uint64]
        L.vkBindImageMemory.restype = c_int
        L.vkCreateImageView.argtypes = [c_void_p, c_void_p, c_void_p, POINTER(c_void_p)]
        L.vkCreateImageView.restype = c_int
        L.vkCmdBindPipeline.argtypes = [c_void_p, c_uint32, c_void_p]
        L.vkCmdBindDescriptorSets.argtypes = [c_void_p, c_uint32, c_void_p, c_uint32, c_uint32, POINTER(c_void_p), c_uint32, c_void_p]
        L.vkCmdDispatch.argtypes = [c_void_p, c_uint32, c_uint32, c_uint32]
        L.vkGetDeviceProcAddr.restype = c_void_p
        L.vkGetDeviceProcAddr.argtypes = [c_void_p, c_char_p]
        L.vkGetInstanceProcAddr.restype = c_void_p
        L.vkGetInstanceProcAddr.argtypes = [c_void_p, c_char_p]
        L.vkResetCommandBuffer.argtypes = [c_void_p, c_uint32]
        L.vkResetCommandBuffer.restype = c_int
        L.vkResetCommandPool.argtypes = [c_void_p, c_void_p, c_uint32]
        L.vkResetCommandPool.restype = c_int
        L.vkCmdPipelineBarrier.argtypes = [
            c_void_p, c_uint32, c_uint32, c_uint32,
            c_uint32, c_void_p, c_uint32, c_void_p, c_uint32, c_void_p,
        ]

    def _pfn(self, name: str, restype, *argtypes):
        addr = self.lib.vkGetDeviceProcAddr(self._device, name.encode())
        if not addr:
            addr = self.lib.vkGetInstanceProcAddr(self._instance, name.encode())
        if not addr:
            raise RuntimeError(f"нет {name}")
        fn = ctypes.CFUNCTYPE(restype, *argtypes)(addr)
        self._keep.append(fn)
        return fn

    def _load_khr(self) -> None:
        self.vkGetBufferDeviceAddress = self._pfn("vkGetBufferDeviceAddress", c_uint64, c_void_p, c_void_p)
        self.vkCreateAS = self._pfn("vkCreateAccelerationStructureKHR", c_int, c_void_p, c_void_p, c_void_p, POINTER(c_void_p))
        self.vkDestroyAS = self._pfn("vkDestroyAccelerationStructureKHR", None, c_void_p, c_void_p, c_void_p)
        self.vkGetASBuildSizes = self._pfn("vkGetAccelerationStructureBuildSizesKHR", None, c_void_p, c_uint32, c_void_p, POINTER(c_uint32), c_void_p)
        self.vkCmdBuildAS = self._pfn("vkCmdBuildAccelerationStructuresKHR", None, c_void_p, c_uint32, c_void_p, c_void_p)
        self.vkGetASAddress = self._pfn("vkGetAccelerationStructureDeviceAddressKHR", c_uint64, c_void_p, c_void_p)

    def _check(self, code: int, name: str) -> None:
        if code != VK_SUCCESS:
            raise RuntimeError(f"{name} = {code}")

    def _cstrs(self, items: list[str]):
        arr = (c_char_p * len(items))(*[x.encode() for x in items])
        self._keep.append(arr)
        return arr

    def _inst_info(self):
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

        app = App(0, None, b"CoreForge", 1, b"cf", 1, API_1_2)
        self._keep.append(app)
        info = Info(1, None, 0, ctypes.pointer(app), 0, None, 0, None)
        self._keep.append(info)
        return ctypes.pointer(info)

    def _pick_gpu(self) -> int:
        n = c_uint32()
        self._check(self.lib.vkEnumeratePhysicalDevices(self._instance, byref(n), None), "enum gpu")
        arr = (c_void_p * n.value)()
        self._check(self.lib.vkEnumeratePhysicalDevices(self._instance, byref(n), arr), "enum gpu2")
        if n.value < 1:
            raise RuntimeError("нет Vulkan GPU")
        return arr[0]

    def _queue_family(self) -> int:
        n = c_uint32()
        self.lib.vkGetPhysicalDeviceQueueFamilyProperties(self._phys, byref(n), None)
        class Q(ctypes.Structure):
            _fields_ = [("flags", c_uint32), ("count", c_uint32), ("ts", c_uint32), ("min", VkExtent3D)]
        props = (Q * n.value)()
        self.lib.vkGetPhysicalDeviceQueueFamilyProperties(self._phys, byref(n), props)
        for i, p in enumerate(props):
            if p.flags & VK_QUEUE_COMPUTE_BIT:
                return i
        raise RuntimeError("нет compute queue")

    def _dev_info(self):
        class FeatAS(ctypes.Structure):
            _fields_ = [("sType", c_uint32), ("pNext", c_void_p), ("as", c_uint32),
                        ("cap", c_uint32), ("indirect", c_uint32), ("host", c_uint32),
                        ("indexing", c_uint32)]

        class FeatRQ(ctypes.Structure):
            _fields_ = [("sType", c_uint32), ("pNext", c_void_p), ("rq", c_uint32)]

        class FeatBDA(ctypes.Structure):
            _fields_ = [("sType", c_uint32), ("pNext", c_void_p), ("bda", c_uint32),
                        ("multi", c_uint32), ("capture", c_uint32)]

        class QInfo(ctypes.Structure):
            _fields_ = [("sType", c_uint32), ("pNext", c_void_p), ("flags", c_uint32),
                        ("family", c_uint32), ("count", c_uint32), ("prios", POINTER(c_float))]

        class DInfo(ctypes.Structure):
            _fields_ = [
                ("sType", c_uint32), ("pNext", c_void_p), ("flags", c_uint32),
                ("nq", c_uint32), ("pq", POINTER(QInfo)),
                ("nl", c_uint32), ("pl", POINTER(c_char_p)),
                ("ne", c_uint32), ("pe", POINTER(c_char_p)),
                ("feat", c_void_p),
            ]

        prio = c_float(1.0)
        self._keep.append(prio)
        qi = QInfo(2, None, 0, self._qfam, 1, ctypes.pointer(prio))
        self._keep.append(qi)
        exts = self._cstrs([
            "VK_KHR_acceleration_structure",
            "VK_KHR_ray_query",
            "VK_KHR_deferred_host_operations",
        ])
        bda = FeatBDA(1000257000, None, 1, 0, 0)
        rq = FeatRQ(1000348013, ctypes.cast(ctypes.pointer(bda), c_void_p), 1)
        asf = FeatAS(1000150001, ctypes.cast(ctypes.pointer(rq), c_void_p), 1, 0, 0, 0, 0)
        self._keep += [bda, rq, asf]
        di = DInfo(3, ctypes.cast(ctypes.pointer(asf), c_void_p), 0, 1, ctypes.pointer(qi), 0, None, 3, exts, None)
        self._keep.append(di)
        return ctypes.pointer(di)

    def _cmd_pool(self) -> int:
        class Info(ctypes.Structure):
            _fields_ = [("sType", c_uint32), ("pNext", c_void_p), ("flags", c_uint32), ("family", c_uint32)]
        info = Info(39, None, VK_CMD_POOL_RESET | VK_CMD_POOL_TRANSIENT, self._qfam)
        pool = c_void_p()
        self._check(self.lib.vkCreateCommandPool(self._device, byref(info), None, byref(pool)), "cmdpool")
        self._keep.append(pool)
        return pool

    def _alloc_cmd(self):
        class Info(ctypes.Structure):
            _fields_ = [("sType", c_uint32), ("pNext", c_void_p), ("pool", c_void_p), ("level", c_uint32), ("count", c_uint32)]
        info = Info(40, None, self._pool, 0, 1)
        cmd = c_void_p()
        self._check(self.lib.vkAllocateCommandBuffers(self._device, byref(info), byref(cmd)), "cmd")
        if not cmd.value:
            raise RuntimeError("пустой command buffer")
        self._keep.append(cmd)
        return cmd

    def _mem_type(self, bits: int, flags: int) -> int:
        class Heap(ctypes.Structure):
            _fields_ = [("size", c_uint64), ("flags", c_uint32)]
        class Type(ctypes.Structure):
            _fields_ = [("prop", c_uint32), ("heap", c_uint32)]
        class Props(ctypes.Structure):
            _fields_ = [("nType", c_uint32), ("types", Type * 32), ("nHeap", c_uint32), ("heaps", Heap * 16)]
        p = Props()
        self.lib.vkGetPhysicalDeviceMemoryProperties(self._phys, byref(p))
        for want in (flags, flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT):
            if not want:
                continue
            for i in range(p.nType):
                if (bits & (1 << i)) and (p.types[i].prop & want) == want:
                    return i
        raise RuntimeError("нет подходящей памяти")

    def _host_buffer(self, data: bytes, usage: int):
        buf, mem = self._mk_buffer(len(data), usage, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
        ptr = c_void_p()
        self._check(self.lib.vkMapMemory(self._device, mem, 0, len(data), 0, byref(ptr)), "map")
        ctypes.memmove(ptr, data, len(data))
        self.lib.vkUnmapMemory(self._device, mem)
        return buf, mem, 0

    def _device_buffer(self, data: bytes, usage: int):
        buf, mem = self._mk_buffer(
            len(data),
            usage,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            address=True,
        )
        ptr = c_void_p()
        self._check(self.lib.vkMapMemory(self._device, mem, 0, len(data), 0, byref(ptr)), "map")
        ctypes.memmove(ptr, data, len(data))
        self.lib.vkUnmapMemory(self._device, mem)
        class Info(ctypes.Structure):
            _fields_ = [("sType", c_uint32), ("pNext", c_void_p), ("buffer", c_void_p)]
        info = Info(1000244001, None, buf)
        addr = self.vkGetBufferDeviceAddress(self._device, byref(info))
        return buf, mem, addr

    def _mk_buffer(self, size: int, usage: int, memflags: int, address: bool = False):
        class BInfo(ctypes.Structure):
            _fields_ = [
                ("sType", c_uint32), ("pNext", c_void_p), ("flags", c_uint32),
                ("size", c_uint64), ("usage", c_uint32), ("share", c_uint32),
                ("nqf", c_uint32), ("pqf", c_void_p),
            ]
        class Req(ctypes.Structure):
            _fields_ = [("size", c_uint64), ("align", c_uint64), ("bits", c_uint32)]
        class AInfo(ctypes.Structure):
            _fields_ = [("sType", c_uint32), ("pNext", c_void_p), ("size", c_uint64), ("type", c_uint32)]
        class FInfo(ctypes.Structure):
            _fields_ = [("sType", c_uint32), ("pNext", c_void_p), ("flags", c_uint32), ("deviceMask", c_uint32)]
        bi = BInfo(12, None, 0, size, usage, 0, 0, None)
        buf = c_void_p()
        self._check(self.lib.vkCreateBuffer(self._device, byref(bi), None, byref(buf)), "buffer")
        req = Req()
        self.lib.vkGetBufferMemoryRequirements(self._device, buf, byref(req))
        flags = FInfo(1000070000, None, VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT if address else 0, 0)
        ai = AInfo(5, ctypes.cast(ctypes.pointer(flags), c_void_p) if address else None, req.size, self._mem_type(req.bits, memflags))
        self._keep.append(flags)
        mem = c_void_p()
        self._check(self.lib.vkAllocateMemory(self._device, byref(ai), None, byref(mem)), "alloc")
        self._check(self.lib.vkBindBufferMemory(self._device, buf, mem, 0), "bind")
        return buf, mem

    def _tri_geom(self, vaddr: int, ntri: int):
        class Tri(ctypes.Structure):
            _pack_ = 8
            _fields_ = [
                ("sType", c_uint32), ("pNext", c_void_p),
                ("vfmt", c_uint32), ("vdata", c_uint64),
                ("vstride", c_uint64), ("vmax", c_uint32), ("itype", c_uint32),
                ("idata", c_uint64), ("tdata", c_uint64),
            ]
        class GeomData(ctypes.Union):
            _pack_ = 8
            _fields_ = [("tri", Tri), ("pad", ctypes.c_ubyte * 64)]
        class Geom(ctypes.Structure):
            _pack_ = 8
            _fields_ = [
                ("sType", c_uint32), ("pNext", c_void_p), ("gtype", c_uint32),
                ("data", GeomData), ("flags", c_uint32),
            ]
        g = Geom()
        g.sType = 1000150006
        g.gtype = VK_GEOMETRY_TYPE_TRIANGLES
        g.flags = VK_GEOMETRY_OPAQUE
        g.data.tri.sType = 1000150005
        g.data.tri.vfmt = VK_FORMAT_RGB32F
        g.data.tri.vdata = vaddr
        g.data.tri.vstride = 12
        g.data.tri.vmax = ntri * 3 - 1
        g.data.tri.itype = 1000165000
        self._keep.append(g)
        return g

    def _inst_geom(self, addr: int, count: int):
        class Inst(ctypes.Structure):
            _pack_ = 8
            _fields_ = [
                ("sType", c_uint32), ("pNext", c_void_p),
                ("arrayOfPointers", c_uint32), ("data", c_uint64),
            ]
        class GeomData(ctypes.Union):
            _pack_ = 8
            _fields_ = [("inst", Inst), ("pad", ctypes.c_ubyte * 64)]
        class Geom(ctypes.Structure):
            _pack_ = 8
            _fields_ = [
                ("sType", c_uint32), ("pNext", c_void_p), ("gtype", c_uint32),
                ("data", GeomData), ("flags", c_uint32),
            ]
        g = Geom()
        g.sType = 1000150006
        g.gtype = 2
        g.flags = 0
        g.data.inst.sType = 1000150004
        g.data.inst.pNext = None
        g.data.inst.arrayOfPointers = 0
        g.data.inst.data = addr
        self._keep.append(g)
        return g

    def _build_as(self, geom, prim_count: int, as_type: int):
        class Build(ctypes.Structure):
            _pack_ = 8
            _fields_ = [
                ("sType", c_uint32), ("pNext", c_void_p), ("type", c_uint32), ("flags", c_uint32),
                ("mode", c_uint32), ("src", c_void_p), ("dst", c_void_p), ("ngeom", c_uint32),
                ("pgeom", c_void_p), ("ppgeom", c_void_p), ("scratch", c_uint64),
            ]
        class Sizes(ctypes.Structure):
            _fields_ = [("sType", c_uint32), ("pNext", c_void_p), ("asSize", c_uint64),
                        ("upd", c_uint64), ("build", c_uint64)]
        p_geom = ctypes.pointer(geom)
        self._keep.append(p_geom)
        build = Build(
            1000150000, None, as_type, VK_BUILD_PREFER_FAST_TRACE, 0, None, None, 1,
            ctypes.cast(p_geom, c_void_p), None, 0,
        )
        sizes = Sizes(1000150020, None, 0, 0, 0)
        cnt = c_uint32(prim_count)
        try:
            self.vkGetASBuildSizes(self._device, VK_BUILD_TYPE_DEVICE, byref(build), byref(cnt), byref(sizes))
        except OSError as exc:
            self.log(f"AS GetBuildSizes type={as_type}: {exc}, запас")
            sizes.asSize = max(2 * 1024 * 1024, prim_count * 256 + 65536)
            sizes.build = max(8 * 1024 * 1024, prim_count * 1024 + 262144)
        if sizes.asSize < 256 or sizes.build < 256:
            sizes.asSize = max(sizes.asSize, 2 * 1024 * 1024)
            sizes.build = max(sizes.build, 8 * 1024 * 1024)
        self.log(f"AS sizes type={as_type} prim={prim_count} as={sizes.asSize} scratch={sizes.build}")
        as_buf, as_mem = self._mk_buffer(
            max(256, sizes.asSize),
            VK_BUFFER_USAGE_ACCEL_STORE | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            address=True,
        )
        scratch, scratch_mem = self._mk_buffer(
            max(256, sizes.build),
            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS | VK_BUFFER_USAGE_STORAGE,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            address=True,
        )
        class AddrInfo(ctypes.Structure):
            _fields_ = [("sType", c_uint32), ("pNext", c_void_p), ("buffer", c_void_p)]
        scratch_addr = self.vkGetBufferDeviceAddress(self._device, byref(AddrInfo(1000244001, None, scratch)))
        if not scratch_addr:
            raise RuntimeError("адрес scratch = 0")
        class ASInfo(ctypes.Structure):
            _pack_ = 8
            _fields_ = [
                ("sType", c_uint32), ("pNext", c_void_p), ("createFlags", c_uint32),
                ("buf", c_void_p), ("off", c_uint64), ("size", c_uint64),
                ("type", c_uint32), ("devAddr", c_uint64),
            ]
        ai = ASInfo(1000150017, None, 0, as_buf, 0, sizes.asSize, as_type, 0)
        accel = c_void_p()
        self._check(self.vkCreateAS(self._device, byref(ai), None, byref(accel)), "create AS")
        build.dst = accel
        build.scratch = scratch_addr
        class Range(ctypes.Structure):
            _fields_ = [("prim", c_uint32), ("primOff", c_uint32), ("first", c_uint32), ("trans", c_uint32)]
        rg = Range(prim_count, 0, 0, 0)
        p_rg = ctypes.pointer(rg)
        pp_rg = (ctypes.POINTER(Range) * 1)(p_rg)
        self._keep += [rg, p_rg, pp_rg, build]
        pp = ctypes.cast(pp_rg, c_void_p)
        def rec_build() -> None:
            self.vkCmdBuildAS(self._cmd, 1, byref(build), pp)
            self._cmd_as_barrier()

        self._submit(rec_build)
        return accel, as_buf

    def _instances(self, grid: int) -> bytes:
        class Addr(ctypes.Structure):
            _fields_ = [("sType", c_uint32), ("pNext", c_void_p), ("as", c_void_p)]
        info = Addr(1000150002, None, self._blas)
        blas_addr = self.vkGetASAddress(self._device, byref(info))
        if not blas_addr:
            raise RuntimeError("адрес BLAS = 0")
        self.log(f"BLAS device address {blas_addr:#x}")
        out = bytearray()
        spacing = 2.2
        off = -(grid - 1) * spacing * 0.5
        for z in range(grid):
            for y in range(grid):
                for x in range(grid):
                    tx = off + x * spacing
                    ty = off + y * spacing
                    tz = off + z * spacing
                    xform = [
                        1, 0, 0, tx,
                        0, 1, 0, ty,
                        0, 0, 1, tz,
                    ]
                    out += struct.pack("<12f", *xform)
                    custom_mask = (0 << 0) | (0xFF << 24)
                    sbt_flags = 0
                    out += struct.pack("<II", custom_mask, sbt_flags)
                    out += struct.pack("<Q", blas_addr)
        return bytes(out)

    def _storage_image(self):
        class Img(ctypes.Structure):
            _fields_ = [
                ("sType", c_uint32), ("pNext", c_void_p), ("flags", c_uint32), ("type", c_uint32),
                ("format", c_uint32), ("extent", VkExtent3D), ("mips", c_uint32), ("layers", c_uint32),
                ("samples", c_uint32), ("tiling", c_uint32), ("usage", c_uint32), ("share", c_uint32),
                ("nqf", c_uint32), ("pqf", c_void_p), ("init", c_uint32),
            ]
        class Req(ctypes.Structure):
            _fields_ = [("size", c_uint64), ("align", c_uint64), ("bits", c_uint32)]
        class AInfo(ctypes.Structure):
            _fields_ = [("sType", c_uint32), ("pNext", c_void_p), ("size", c_uint64), ("type", c_uint32)]
        class View(ctypes.Structure):
            _fields_ = [
                ("sType", c_uint32), ("pNext", c_void_p), ("flags", c_uint32), ("image", c_void_p),
                ("view", c_uint32), ("format", c_uint32),
                ("r", c_uint32), ("g", c_uint32), ("b", c_uint32), ("a", c_uint32),
                ("aspect", c_uint32), ("baseMip", c_uint32), ("mipCnt", c_uint32),
                ("baseLayer", c_uint32), ("layerCnt", c_uint32),
            ]
        info = Img(14, None, 0, 1, VK_FORMAT_RGBA8, VkExtent3D(self.w, self.h, 1), 1, 1, 1, 0, 8, 0, 0, None, VK_IMAGE_LAYOUT_UNDEFINED)
        image = c_void_p()
        self._check(self.lib.vkCreateImage(self._device, byref(info), None, byref(image)), "image")
        req = Req()
        self.lib.vkGetImageMemoryRequirements(self._device, image, byref(req))
        ai = AInfo(5, None, req.size, self._mem_type(req.bits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT))
        mem = c_void_p()
        self._check(self.lib.vkAllocateMemory(self._device, byref(ai), None, byref(mem)), "imgmem")
        self._check(self.lib.vkBindImageMemory(self._device, image, mem, 0), "bindimg")
        vi = View(15, None, 0, image, 1, VK_FORMAT_RGBA8, 0, 1, 2, 3, 1, 0, 1, 0, 1)
        view = c_void_p()
        self._check(self.lib.vkCreateImageView(self._device, byref(vi), None, byref(view)), "view")
        return image, mem, view

    def _pipeline(self):
        spv = make_rt_spv()
        class SM(ctypes.Structure):
            _fields_ = [("sType", c_uint32), ("pNext", c_void_p), ("flags", c_uint32),
                        ("size", ctypes.c_size_t), ("code", c_void_p)]
        buf = ctypes.create_string_buffer(spv, len(spv))
        self._keep.append(buf)
        smi = SM(16, None, 0, len(spv), ctypes.addressof(buf))
        module = c_void_p()
        self._check(self.lib.vkCreateShaderModule(self._device, byref(smi), None, byref(module)), "shader")
        class Bind(ctypes.Structure):
            _fields_ = [("bind", c_uint32), ("type", c_uint32), ("count", c_uint32), ("stage", c_uint32), ("immut", c_void_p)]
        binds = (Bind * 3)(
            Bind(0, VK_DESCRIPTOR_TYPE_AS, 1, VK_SHADER_STAGE_COMPUTE, None),
            Bind(1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE, None),
            Bind(2, VK_DESCRIPTOR_TYPE_UNIFORM, 1, VK_SHADER_STAGE_COMPUTE, None),
        )
        class LInfo(ctypes.Structure):
            _fields_ = [("sType", c_uint32), ("pNext", c_void_p), ("flags", c_uint32),
                        ("n", c_uint32), ("pb", c_void_p)]
        li = LInfo(32, None, 0, 3, ctypes.addressof(binds))
        self._keep.append(binds)
        dsl = c_void_p()
        self._check(self.lib.vkCreateDescriptorSetLayout(self._device, byref(li), None, byref(dsl)), "dsl")
        class PL(ctypes.Structure):
            _fields_ = [("sType", c_uint32), ("pNext", c_void_p), ("flags", c_uint32),
                        ("nset", c_uint32), ("pset", POINTER(c_void_p)), ("npush", c_uint32), ("ppush", c_void_p)]
        ds_ptr = c_void_p(dsl.value)
        self._keep.append(ds_ptr)
        pli = PL(30, None, 0, 1, ctypes.pointer(ds_ptr), 0, None)
        layout = c_void_p()
        self._check(self.lib.vkCreatePipelineLayout(self._device, byref(pli), None, byref(layout)), "layout")
        class Stage(ctypes.Structure):
            _fields_ = [("sType", c_uint32), ("pNext", c_void_p), ("flags", c_uint32),
                        ("stage", c_uint32), ("module", c_void_p), ("name", c_char_p), ("spec", c_void_p)]
        class CPI(ctypes.Structure):
            _fields_ = [("sType", c_uint32), ("pNext", c_void_p), ("flags", c_uint32),
                        ("stage", Stage), ("layout", c_void_p), ("base", c_void_p), ("baseIndex", c_int)]
        name = ctypes.create_string_buffer(b"main")
        self._keep += [name, module]
        st = Stage(18, None, 0, VK_SHADER_STAGE_COMPUTE, module, ctypes.cast(name, c_char_p), None)
        cpi = CPI(29, None, 0, st, layout, None, -1)
        self._keep += [st, cpi]
        pipe = c_void_p()
        self._check(self.lib.vkCreateComputePipelines(self._device, None, 1, byref(cpi), None, byref(pipe)), "pipe")
        class Size(ctypes.Structure):
            _fields_ = [("type", c_uint32), ("count", c_uint32)]
        sizes = (Size * 3)(Size(VK_DESCRIPTOR_TYPE_AS, 1), Size(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1), Size(VK_DESCRIPTOR_TYPE_UNIFORM, 1))
        class DP(ctypes.Structure):
            _fields_ = [("sType", c_uint32), ("pNext", c_void_p), ("flags", c_uint32),
                        ("max", c_uint32), ("n", c_uint32), ("ps", c_void_p)]
        dpi = DP(33, None, 0, 1, 3, ctypes.addressof(sizes))
        self._keep.append(sizes)
        pool = c_void_p()
        self._check(self.lib.vkCreateDescriptorPool(self._device, byref(dpi), None, byref(pool)), "dpool")
        class DA(ctypes.Structure):
            _fields_ = [("sType", c_uint32), ("pNext", c_void_p), ("pool", c_void_p),
                        ("n", c_uint32), ("pl", POINTER(c_void_p))]
        dai = DA(34, None, pool, 1, ctypes.pointer(ds_ptr))
        dset = c_void_p()
        self._check(self.lib.vkAllocateDescriptorSets(self._device, byref(dai), byref(dset)), "dset")
        self._dsl = dsl
        self._dpool = pool
        self._module = module
        return pipe, layout, dset

    def _write_descriptors(self) -> None:
        class WAS(ctypes.Structure):
            _fields_ = [("sType", c_uint32), ("pNext", c_void_p), ("n", c_uint32), ("pas", c_void_p)]
        class Img(ctypes.Structure):
            _fields_ = [("samp", c_void_p), ("view", c_void_p), ("layout", c_uint32)]
        class Buf(ctypes.Structure):
            _fields_ = [("buffer", c_void_p), ("off", c_uint64), ("range", c_uint64)]
        class Write(ctypes.Structure):
            _fields_ = [
                ("sType", c_uint32), ("pNext", c_void_p), ("dst", c_void_p), ("bind", c_uint32),
                ("arr", c_uint32), ("count", c_uint32), ("type", c_uint32),
                ("pimg", c_void_p), ("pbuf", c_void_p), ("ptex", c_void_p),
            ]
        tlas_val = self._tlas.value if isinstance(self._tlas, c_void_p) else int(self._tlas)
        as_arr = (c_uint64 * 1)(tlas_val or 0)
        was = WAS()
        was.sType = 1000150007
        was.pNext = None
        was.n = 1
        was.pas = ctypes.cast(as_arr, c_void_p)
        img = Img(None, self._view, VK_IMAGE_LAYOUT_GENERAL)
        buf = Buf(self._ubo, 0, 16)
        self._keep += [was, img, buf, as_arr]
        w0 = Write()
        w0.sType = 35
        w0.pNext = ctypes.cast(ctypes.pointer(was), c_void_p)
        w0.dst = self._dset
        w0.bind = 0
        w0.arr = 0
        w0.count = 1
        w0.type = VK_DESCRIPTOR_TYPE_AS
        w1 = Write()
        w1.sType = 35
        w1.dst = self._dset
        w1.bind = 1
        w1.count = 1
        w1.type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE
        w1.pimg = ctypes.addressof(img)
        w2 = Write()
        w2.sType = 35
        w2.dst = self._dset
        w2.bind = 2
        w2.count = 1
        w2.type = VK_DESCRIPTOR_TYPE_UNIFORM
        w2.pbuf = ctypes.addressof(buf)
        writes = (Write * 3)(w0, w1, w2)
        self._keep += [writes, w0, w1, w2]
        if not tlas_val:
            raise RuntimeError("дескриптор RT: TLAS handle = 0")
        self.lib.vkUpdateDescriptorSets(self._device, 3, writes, 0, None)

    def _submit(self, record) -> None:
        class Begin(ctypes.Structure):
            _fields_ = [("sType", c_uint32), ("pNext", c_void_p), ("flags", c_uint32), ("inh", c_void_p)]
        class Submit(ctypes.Structure):
            _fields_ = [
                ("sType", c_uint32), ("pNext", c_void_p),
                ("waitN", c_uint32), ("waits", c_void_p), ("masks", c_void_p),
                ("cmdN", c_uint32), ("cmds", POINTER(c_void_p)),
                ("sigN", c_uint32), ("sigs", c_void_p),
            ]
        self._check(self.lib.vkResetCommandPool(self._device, self._pool, 0), "reset pool")
        self._check(self.lib.vkResetCommandBuffer(self._cmd, 0), "reset")
        b = Begin(42, None, 1, None)
        self._check(self.lib.vkBeginCommandBuffer(self._cmd, byref(b)), "begin")
        record()
        self._check(self.lib.vkEndCommandBuffer(self._cmd), "end")
        cmd = self._cmd if isinstance(self._cmd, c_void_p) else c_void_p(self._cmd)
        self._keep.append(cmd)
        s = Submit(4, None, 0, None, None, 1, ctypes.pointer(cmd), 0, None)
        self._check(self.lib.vkQueueSubmit(self._queue, 1, byref(s), None), "submit")
        self._check(self.lib.vkQueueWaitIdle(self._queue), "idle")

    def _cmd_as_barrier(self) -> None:
        class MB(ctypes.Structure):
            _fields_ = [("sType", c_uint32), ("pNext", c_void_p), ("src", c_uint32), ("dst", c_uint32)]
        mb = MB(46, None, VK_ACCESS_AS_WRITE, VK_ACCESS_AS_READ | VK_ACCESS_AS_WRITE)
        self._keep.append(mb)
        self.lib.vkCmdPipelineBarrier(
            self._cmd, VK_PIPELINE_STAGE_AS_BUILD, VK_PIPELINE_STAGE_AS_BUILD | VK_PIPELINE_STAGE_COMPUTE,
            0, 1, byref(mb), 0, None, 0, None,
        )

    def _transition_image(self) -> None:
        class Range(ctypes.Structure):
            _fields_ = [
                ("aspect", c_uint32), ("baseMip", c_uint32), ("mipCnt", c_uint32),
                ("baseLayer", c_uint32), ("layerCnt", c_uint32),
            ]

        class IMB(ctypes.Structure):
            _fields_ = [
                ("sType", c_uint32), ("pNext", c_void_p),
                ("srcAccess", c_uint32), ("dstAccess", c_uint32),
                ("oldLayout", c_uint32), ("newLayout", c_uint32),
                ("srcQueue", c_uint32), ("dstQueue", c_uint32),
                ("image", c_void_p), ("sub", Range),
            ]

        imb = IMB(
            45, None, 0, VK_ACCESS_SHADER_WRITE,
            VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
            0xFFFFFFFF, 0xFFFFFFFF, self._image,
            Range(VK_IMAGE_ASPECT_COLOR, 0, 1, 0, 1),
        )
        self._keep.append(imb)

        def rec() -> None:
            self.lib.vkCmdPipelineBarrier(
                self._cmd, VK_PIPELINE_STAGE_TOP, VK_PIPELINE_STAGE_COMPUTE,
                0, 0, None, 0, None, 1, byref(imb),
            )

        self._submit(rec)

    def dispatch(self, t: float, batches: int = 4) -> None:
        payload = struct.pack("<ff", float(t), float(self.work)) + b"\x00" * 8
        ptr = c_void_p()
        self._check(self.lib.vkMapMemory(self._device, self._ubomem, 0, 16, 0, byref(ptr)), "map ubo")
        ctypes.memmove(ptr, payload, 16)
        self.lib.vkUnmapMemory(self._device, self._ubomem)
        gx = (self.w + 7) // 8
        gy = (self.h + 7) // 8
        dset = c_void_p(self._dset.value if isinstance(self._dset, c_void_p) else self._dset)
        self._keep.append(dset)
        n = max(1, int(batches))

        def rec() -> None:
            self.lib.vkCmdBindPipeline(self._cmd, VK_PIPELINE_COMPUTE, self._pipe)
            self.lib.vkCmdBindDescriptorSets(self._cmd, VK_PIPELINE_COMPUTE, self._layout, 0, 1, byref(dset), 0, None)
            for _ in range(n):
                self.lib.vkCmdDispatch(self._cmd, gx, gy, 1)

        self._submit(rec)

    def close(self) -> None:
        try:
            self.lib.vkDeviceWaitIdle = getattr(self.lib, "vkDeviceWaitIdle")
            self.lib.vkDeviceWaitIdle.argtypes = [c_void_p]
            self.lib.vkDeviceWaitIdle(self._device)
            self.lib.vkDestroyDevice(self._device, None)
            self.lib.vkDestroyInstance(self._instance, None)
        except Exception:
            pass


def _cube_mesh() -> tuple[bytes, int]:
    p = [
        (-0.6, -0.6, -0.6), (0.6, -0.6, -0.6), (0.6, 0.6, -0.6), (-0.6, 0.6, -0.6),
        (-0.6, -0.6, 0.6), (0.6, -0.6, 0.6), (0.6, 0.6, 0.6), (-0.6, 0.6, 0.6),
    ]
    faces = [
        (0, 1, 2), (0, 2, 3), (4, 6, 5), (4, 7, 6),
        (0, 4, 5), (0, 5, 1), (2, 6, 7), (2, 7, 3),
        (0, 3, 7), (0, 7, 4), (1, 5, 6), (1, 6, 2),
    ]
    raw = bytearray()
    for a, b, c in faces:
        for i in (a, b, c):
            raw += struct.pack("<3f", *p[i])
    return bytes(raw), len(faces)
