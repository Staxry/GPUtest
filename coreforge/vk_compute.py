"""Vulkan compute / VRAM / copy — путь для AMD и запасной для NVIDIA."""

from __future__ import annotations

import ctypes
import struct
import time
from ctypes import POINTER, byref, c_char_p, c_float, c_int, c_uint32, c_uint64, c_void_p

VK_SUCCESS = 0
VK_QUEUE_COMPUTE = 2
VK_MEMORY_HOST_VISIBLE = 2
VK_MEMORY_HOST_COHERENT = 4
VK_MEMORY_DEVICE_LOCAL = 1
VK_BUFFER_STORAGE = 0x20
VK_BUFFER_TRANSFER_SRC = 0x01
VK_BUFFER_TRANSFER_DST = 0x02
VK_SHADER_COMPUTE = 0x20
API_1_2 = (1 << 22) | (2 << 12)


def _compute_spv() -> bytes:
    w: list[int] = []

    def op(code: int, *xs: int) -> None:
        w.append(((1 + len(xs)) << 16) | (code & 0xFFFF))
        w.extend(int(x) & 0xFFFFFFFF for x in xs)

    def lit(s: str) -> list[int]:
        raw = s.encode() + b"\x00"
        raw += b"\x00" * ((4 - len(raw) % 4) % 4)
        return list(struct.unpack("<" + "I" * (len(raw) // 4), raw))

    w += [0x07230203, 0x00010500, 0x00010001, 40, 0]
    op(17, 1)
    op(14, 0, 1)
    op(15, 5, 4, *lit("main"), 10)
    op(16, 4, 17, 256, 1, 1)
    op(71, 10, 11, 28)
    op(71, 13, 33, 0)
    op(71, 13, 34, 0)
    op(71, 12, 2)
    op(72, 12, 0, 35, 0)
    op(71, 13, 35, 0)
    op(19, 1)
    op(33, 2, 1)
    op(21, 5, 32, 0)
    op(22, 6, 32)
    op(23, 7, 5, 3)
    op(32, 8, 1, 7)
    op(59, 8, 10, 1)
    op(27, 11, 6)
    op(30, 12, 11)
    op(32, 9, 2, 12)
    op(59, 9, 13, 2)
    op(50, 14, 5, 0)
    op(50, 15, 5, 256)
    op(50, 16, 6, 0x3A83126F)
    op(50, 17, 6, 0x3F800000)
    op(50, 18, 6, 0x3FCF1BBD)
    op(50, 19, 6, 0x3C51B717)
    op(50, 20, 5, 4000)
    op(54, 1, 4, 0, 2)
    op(248, 21)
    op(61, 7, 22, 10)
    op(81, 5, 23, 22, 0)
    op(112, 6, 24, 23)
    op(133, 6, 25, 24, 16)
    op(129, 6, 26, 25, 17)
    op(80, 5, 27, 20)
    op(248, 28)
    op(246, 29, 30, 0)
    op(249, 30)
    op(248, 30)
    op(133, 6, 31, 26, 18)
    op(129, 6, 26, 31, 19)
    op(130, 5, 27, 27, 14)
    op(177, 32, 27, 14)
    op(250, 32, 28, 29)
    op(248, 29)
    op(52, 33, 5, 23, 15)
    op(41, 5, 34, 33, 13)
    op(62, 34, 26)
    op(253)
    op(56)
    return b"".join(struct.pack("<I", x & 0xFFFFFFFF) for x in w)


class VkComputeStress:
    def __init__(self, stop_event, intensity: int, vram_gb: float, do_alu: bool, do_vram: bool, do_copy: bool, on_log, on_tflops=None, on_gbs=None) -> None:
        self.stop_event = stop_event
        self.intensity = max(10, min(100, intensity))
        self.vram_gb = max(0.25, vram_gb)
        self.do_alu = do_alu
        self.do_vram = do_vram
        self.do_copy = do_copy
        self.on_log = on_log
        self.on_tflops = on_tflops or (lambda _v: None)
        self.on_gbs = on_gbs or (lambda _v: None)
        self._alive = True

    def request_close(self) -> None:
        self._alive = False

    def run(self) -> None:
        try:
            ctx = _VkComp()
            self.on_log(f"Vulkan compute: {ctx.name}")
            bufs = []
            if self.do_vram:
                want = int(self.vram_gb * 1024**3)
                got = ctx.alloc_blocks(want)
                bufs = ctx.blocks
                self.on_log(f"Vulkan VRAM: {got / 1024**3:.2f} ГБ")
            work = max(256, int(4096 * (self.intensity / 100.0)))
            last = time.perf_counter()
            flops = 0.0
            moved = 0.0
            while self._alive and not self.stop_event.is_set():
                if self.do_alu:
                    ctx.dispatch(work)
                    flops += work * 256 * 2
                if self.do_copy and len(bufs) >= 2:
                    n = ctx.copy_pair(bufs[0], bufs[1])
                    moved += n
                elif self.do_copy:
                    n = ctx.copy_scratch()
                    moved += n
                now = time.perf_counter()
                if now - last >= 0.4:
                    dt = now - last
                    self.on_tflops(flops / dt / 1e12)
                    self.on_gbs(moved / dt / 1e9)
                    flops = moved = 0.0
                    last = now
            ctx.close()
        except Exception as exc:
            self.on_log(f"Vulkan compute: {exc}")
            while self._alive and not self.stop_event.is_set():
                time.sleep(0.25)


class _VkComp:
    def __init__(self) -> None:
        self.lib = ctypes.WinDLL("vulkan-1.dll")
        self._keep: list = []
        self.blocks: list[tuple] = []
        self._bind()
        self._instance = c_void_p()
        self._check(self.lib.vkCreateInstance(self._inst_info(), None, byref(self._instance)), "instance")
        n = c_uint32()
        self._check(self.lib.vkEnumeratePhysicalDevices(self._instance, byref(n), None), "enum")
        arr = (c_void_p * n.value)()
        self._check(self.lib.vkEnumeratePhysicalDevices(self._instance, byref(n), arr), "enum2")
        self._phys = arr[0]
        self.name = self._gpu_name()
        self._qfam = self._queue_family()
        self._device = c_void_p()
        self._check(self.lib.vkCreateDevice(self._phys, self._dev_info(), None, byref(self._device)), "device")
        self._queue = c_void_p()
        self.lib.vkGetDeviceQueue(self._device, self._qfam, 0, byref(self._queue))
        self._pool = self._cmd_pool()
        self._cmd = self._alloc_cmd()
        self._scratch_a, self._scratch_sz = self._mk_buffer(4 * 1024 * 1024, VK_BUFFER_STORAGE | VK_BUFFER_TRANSFER_SRC | VK_BUFFER_TRANSFER_DST)
        self._scratch_b, _ = self._mk_buffer(4 * 1024 * 1024, VK_BUFFER_STORAGE | VK_BUFFER_TRANSFER_SRC | VK_BUFFER_TRANSFER_DST)
        self._pipe = self._layout = self._dset = None
        try:
            self._pipe, self._layout, self._dset = self._pipeline(self._scratch_a)
        except Exception:
            self._pipe = None

    def _bind(self) -> None:
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
        L.vkResetCommandPool.argtypes = [c_void_p, c_void_p, c_uint32]
        L.vkResetCommandPool.restype = c_int
        L.vkCreateBuffer.argtypes = [c_void_p, c_void_p, c_void_p, POINTER(c_void_p)]
        L.vkCreateBuffer.restype = c_int
        L.vkGetBufferMemoryRequirements.argtypes = [c_void_p, c_void_p, c_void_p]
        L.vkAllocateMemory.argtypes = [c_void_p, c_void_p, c_void_p, POINTER(c_void_p)]
        L.vkAllocateMemory.restype = c_int
        L.vkBindBufferMemory.argtypes = [c_void_p, c_void_p, c_void_p, c_uint64]
        L.vkBindBufferMemory.restype = c_int
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
        L.vkCmdBindPipeline.argtypes = [c_void_p, c_uint32, c_void_p]
        L.vkCmdBindDescriptorSets.argtypes = [c_void_p, c_uint32, c_void_p, c_uint32, c_uint32, POINTER(c_void_p), c_uint32, c_void_p]
        L.vkCmdDispatch.argtypes = [c_void_p, c_uint32, c_uint32, c_uint32]
        L.vkCmdCopyBuffer.argtypes = [c_void_p, c_void_p, c_void_p, c_uint32, c_void_p]

    def _check(self, code: int, name: str) -> None:
        if code != VK_SUCCESS:
            raise RuntimeError(f"{name}={code}")

    def _inst_info(self):
        class App(ctypes.Structure):
            _fields_ = [("sType", c_uint32), ("pNext", c_void_p), ("pApp", c_char_p), ("appVer", c_uint32), ("pEng", c_char_p), ("engVer", c_uint32), ("api", c_uint32)]

        class Info(ctypes.Structure):
            _fields_ = [("sType", c_uint32), ("pNext", c_void_p), ("flags", c_uint32), ("pApp", POINTER(App)), ("nLayer", c_uint32), ("ppLayer", POINTER(c_char_p)), ("nExt", c_uint32), ("ppExt", POINTER(c_char_p))]

        app = App(0, None, b"CoreForge", 1, b"cf", 1, API_1_2)
        info = Info(1, None, 0, ctypes.pointer(app), 0, None, 0, None)
        self._keep += [app, info]
        return ctypes.pointer(info)

    def _gpu_name(self) -> str:
        class P(ctypes.Structure):
            _fields_ = [("api", c_uint32), ("drv", c_uint32), ("ven", c_uint32), ("dev", c_uint32), ("dt", c_uint32), ("name", ctypes.c_char * 256), ("rest", ctypes.c_ubyte * 560)]
        p = P()
        self.lib.vkGetPhysicalDeviceProperties(self._phys, byref(p))
        return p.name.decode("utf-8", "replace")

    def _queue_family(self) -> int:
        n = c_uint32()
        self.lib.vkGetPhysicalDeviceQueueFamilyProperties(self._phys, byref(n), None)

        class Q(ctypes.Structure):
            _fields_ = [("flags", c_uint32), ("count", c_uint32), ("ts", c_uint32), ("minw", c_uint32), ("minh", c_uint32), ("mind", c_uint32)]

        props = (Q * n.value)()
        self.lib.vkGetPhysicalDeviceQueueFamilyProperties(self._phys, byref(n), props)
        for i, q in enumerate(props):
            if q.flags & VK_QUEUE_COMPUTE:
                return i
        raise RuntimeError("нет compute queue")

    def _dev_info(self):
        class Q(ctypes.Structure):
            _fields_ = [("sType", c_uint32), ("pNext", c_void_p), ("flags", c_uint32), ("family", c_uint32), ("count", c_uint32), ("prios", POINTER(c_float))]

        class D(ctypes.Structure):
            _fields_ = [("sType", c_uint32), ("pNext", c_void_p), ("flags", c_uint32), ("nq", c_uint32), ("pq", POINTER(Q)), ("nl", c_uint32), ("pl", POINTER(c_char_p)), ("ne", c_uint32), ("pe", POINTER(c_char_p)), ("feat", c_void_p)]

        prio = c_float(1.0)
        qi = Q(2, None, 0, self._qfam, 1, ctypes.pointer(prio))
        di = D(3, None, 0, 1, ctypes.pointer(qi), 0, None, 0, None, None)
        self._keep += [prio, qi, di]
        return ctypes.pointer(di)

    def _cmd_pool(self):
        class I(ctypes.Structure):
            _fields_ = [("sType", c_uint32), ("pNext", c_void_p), ("flags", c_uint32), ("family", c_uint32)]
        pool = c_void_p()
        self._check(self.lib.vkCreateCommandPool(self._device, byref(I(39, None, 3, self._qfam)), None, byref(pool)), "pool")
        return pool

    def _alloc_cmd(self):
        class I(ctypes.Structure):
            _fields_ = [("sType", c_uint32), ("pNext", c_void_p), ("pool", c_void_p), ("level", c_uint32), ("count", c_uint32)]
        cmd = c_void_p()
        self._check(self.lib.vkAllocateCommandBuffers(self._device, byref(I(40, None, self._pool, 0, 1)), byref(cmd)), "cmd")
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
        for i in range(p.nType):
            if (bits & (1 << i)) and (p.types[i].prop & flags) == flags:
                return i
        for i in range(p.nType):
            if bits & (1 << i):
                return i
        raise RuntimeError("нет memory type")

    def _mk_buffer(self, size: int, usage: int, local: bool = True):
        class B(ctypes.Structure):
            _fields_ = [("sType", c_uint32), ("pNext", c_void_p), ("flags", c_uint32), ("size", c_uint64), ("usage", c_uint32), ("share", c_uint32), ("nqf", c_uint32), ("pqf", c_void_p)]

        class R(ctypes.Structure):
            _fields_ = [("size", c_uint64), ("align", c_uint64), ("bits", c_uint32)]

        class A(ctypes.Structure):
            _fields_ = [("sType", c_uint32), ("pNext", c_void_p), ("size", c_uint64), ("type", c_uint32)]

        buf = c_void_p()
        self._check(self.lib.vkCreateBuffer(self._device, byref(B(12, None, 0, size, usage, 0, 0, None)), None, byref(buf)), "buf")
        req = R()
        self.lib.vkGetBufferMemoryRequirements(self._device, buf, byref(req))
        flags = VK_MEMORY_DEVICE_LOCAL if local else (VK_MEMORY_HOST_VISIBLE | VK_MEMORY_HOST_COHERENT)
        mem = c_void_p()
        self._check(self.lib.vkAllocateMemory(self._device, byref(A(5, None, req.size, self._mem_type(req.bits, flags))), None, byref(mem)), "mem")
        self._check(self.lib.vkBindBufferMemory(self._device, buf, mem, 0), "bind")
        return (buf, mem), size

    def alloc_blocks(self, want: int) -> int:
        chunk = 256 * 1024 * 1024
        got = 0
        while got < want:
            size = min(chunk, want - got)
            if size < 4 * 1024 * 1024:
                break
            try:
                pair, _ = self._mk_buffer(size, VK_BUFFER_STORAGE | VK_BUFFER_TRANSFER_SRC | VK_BUFFER_TRANSFER_DST)
            except Exception:
                size //= 2
                if size < 4 * 1024 * 1024:
                    break
                continue
            self.blocks.append((pair, size))
            got += size
        return got

    def _pipeline(self, ssbo):
        spv = _compute_spv()
        class SM(ctypes.Structure):
            _fields_ = [("sType", c_uint32), ("pNext", c_void_p), ("flags", c_uint32), ("size", ctypes.c_size_t), ("code", c_void_p)]
        raw = ctypes.create_string_buffer(spv, len(spv))
        self._keep.append(raw)
        module = c_void_p()
        self._check(self.lib.vkCreateShaderModule(self._device, byref(SM(16, None, 0, len(spv), ctypes.addressof(raw))), None, byref(module)), "shader")

        class Bind(ctypes.Structure):
            _fields_ = [("bind", c_uint32), ("type", c_uint32), ("count", c_uint32), ("stage", c_uint32), ("immut", c_void_p)]
        b = Bind(0, 7, 1, VK_SHADER_COMPUTE, None)

        class LI(ctypes.Structure):
            _fields_ = [("sType", c_uint32), ("pNext", c_void_p), ("flags", c_uint32), ("n", c_uint32), ("pb", c_void_p)]
        dsl = c_void_p()
        self._check(self.lib.vkCreateDescriptorSetLayout(self._device, byref(LI(32, None, 0, 1, ctypes.addressof(b))), None, byref(dsl)), "dsl")
        self._keep.append(b)

        class PL(ctypes.Structure):
            _fields_ = [("sType", c_uint32), ("pNext", c_void_p), ("flags", c_uint32), ("nset", c_uint32), ("pset", POINTER(c_void_p)), ("npush", c_uint32), ("ppush", c_void_p)]
        ds = c_void_p(dsl.value)
        self._keep.append(ds)
        layout = c_void_p()
        self._check(self.lib.vkCreatePipelineLayout(self._device, byref(PL(30, None, 0, 1, ctypes.pointer(ds), 0, None)), None, byref(layout)), "layout")

        class Stage(ctypes.Structure):
            _fields_ = [("sType", c_uint32), ("pNext", c_void_p), ("flags", c_uint32), ("stage", c_uint32), ("module", c_void_p), ("name", c_char_p), ("spec", c_void_p)]

        class CPI(ctypes.Structure):
            _fields_ = [("sType", c_uint32), ("pNext", c_void_p), ("flags", c_uint32), ("stage", Stage), ("layout", c_void_p), ("base", c_void_p), ("baseIndex", c_int)]

        name = ctypes.create_string_buffer(b"main")
        self._keep.append(name)
        st = Stage(18, None, 0, VK_SHADER_COMPUTE, module, ctypes.cast(name, c_char_p), None)
        cpi = CPI(29, None, 0, st, layout, None, -1)
        pipe = c_void_p()
        self._check(self.lib.vkCreateComputePipelines(self._device, None, 1, byref(cpi), None, byref(pipe)), "pipe")

        class Sz(ctypes.Structure):
            _fields_ = [("type", c_uint32), ("count", c_uint32)]
        sz = Sz(7, 1)

        class DP(ctypes.Structure):
            _fields_ = [("sType", c_uint32), ("pNext", c_void_p), ("flags", c_uint32), ("max", c_uint32), ("n", c_uint32), ("ps", c_void_p)]
        pool = c_void_p()
        self._check(self.lib.vkCreateDescriptorPool(self._device, byref(DP(33, None, 0, 1, 1, ctypes.addressof(sz))), None, byref(pool)), "dpool")
        self._keep.append(sz)

        class DA(ctypes.Structure):
            _fields_ = [("sType", c_uint32), ("pNext", c_void_p), ("pool", c_void_p), ("n", c_uint32), ("pl", POINTER(c_void_p))]
        dset = c_void_p()
        self._check(self.lib.vkAllocateDescriptorSets(self._device, byref(DA(34, None, pool, 1, ctypes.pointer(ds))), byref(dset)), "dset")

        class Buf(ctypes.Structure):
            _fields_ = [("buffer", c_void_p), ("off", c_uint64), ("range", c_uint64)]

        class W(ctypes.Structure):
            _fields_ = [("sType", c_uint32), ("pNext", c_void_p), ("dst", c_void_p), ("bind", c_uint32), ("arr", c_uint32), ("count", c_uint32), ("type", c_uint32), ("pimg", c_void_p), ("pbuf", c_void_p), ("ptex", c_void_p)]

        bi = Buf(ssbo[0], 0, 0xFFFFFFFFFFFFFFFF)
        wr = W(35, None, dset, 0, 0, 1, 7, None, ctypes.addressof(bi), None)
        self._keep += [bi, wr]
        self.lib.vkUpdateDescriptorSets(self._device, 1, byref(wr), 0, None)
        return pipe, layout, dset

    def _submit(self, rec) -> None:
        class B(ctypes.Structure):
            _fields_ = [("sType", c_uint32), ("pNext", c_void_p), ("flags", c_uint32), ("inh", c_void_p)]

        class S(ctypes.Structure):
            _fields_ = [("sType", c_uint32), ("pNext", c_void_p), ("wn", c_uint32), ("w", c_void_p), ("m", c_void_p), ("cn", c_uint32), ("c", POINTER(c_void_p)), ("sn", c_uint32), ("s", c_void_p)]

        self._check(self.lib.vkResetCommandPool(self._device, self._pool, 0), "reset")
        self._check(self.lib.vkBeginCommandBuffer(self._cmd, byref(B(42, None, 1, None))), "begin")
        rec()
        self._check(self.lib.vkEndCommandBuffer(self._cmd), "end")
        cmd = self._cmd
        s = S(4, None, 0, None, None, 1, ctypes.pointer(cmd), 0, None)
        self._check(self.lib.vkQueueSubmit(self._queue, 1, byref(s), None), "submit")
        self._check(self.lib.vkQueueWaitIdle(self._queue), "idle")

    def dispatch(self, groups: int) -> None:
        if not self._pipe:
            self.copy_scratch()
            return
        dset = c_void_p(self._dset.value)

        def rec() -> None:
            self.lib.vkCmdBindPipeline(self._cmd, 1, self._pipe)
            self.lib.vkCmdBindDescriptorSets(self._cmd, 1, self._layout, 0, 1, byref(dset), 0, None)
            self.lib.vkCmdDispatch(self._cmd, groups, 1, 1)

        self._submit(rec)

    def copy_pair(self, a, b) -> int:
        (buf_a, _), size_a = a
        (buf_b, _), size_b = b
        n = min(size_a, size_b)

        class R(ctypes.Structure):
            _fields_ = [("src", c_uint64), ("dst", c_uint64), ("size", c_uint64)]

        rg = R(0, 0, n)

        def rec() -> None:
            self.lib.vkCmdCopyBuffer(self._cmd, buf_a, buf_b, 1, byref(rg))

        self._submit(rec)
        return n

    def copy_scratch(self) -> int:
        (a, _), sa = self._scratch_a, self._scratch_sz
        (b, _), _sb = self._scratch_b

        class R(ctypes.Structure):
            _fields_ = [("src", c_uint64), ("dst", c_uint64), ("size", c_uint64)]

        rg = R(0, 0, sa)

        def rec() -> None:
            self.lib.vkCmdCopyBuffer(self._cmd, a, b, 1, byref(rg))

        self._submit(rec)
        return sa

    def close(self) -> None:
        try:
            self.lib.vkDestroyDevice(self._device, None)
            self.lib.vkDestroyInstance(self._instance, None)
        except Exception:
            pass
