"""Обёртка над CUDA Driver API (nvcuda.dll) — без CUDA Toolkit и PyTorch."""

from __future__ import annotations

import ctypes
from ctypes import (
    POINTER,
    c_char,
    c_char_p,
    c_int,
    c_size_t,
    c_uint,
    c_uint32,
    c_uint64,
    c_void_p,
    create_string_buffer,
)
from typing import Any


CUDA_SUCCESS = 0
CU_CTX_SCHED_AUTO = 0
CU_CTX_SCHED_SPIN = 1
CU_STREAM_NON_BLOCKING = 1
CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16
CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75
CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76
CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 1
CU_DEVICE_ATTRIBUTE_WARP_SIZE = 10
CU_DEVICE_ATTRIBUTE_CLOCK_RATE = 13
CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE = 36
CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH = 37
CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE = 38
CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK = 8

CU_JIT_ERROR_LOG_BUFFER = 5
CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES = 6
CU_JIT_INFO_LOG_BUFFER = 3
CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES = 4
CU_JIT_OPTIMIZATION_LEVEL = 7
CU_JIT_TARGET_FROM_CUCONTEXT = 8


class CudaError(RuntimeError):
    def __init__(self, code: int, api: str, extra: str = "") -> None:
        self.code = code
        self.api = api
        msg = f"{api} вернул код {code}"
        if extra:
            msg = f"{msg}: {extra}"
        super().__init__(msg)


def _load_nvcuda() -> ctypes.WinDLL:
    names = ("nvcuda.dll",)
    last: Exception | None = None
    for name in names:
        try:
            lib = ctypes.WinDLL(name)
            return lib
        except OSError as exc:
            last = exc
    raise CudaError(-1, "LoadLibrary", f"nvcuda.dll не найдена ({last})")


class Cuda:
    def __init__(self) -> None:
        self.lib = _load_nvcuda()
        self._bind()
        self._check(self.lib.cuInit(0), "cuInit")

        self.device = c_int()
        self._check(self.lib.cuDeviceGet(ctypes.byref(self.device), 0), "cuDeviceGet")

        self.context = c_void_p()
        self._check(
            self.lib.cuCtxCreate_v2(ctypes.byref(self.context), CU_CTX_SCHED_SPIN, self.device),
            "cuCtxCreate",
        )
        self._prio_lo = 0
        self._prio_hi = 0
        try:
            lo = c_int()
            hi = c_int()
            self._check(
                self.lib.cuCtxGetStreamPriorityRange(ctypes.byref(lo), ctypes.byref(hi)),
                "cuCtxGetStreamPriorityRange",
            )
            self._prio_lo = int(lo.value)
            self._prio_hi = int(hi.value)
        except CudaError:
            pass

        self.name = self._device_name()
        self.total_mem = self._total_mem()
        self.sm_count = self.device_attr(CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
        self.cc_major = self.device_attr(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
        self.cc_minor = self.device_attr(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
        self.warp_size = self.device_attr(CU_DEVICE_ATTRIBUTE_WARP_SIZE)
        self.clock_khz = self.device_attr(CU_DEVICE_ATTRIBUTE_CLOCK_RATE)
        self.mem_clock_khz = self.device_attr(CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE)
        self.bus_width = self.device_attr(CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH)
        self.l2_cache = self.device_attr(CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE)
        self.driver_version = self._driver_version()
        self.arch = f"sm_{self.cc_major}{self.cc_minor}"
        self.has_tensor = self.cc_major >= 7
        self.has_fp8_tensor = (self.cc_major, self.cc_minor) >= (8, 9)

    def _bind(self) -> None:
        L = self.lib
        L.cuInit.argtypes = [c_uint]
        L.cuInit.restype = c_int
        L.cuDeviceGet.argtypes = [POINTER(c_int), c_int]
        L.cuDeviceGet.restype = c_int
        L.cuDeviceGetName.argtypes = [c_char_p, c_int, c_int]
        L.cuDeviceGetName.restype = c_int
        L.cuDeviceTotalMem_v2.argtypes = [POINTER(c_size_t), c_int]
        L.cuDeviceTotalMem_v2.restype = c_int
        L.cuDeviceGetAttribute.argtypes = [POINTER(c_int), c_int, c_int]
        L.cuDeviceGetAttribute.restype = c_int
        L.cuDriverGetVersion.argtypes = [POINTER(c_int)]
        L.cuDriverGetVersion.restype = c_int
        L.cuCtxCreate_v2.argtypes = [POINTER(c_void_p), c_uint, c_int]
        L.cuCtxCreate_v2.restype = c_int
        L.cuCtxDestroy_v2.argtypes = [c_void_p]
        L.cuCtxDestroy_v2.restype = c_int
        L.cuCtxSetCurrent.argtypes = [c_void_p]
        L.cuCtxSetCurrent.restype = c_int
        L.cuCtxSynchronize.argtypes = []
        L.cuCtxSynchronize.restype = c_int
        L.cuMemAlloc_v2.argtypes = [POINTER(c_uint64), c_size_t]
        L.cuMemAlloc_v2.restype = c_int
        L.cuMemFree_v2.argtypes = [c_uint64]
        L.cuMemFree_v2.restype = c_int
        L.cuMemsetD8_v2.argtypes = [c_uint64, ctypes.c_ubyte, c_size_t]
        L.cuMemsetD8_v2.restype = c_int
        L.cuMemcpyDtoD_v2.argtypes = [c_uint64, c_uint64, c_size_t]
        L.cuMemcpyDtoD_v2.restype = c_int
        L.cuMemcpyDtoDAsync_v2.argtypes = [c_uint64, c_uint64, c_size_t, c_void_p]
        L.cuMemcpyDtoDAsync_v2.restype = c_int
        L.cuMemcpyHtoDAsync_v2.argtypes = [c_uint64, c_void_p, c_size_t, c_void_p]
        L.cuMemcpyHtoDAsync_v2.restype = c_int
        L.cuMemcpyDtoHAsync_v2.argtypes = [c_void_p, c_uint64, c_size_t, c_void_p]
        L.cuMemcpyDtoHAsync_v2.restype = c_int
        L.cuMemAllocHost_v2.argtypes = [POINTER(c_void_p), c_size_t]
        L.cuMemAllocHost_v2.restype = c_int
        L.cuMemFreeHost.argtypes = [c_void_p]
        L.cuMemFreeHost.restype = c_int
        L.cuModuleLoadData.argtypes = [POINTER(c_void_p), c_void_p]
        L.cuModuleLoadData.restype = c_int
        L.cuModuleLoadDataEx.argtypes = [
            POINTER(c_void_p),
            c_void_p,
            c_uint,
            POINTER(c_int),
            POINTER(c_void_p),
        ]
        L.cuModuleLoadDataEx.restype = c_int
        L.cuModuleGetFunction.argtypes = [POINTER(c_void_p), c_void_p, c_char_p]
        L.cuModuleGetFunction.restype = c_int
        L.cuModuleUnload.argtypes = [c_void_p]
        L.cuModuleUnload.restype = c_int
        L.cuLaunchKernel.argtypes = [
            c_void_p,
            c_uint,
            c_uint,
            c_uint,
            c_uint,
            c_uint,
            c_uint,
            c_uint,
            c_void_p,
            POINTER(c_void_p),
            POINTER(c_void_p),
        ]
        L.cuLaunchKernel.restype = c_int
        L.cuStreamCreate.argtypes = [POINTER(c_void_p), c_uint]
        L.cuStreamCreate.restype = c_int
        L.cuStreamCreateWithPriority.argtypes = [POINTER(c_void_p), c_uint, c_int]
        L.cuStreamCreateWithPriority.restype = c_int
        L.cuCtxGetStreamPriorityRange.argtypes = [POINTER(c_int), POINTER(c_int)]
        L.cuCtxGetStreamPriorityRange.restype = c_int
        L.cuStreamDestroy_v2.argtypes = [c_void_p]
        L.cuStreamDestroy_v2.restype = c_int
        L.cuStreamSynchronize.argtypes = [c_void_p]
        L.cuStreamSynchronize.restype = c_int
        L.cuEventCreate.argtypes = [POINTER(c_void_p), c_uint]
        L.cuEventCreate.restype = c_int
        L.cuEventDestroy_v2.argtypes = [c_void_p]
        L.cuEventDestroy_v2.restype = c_int
        L.cuEventRecord.argtypes = [c_void_p, c_void_p]
        L.cuEventRecord.restype = c_int
        L.cuEventSynchronize.argtypes = [c_void_p]
        L.cuEventSynchronize.restype = c_int
        L.cuEventElapsedTime.argtypes = [POINTER(ctypes.c_float), c_void_p, c_void_p]
        L.cuEventElapsedTime.restype = c_int
        L.cuGetErrorString.argtypes = [c_int, POINTER(c_char_p)]
        L.cuGetErrorString.restype = c_int

    def _check(self, code: int, api: str, extra: str = "") -> None:
        if code != CUDA_SUCCESS:
            text = self.error_string(code)
            raise CudaError(code, api, extra or text)

    def error_string(self, code: int) -> str:
        ptr = c_char_p()
        if self.lib.cuGetErrorString(code, ctypes.byref(ptr)) == CUDA_SUCCESS and ptr.value:
            return ptr.value.decode("utf-8", errors="replace")
        return f"CUDA error {code}"

    def make_current(self) -> None:
        self._check(self.lib.cuCtxSetCurrent(self.context), "cuCtxSetCurrent")

    def _device_name(self) -> str:
        buf = create_string_buffer(256)
        self._check(self.lib.cuDeviceGetName(buf, 256, self.device), "cuDeviceGetName")
        return buf.value.decode("utf-8", errors="replace")

    def _total_mem(self) -> int:
        size = c_size_t()
        self._check(self.lib.cuDeviceTotalMem_v2(ctypes.byref(size), self.device), "cuDeviceTotalMem")
        return int(size.value)

    def _driver_version(self) -> str:
        ver = c_int()
        self._check(self.lib.cuDriverGetVersion(ctypes.byref(ver)), "cuDriverGetVersion")
        major, minor = divmod(ver.value, 1000)
        return f"{major}.{minor // 10}"

    def device_attr(self, attr: int) -> int:
        value = c_int()
        self._check(
            self.lib.cuDeviceGetAttribute(ctypes.byref(value), attr, self.device),
            "cuDeviceGetAttribute",
        )
        return int(value.value)

    def alloc(self, nbytes: int) -> int:
        ptr = c_uint64()
        self._check(self.lib.cuMemAlloc_v2(ctypes.byref(ptr), nbytes), "cuMemAlloc")
        return int(ptr.value)

    def free(self, ptr: int) -> None:
        if ptr:
            self._check(self.lib.cuMemFree_v2(ptr), "cuMemFree")

    def memset(self, ptr: int, value: int, nbytes: int) -> None:
        self._check(self.lib.cuMemsetD8_v2(ptr, value, nbytes), "cuMemsetD8")

    def memcpy_dto_d(self, dst: int, src: int, nbytes: int) -> None:
        self._check(self.lib.cuMemcpyDtoD_v2(dst, src, nbytes), "cuMemcpyDtoD")

    def memcpy_dto_d_async(self, dst: int, src: int, nbytes: int, stream: int) -> None:
        self._check(
            self.lib.cuMemcpyDtoDAsync_v2(dst, src, nbytes, c_void_p(stream) if stream else None),
            "cuMemcpyDtoDAsync",
        )

    def memcpy_htod_async(self, dst: int, src: int, nbytes: int, stream: int) -> None:
        self._check(
            self.lib.cuMemcpyHtoDAsync_v2(dst, c_void_p(src), nbytes, c_void_p(stream) if stream else None),
            "cuMemcpyHtoDAsync",
        )

    def memcpy_dtoh_async(self, dst: int, src: int, nbytes: int, stream: int) -> None:
        self._check(
            self.lib.cuMemcpyDtoHAsync_v2(c_void_p(dst), src, nbytes, c_void_p(stream) if stream else None),
            "cuMemcpyDtoHAsync",
        )

    def alloc_host(self, nbytes: int) -> int:
        ptr = c_void_p()
        self._check(self.lib.cuMemAllocHost_v2(ctypes.byref(ptr), nbytes), "cuMemAllocHost")
        return ptr.value or 0

    def free_host(self, ptr: int) -> None:
        if ptr:
            self.lib.cuMemFreeHost(c_void_p(ptr))

    def stream_async(self) -> int:
        s = c_void_p()
        self._check(self.lib.cuStreamCreate(ctypes.byref(s), 1), "cuStreamCreate")
        return s.value or 0

    def load_ptx(self, ptx: str) -> int:
        module = c_void_p()
        data = create_string_buffer(ptx.encode("utf-8"))
        code = self.lib.cuModuleLoadData(ctypes.byref(module), data)
        if code != CUDA_SUCCESS:
            err_buf = create_string_buffer(8192)
            err_size = c_uint(8192)
            opt_level = c_uint(4)
            options = (c_int * 3)(
                CU_JIT_ERROR_LOG_BUFFER,
                CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
                CU_JIT_OPTIMIZATION_LEVEL,
            )
            values = (c_void_p * 3)(
                ctypes.cast(err_buf, c_void_p),
                ctypes.cast(ctypes.pointer(err_size), c_void_p),
                ctypes.cast(ctypes.pointer(opt_level), c_void_p),
            )
            code2 = self.lib.cuModuleLoadDataEx(
                ctypes.byref(module), data, 3, options, values
            )
            log = err_buf.value.decode("utf-8", errors="replace").strip()
            raise CudaError(code2 or code, "cuModuleLoadData", log or self.error_string(code))
        return module.value or 0

    def get_function(self, module: int, name: str) -> int:
        fn = c_void_p()
        self._check(
            self.lib.cuModuleGetFunction(ctypes.byref(fn), c_void_p(module), name.encode("utf-8")),
            "cuModuleGetFunction",
            name,
        )
        return fn.value or 0

    def unload(self, module: int) -> None:
        if module:
            self.lib.cuModuleUnload(c_void_p(module))

    def stream(self, high_priority: bool = False) -> int:
        s = c_void_p()
        if high_priority and hasattr(self.lib, "cuStreamCreateWithPriority"):
            try:
                self._check(
                    self.lib.cuStreamCreateWithPriority(
                        ctypes.byref(s), CU_STREAM_NON_BLOCKING, self._prio_hi
                    ),
                    "cuStreamCreateWithPriority",
                )
                return s.value or 0
            except CudaError:
                pass
        self._check(self.lib.cuStreamCreate(ctypes.byref(s), 0), "cuStreamCreate")
        return s.value or 0

    def destroy_stream(self, stream: int) -> None:
        if stream:
            self.lib.cuStreamDestroy_v2(c_void_p(stream))

    def sync_stream(self, stream: int) -> None:
        self._check(self.lib.cuStreamSynchronize(c_void_p(stream) if stream else None), "cuStreamSynchronize")

    def launch(
        self,
        fn: int,
        grid: tuple[int, int, int],
        block: tuple[int, int, int],
        args: list[Any],
        stream: int = 0,
    ) -> list[Any]:
        """args — список ctypes-значений; ссылки нужно держать до sync."""
        keep = list(args)
        pack = (c_void_p * len(keep))()
        for i, value in enumerate(keep):
            pack[i] = ctypes.cast(ctypes.pointer(value), c_void_p)
        self._check(
            self.lib.cuLaunchKernel(
                c_void_p(fn),
                grid[0],
                grid[1],
                grid[2],
                block[0],
                block[1],
                block[2],
                0,
                c_void_p(stream) if stream else None,
                pack,
                None,
            ),
            "cuLaunchKernel",
        )
        return keep

    def timed_ms(self, stream: int, fn) -> float:
        start = c_void_p()
        stop = c_void_p()
        self._check(self.lib.cuEventCreate(ctypes.byref(start), 0), "cuEventCreate")
        self._check(self.lib.cuEventCreate(ctypes.byref(stop), 0), "cuEventCreate")
        try:
            self._check(self.lib.cuEventRecord(start, c_void_p(stream) if stream else None), "cuEventRecord")
            fn()
            self._check(self.lib.cuEventRecord(stop, c_void_p(stream) if stream else None), "cuEventRecord")
            self._check(self.lib.cuEventSynchronize(stop), "cuEventSynchronize")
            ms = ctypes.c_float()
            self._check(self.lib.cuEventElapsedTime(ctypes.byref(ms), start, stop), "cuEventElapsedTime")
            return float(ms.value)
        finally:
            self.lib.cuEventDestroy_v2(start)
            self.lib.cuEventDestroy_v2(stop)

    def close(self) -> None:
        if getattr(self, "context", None):
            self.lib.cuCtxDestroy_v2(self.context)
            self.context = None
