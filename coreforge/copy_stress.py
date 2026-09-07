"""Нагрузка Copy-движка диспетчера: DMA D2D + pinned H2D/D2H, не compute-ядра."""

from __future__ import annotations

import time
from typing import Callable

from coreforge.cuda_api import Cuda, CudaError


MB = 1024 * 1024


class CopyStress:
    def __init__(self, stop_event, intensity: int, on_log: Callable[[str], None], reserved_bytes: int = 0) -> None:
        self.stop = stop_event
        self.intensity = intensity
        self.on_log = on_log
        self.reserved_bytes = reserved_bytes
        self._alive = True

    def request_close(self) -> None:
        self._alive = False

    def run(self) -> None:
        try:
            self._run()
        except Exception as exc:
            self.on_log(f"Copy-движок: {exc}")
            while self._alive and not self.stop.is_set():
                time.sleep(0.3)

    def _run(self) -> None:
        cuda = Cuda()
        dev_a = dev_b = host_a = host_b = 0
        s1 = s2 = s3 = 0
        try:
            dev_bytes = 0
            host_bytes = 0
            room = max(2 * MB, 512 * MB - max(0, self.reserved_bytes // 32))
            sizes = [s for s in (32 * MB, 16 * MB, 8 * MB, 4 * MB, 2 * MB, 1 * MB) if s <= room]
            for size in sizes or (1 * MB,):
                try:
                    dev_a = cuda.alloc(size)
                    dev_b = cuda.alloc(size)
                    cuda.memset(dev_a, 0xA5, size)
                    cuda.memset(dev_b, 0x5A, size)
                    dev_bytes = size
                    break
                except CudaError:
                    if dev_a:
                        try:
                            cuda.free(dev_a)
                        except CudaError:
                            pass
                    if dev_b:
                        try:
                            cuda.free(dev_b)
                        except CudaError:
                            pass
                    dev_a = dev_b = 0
            if not dev_bytes:
                raise RuntimeError("нет свободной VRAM для DMA")

            for size in (16 * MB, 8 * MB, 4 * MB, 2 * MB, 1 * MB):
                try:
                    host_a = cuda.alloc_host(size)
                    host_b = cuda.alloc_host(size)
                    host_bytes = size
                    break
                except CudaError:
                    if host_a:
                        cuda.free_host(host_a)
                    if host_b:
                        cuda.free_host(host_b)
                    host_a = host_b = 0

            s1 = cuda.stream_async()
            s2 = cuda.stream_async() if host_bytes else 0
            s3 = cuda.stream_async()
            reps = max(12, self.intensity // 4)
            self.on_log(f"Copy DMA: {dev_bytes // MB} МБ D2D, {host_bytes // MB} МБ H2D/D2H")
            while self._alive and not self.stop.is_set():
                cuda.make_current()
                for _ in range(reps):
                    cuda.memcpy_dto_d_async(dev_b, dev_a, dev_bytes, s1)
                    cuda.memcpy_dto_d_async(dev_a, dev_b, dev_bytes, s1)
                    cuda.memcpy_dto_d_async(dev_b, dev_a, dev_bytes, s3)
                if host_bytes and s2:
                    cuda.memcpy_dtoh_async(host_a, dev_a, min(host_bytes, dev_bytes), s2)
                    cuda.memcpy_htod_async(dev_b, host_b, min(host_bytes, dev_bytes), s2)
                cuda.sync_stream(s1)
                cuda.sync_stream(s3)
                if s2:
                    cuda.sync_stream(s2)
        finally:
            if s1:
                cuda.destroy_stream(s1)
            if s2:
                cuda.destroy_stream(s2)
            if s3:
                cuda.destroy_stream(s3)
            if host_a:
                cuda.free_host(host_a)
            if host_b:
                cuda.free_host(host_b)
            if dev_a:
                try:
                    cuda.free(dev_a)
                except CudaError:
                    pass
            if dev_b:
                try:
                    cuda.free(dev_b)
                except CudaError:
                    pass
            cuda.close()
