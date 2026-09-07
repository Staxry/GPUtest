"""Оркестрация нагрузки: вместе / по очереди, лимит температуры, калибровка ядер."""

from __future__ import annotations

import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Callable

from ctypes import c_uint32, c_uint64

from coreforge.cuda_api import Cuda, CudaError
from coreforge.nvml_api import GpuSample
from coreforge.ptx import (
    ALU_FLOPS_PER_ITER,
    FLOPS_F16_WARP,
    FLOPS_FP8_WARP,
    FLOPS_TF32_WARP,
    build_ptx,
)
from coreforge.sensors import Sensors
from coreforge.vendor import GpuCaps, detect


GB = 1024**3
MB = 1024**2
HEADROOM = 900 * MB
CHUNK = 256 * MB
TARGET_MS = 70.0


@dataclass
class BurnConfig:
    together: bool = True
    cuda_cores: bool = True
    tensor: bool = True
    shaders: bool = True
    rt_cores: bool = True
    vram: bool = True
    copy_engine: bool = True
    nvenc: bool = True
    nvdec: bool = True
    vram_gb: float = 6.0
    intensity: int = 90
    temp_limit: int = 83
    duration_sec: int = 0
    sequential_sec: int = 30
    shader_width: int = 1920
    shader_height: int = 1080
    tensor_mode: str = "all"  # f16 / tf32 / fp8 / all
    fsr_mode: str = "Выкл"
    render_choice: str = "Авто"
    vsync: bool = False


@dataclass
class BurnStats:
    running: bool = False
    stage: str = "ожидание"
    message: str = ""
    stopped_reason: str = ""
    alu_tflops: float = 0.0
    tensor_tflops: float = 0.0
    vram_gbs: float = 0.0
    shader_fps: float = 0.0
    rt_mrays: float = 0.0
    allocated_bytes: int = 0
    elapsed: float = 0.0
    sample: GpuSample = field(default_factory=GpuSample)


class BurnEngine:
    def __init__(self) -> None:
        self.caps: GpuCaps = detect()
        self.nvml = Sensors(self.caps)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._gl_thread: threading.Thread | None = None
        self._rt_thread: threading.Thread | None = None
        self._media_thread: threading.Thread | None = None
        self._media_proc = None
        self._copy_thread: threading.Thread | None = None
        self._vk_thread: threading.Thread | None = None
        self.stats = BurnStats()
        self.gpu_info: dict = {}
        self._shader_fps = 0.0
        self._rt_mrays = 0.0
        self._on_log: Callable[[str], None] | None = None
        self._sizes: dict[int, int] = {}
        self._shader_scene = "—"
        self._shader_native_fps = 0.0
        self._fsr_info = ""
        self._blocks_text = ""
        self._duration_sec = 0

    def probe(self) -> dict:
        caps = self.caps
        info = {
            "name": caps.name,
            "vram_total": caps.vram_total or 8 * 1024**3,
            "sm": 0,
            "cc": caps.vendor,
            "arch": caps.vendor,
            "driver": "Vulkan" if caps.has_vulkan else "?",
            "has_tensor": False,
            "has_fp8": False,
            "vendor": caps.vendor,
            "has_cuda": caps.has_cuda,
            "has_nvenc": caps.has_nvenc,
            "has_nvdec": caps.has_nvdec,
            "has_ray_query": caps.has_ray_query,
        }
        if caps.has_cuda:
            cuda = Cuda()
            try:
                info.update({
                    "name": cuda.name,
                    "vram_total": cuda.total_mem,
                    "sm": cuda.sm_count,
                    "cc": f"{cuda.cc_major}.{cuda.cc_minor}",
                    "arch": cuda.arch,
                    "driver": f"CUDA {cuda.driver_version}",
                    "has_tensor": cuda.has_tensor,
                    "has_fp8": cuda.has_fp8_tensor,
                    "clock_mhz": cuda.clock_khz / 1000.0,
                    "mem_clock_mhz": cuda.mem_clock_khz / 1000.0,
                    "vendor": "nvidia" if caps.vendor != "amd" else "amd",
                })
            finally:
                cuda.close()
        self.gpu_info = info
        return info

    def set_logger(self, fn: Callable[[str], None]) -> None:
        self._on_log = fn

    def log(self, text: str) -> None:
        if self._on_log:
            self._on_log(text)

    def snapshot(self) -> BurnStats:
        with self._lock:
            s = self.stats
            copy = BurnStats(
                running=s.running,
                stage=s.stage,
                message=s.message,
                stopped_reason=s.stopped_reason,
                alu_tflops=s.alu_tflops,
                tensor_tflops=s.tensor_tflops,
                vram_gbs=s.vram_gbs,
                shader_fps=self._shader_fps,
                rt_mrays=self._rt_mrays,
                allocated_bytes=s.allocated_bytes,
                elapsed=s.elapsed,
                sample=self.nvml.sample(),
            )
            return copy

    def start(self, config: BurnConfig) -> None:
        if self._thread and self._thread.is_alive():
            return
        if not self.caps.is_nvidia:
            config.cuda_cores = False
            config.tensor = False
            config.nvenc = False
            config.nvdec = False
        if not self.caps.has_ray_query and not self.caps.is_nvidia:
            config.rt_cores = False
        self._stop.clear()
        parts = []
        if config.cuda_cores:
            parts.append("CUDA")
        if config.tensor:
            parts.append("тензор")
        if config.shaders:
            parts.append("шейдер")
        if config.rt_cores:
            parts.append("RT")
        if config.vram:
            parts.append("VRAM")
        if config.copy_engine:
            parts.append("Copy")
        if config.nvenc:
            parts.append("NVENC")
        if config.nvdec:
            parts.append("NVDEC")
        self._blocks_text = " · ".join(parts) or "—"
        self._shader_scene = "—"
        self._duration_sec = int(config.duration_sec)
        with self._lock:
            self.stats = BurnStats(running=True, stage="запуск", message="подготовка блоков")
        self._thread = threading.Thread(target=self._run, args=(config,), daemon=True, name="coreforge-engine")
        self._thread.start()

    def stop(self, reason: str = "остановлено") -> None:
        self._stop.set()
        with self._lock:
            self.stats.stopped_reason = reason
            self.stats.message = reason
        self._halt_workers()

    def shutdown(self, reason: str = "выход") -> None:
        self.stop(reason)
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)
        self._halt_workers()

    def _halt_workers(self) -> None:
        try:
            self._stop_shaders()
            self._stop_rt()
            self._stop_copy()
            self._stop_media()
            self._stop_vk_compute()
        except Exception:
            pass

    def _set(self, **kwargs) -> None:
        with self._lock:
            for k, v in kwargs.items():
                setattr(self.stats, k, v)

    def _run(self, cfg: BurnConfig) -> None:
        cuda: Cuda | None = None
        module = 0
        ptrs: list[int] = []
        streams: list[int] = []
        started = 0.0
        fns: dict = {}
        scratch = 0
        grid = 1
        block = 256
        alu_iters = 0
        ten_iters = 0
        vram_ptrs: list[int] = []
        nvec = 0
        need_kernels = bool(cfg.cuda_cores or cfg.tensor or cfg.vram)
        try:
            if cfg.duration_sec <= 0:
                self.log("длительность: бесконечно, стоп вручную или по температуре")
            else:
                self.log(f"длительность: {cfg.duration_sec} с")
            if need_kernels:
                cuda = Cuda()
                self.log(f"CUDA: {cuda.name}  {cuda.arch}  {cuda.sm_count} SM  драйвер {cuda.driver_version}")
                ptx = build_ptx(cuda.arch, cuda.has_fp8_tensor)
                module = cuda.load_ptx(ptx)
                fns = {
                    "alu": cuda.get_function(module, "burn_alu"),
                    "f16": cuda.get_function(module, "burn_tensor_f16"),
                    "tf32": cuda.get_function(module, "burn_tensor_tf32"),
                    "copy": cuda.get_function(module, "copy_bw"),
                    "walk": cuda.get_function(module, "random_walk"),
                }
                if cuda.has_fp8_tensor:
                    try:
                        fns["fp8"] = cuda.get_function(module, "burn_tensor_fp8")
                    except CudaError as exc:
                        self.log(f"FP8 тензоры недоступны: {exc}")

                scratch = cuda.alloc(256 * 4)
                ptrs.append(scratch)
                intensity = max(10, min(100, cfg.intensity)) / 100.0
                grid = max(1, int(cuda.sm_count * 32 * intensity))
                self._set(stage="калибровка", message="подбор интенсивности ядер…")
                if cfg.cuda_cores:
                    alu_iters = self._calibrate(cuda, fns["alu"], scratch, grid, block, intensity, 8.0)
                ten_iters = 200
                if cfg.tensor and cuda.has_tensor:
                    ten_iters = self._calibrate(cuda, fns["f16"], scratch, grid, block, intensity, 4.0)
                if cfg.vram:
                    vram_ptrs, nvec = self._alloc_vram(cuda, cfg.vram_gb)
                    ptrs.extend(vram_ptrs)
                    self._set(allocated_bytes=sum(self._sizes.get(p, 0) for p in vram_ptrs))
                elif cfg.cuda_cores or cfg.tensor:
                    # без галочки VRAM всё равно молотим шину — иначе ватты сильно ниже TGP
                    vram_ptrs, nvec = self._alloc_vram(cuda, min(1.75, max(0.75, cfg.vram_gb * 0.2)))
                    ptrs.extend(vram_ptrs)
                    self._set(allocated_bytes=sum(self._sizes.get(p, 0) for p in vram_ptrs))
                    self.log("без галочки VRAM: служебный буфер для полной мощности")
            else:
                self.log("CUDA/тензор/VRAM выключены — калибровка ядер пропущена")
            if not self.caps.has_cuda and (cfg.cuda_cores or cfg.vram or cfg.copy_engine):
                self._start_vk_compute(cfg)
                self.log("AMD/Vulkan: compute + VRAM + Copy через Vulkan")

            stages: list[tuple[str, str]] = []
            if cfg.together:
                if cfg.cuda_cores and cuda:
                    stages.append(("alu", "CUDA-ядра + SFU + INT"))
                if cfg.tensor and cuda and cuda.has_tensor:
                    stages.append(("tensor", "тензорные ядра"))
                if len(vram_ptrs) >= 1 and (cfg.vram or cfg.cuda_cores or cfg.tensor):
                    stages.append(("vram", "видеопамять"))
                if cfg.copy_engine:
                    self._start_copy(cfg)
                    stages.append(("copy", "Copy-движок (DMA)"))
                if cfg.nvenc or cfg.nvdec:
                    self._start_media(cfg)
                    stages.append(("media", "NVENC/NVDEC"))
                if cfg.shaders:
                    self._start_shaders(cfg)
                    stages.append(("shaders", "шейдеры"))
                if cfg.rt_cores:
                    self._start_rt(cfg)
                    stages.append(("rt", "RT-ядра"))
                if not self.caps.has_cuda and (cfg.cuda_cores or cfg.vram or cfg.copy_engine):
                    stages.append(("vk", "Vulkan compute / VRAM"))
                self._set(stage="вместе", message=" + ".join(s[1] for s in stages) or "нет модулей")
                started = time.perf_counter()
                if cuda and stages and any(k in {"alu", "tensor", "vram"} for k, _ in stages):
                    self._loop_parallel(
                        cuda, fns, scratch, grid, block, alu_iters, ten_iters,
                        vram_ptrs, nvec, cfg, started, stages,
                    )
                else:
                    self._idle_wait(cfg, started)
            else:
                order: list[tuple[str, str]] = []
                if cfg.cuda_cores and cuda:
                    order.append(("alu", "CUDA-ядра + SFU + INT"))
                if cfg.tensor and cuda and cuda.has_tensor:
                    order.append(("tensor", "тензорные ядра"))
                if cfg.shaders:
                    order.append(("shaders", "шейдеры"))
                if cfg.rt_cores:
                    order.append(("rt", "RT-ядра"))
                if cfg.copy_engine:
                    order.append(("copy", "Copy-движок (DMA)"))
                if cfg.nvenc or cfg.nvdec:
                    order.append(("media", "NVENC/NVDEC"))
                if cfg.vram and vram_ptrs:
                    order.append(("vram", "видеопамять"))
                for key, title in order:
                    if self._stop.is_set() or self._expired(cfg, started) or self._overheat(cfg):
                        break
                    self._set(stage=title, message=f"раздельный режим: {title}")
                    self.log(f"этап: {title}")
                    if key == "shaders":
                        self._start_shaders(cfg)
                    elif key == "rt":
                        self._start_rt(cfg)
                    elif key == "copy":
                        self._start_copy(cfg)
                    elif key == "media":
                        self._start_media(cfg)
                    if started == 0.0:
                        started = time.perf_counter()
                    slice_end = time.perf_counter() + max(5, cfg.sequential_sec)
                    if cuda and key in {"alu", "tensor", "vram"}:
                        self._loop_parallel(
                            cuda, fns, scratch, grid, block, alu_iters, ten_iters,
                            vram_ptrs, nvec, cfg, started, [(key, title)],
                            until=slice_end,
                        )
                    else:
                        self._idle_wait(cfg, started, until=slice_end)
                    self._stop_shaders()
                    self._stop_rt()
                    self._stop_copy()
                    self._stop_media()
            if not self.stats.stopped_reason:
                self._set(stopped_reason="завершено")
        except Exception as exc:
            self.log(f"ошибка: {exc}")
            self._set(message=str(exc), stopped_reason="ошибка")
        finally:
            self._stop_shaders()
            self._stop_rt()
            self._stop_copy()
            self._stop_media()
            self._stop_vk_compute()
            if cuda:
                for s in streams:
                    cuda.destroy_stream(s)
                for p in ptrs:
                    try:
                        cuda.free(p)
                    except CudaError:
                        pass
                if module:
                    cuda.unload(module)
                cuda.close()
            self._set(running=False, stage="остановлено", allocated_bytes=0)
            self._shader_fps = 0.0
            self._rt_mrays = 0.0

    def _idle_wait(self, cfg: BurnConfig, started: float, until: float | None = None) -> None:
        while not self._stop.is_set():
            if self._expired(cfg, started) or self._overheat(cfg):
                break
            if until is not None and time.perf_counter() >= until:
                break
            self._set(elapsed=time.perf_counter() - started)
            time.sleep(0.2)

    def _expired(self, cfg: BurnConfig, started: float) -> bool:
        if cfg.duration_sec <= 0 or started <= 0:
            return False
        return (time.perf_counter() - started) >= cfg.duration_sec

    def _overheat(self, cfg: BurnConfig) -> bool:
        sample = self.nvml.sample()
        if sample.ok and sample.temp >= cfg.temp_limit:
            self.stop(f"лимит температуры {sample.temp}°C")
            return True
        return False

    def _calibrate(
        self,
        cuda: Cuda,
        fn: int,
        scratch: int,
        grid: int,
        block: int,
        intensity: float,
        flop_scale: float,
    ) -> int:
        out = c_uint64(scratch)
        probe = max(8192, int(24_000 * intensity))
        try:
            cuda.launch(fn, (grid, 1, 1), (block, 1, 1), [out, c_uint32(2048)], 0)
            cuda.sync_stream(0)
            ms = cuda.timed_ms(
                0,
                lambda: (
                    cuda.launch(fn, (grid, 1, 1), (block, 1, 1), [out, c_uint32(probe)], 0),
                    cuda.sync_stream(0),
                ),
            )
        except CudaError:
            return probe
        if ms < 0.8:
            ms = 0.8
        iters = int(probe * TARGET_MS / ms)
        iters = max(2048, min(400_000, iters))
        self.log(f"калибровка ядра: {iters} итераций / запуск ({ms:.1f} мс проба)")
        return iters

    def _alloc_vram(self, cuda: Cuda, want_gb: float) -> tuple[list[int], int]:
        self._sizes = {}
        total = cuda.total_mem
        want = int(max(0.25, want_gb) * GB)
        cap = max(CHUNK, total - HEADROOM)
        want = min(want, cap)
        ptrs: list[int] = []
        left = want
        while left > 0:
            size = min(CHUNK, left)
            if size < 16 * 1024:
                break
            try:
                p = cuda.alloc(size)
            except CudaError:
                size = size // 2
                if size < 16 * 1024:
                    break
                continue
            try:
                cuda.memset(p, 0x5A, size)
            except CudaError:
                pass
            ptrs.append(p)
            self._sizes[p] = size
            left -= size
        allocated = sum(self._sizes.values())
        nvec = max(1, allocated // 16)
        self.log(f"VRAM занято: {allocated / GB:.2f} ГБ в {len(ptrs)} блоках")
        self._set(allocated_bytes=allocated)
        return ptrs, nvec

    def _loop_parallel(
        self,
        cuda: Cuda,
        fns: dict,
        scratch: int,
        grid: int,
        block: int,
        alu_iters: int,
        ten_iters: int,
        vram_ptrs: list[int],
        nvec: int,
        cfg: BurnConfig,
        started: float,
        stages: list[tuple[str, str]],
        until: float | None = None,
    ) -> None:
        keys = {k for k, _ in stages}
        stream_alu = cuda.stream(high_priority=True) if "alu" in keys else 0
        stream_ten = cuda.stream(high_priority=True) if "tensor" in keys else 0
        stream_ten2 = cuda.stream(high_priority=True) if "tensor" in keys else 0
        stream_mem = cuda.stream() if "vram" in keys else 0
        out = c_uint64(scratch)
        pair = 0
        nptr = len(vram_ptrs)
        compute_on = "alu" in keys or "tensor" in keys
        mem_grid = max(1, grid // 4) if compute_on else grid
        held: list = []

        tensor_cycle = []
        if "tensor" in keys:
            mode = cfg.tensor_mode
            if mode in ("all", "f16"):
                tensor_cycle.append(("f16", fns["f16"], FLOPS_F16_WARP))
            if mode in ("all", "tf32") and "tf32" in fns:
                tensor_cycle.append(("tf32", fns["tf32"], FLOPS_TF32_WARP))
            if mode in ("all", "fp8") and "fp8" in fns:
                tensor_cycle.append(("fp8", fns["fp8"], FLOPS_FP8_WARP))
            if not tensor_cycle:
                tensor_cycle.append(("f16", fns["f16"], FLOPS_F16_WARP))

        t_idx = 0
        tick = 0
        last = time.perf_counter()
        alu_flops = 0.0
        ten_flops = 0.0
        bytes_moved = 0.0
        mem_launched = False

        try:
            while not self._stop.is_set():
                if self._expired(cfg, started) or self._overheat(cfg):
                    break
                if until is not None and time.perf_counter() >= until:
                    break

                if not keys.intersection({"alu", "tensor", "vram"}):
                    time.sleep(0.2)
                    self._set(elapsed=time.perf_counter() - started)
                    continue

                waves = 8 if compute_on else 3
                warps = grid * block / 32.0
                held.clear()
                alu_flops = 0.0
                ten_flops = 0.0
                for _wave in range(waves):
                    if "alu" in keys:
                        it = c_uint32(alu_iters)
                        held.extend(cuda.launch(fns["alu"], (grid, 1, 1), (block, 1, 1), [out, it], stream_alu))
                        alu_flops += grid * block * alu_iters * ALU_FLOPS_PER_ITER
                    if "tensor" in keys:
                        _n, fn, flop_w = tensor_cycle[t_idx % len(tensor_cycle)]
                        t_idx += 1
                        it = c_uint32(ten_iters)
                        st = stream_ten2 if (t_idx & 1) and stream_ten2 else stream_ten
                        held.extend(cuda.launch(fn, (grid, 1, 1), (block, 1, 1), [out, it], st))
                        ten_flops += warps * ten_iters * flop_w
                    if "vram" in keys and nptr and (_wave == 0 or not compute_on):
                        src = vram_ptrs[pair % nptr]
                        dst = vram_ptrs[(pair + 1) % nptr]
                        pair += 1
                        nvec_pair = min(self._sizes.get(src, 0), self._sizes.get(dst, 0)) // 16
                        if nvec_pair > 0:
                            rounds = max(2, int(4 + cfg.intensity / 20)) if compute_on else max(4, int(8 + cfg.intensity / 10))
                            held.extend(cuda.launch(
                                fns["copy"],
                                (mem_grid, 1, 1),
                                (block, 1, 1),
                                [c_uint64(src), c_uint64(dst), c_uint64(nvec_pair), c_uint32(rounds)],
                                stream_mem,
                            ))
                            bytes_moved += nvec_pair * 16 * 2 * rounds
                            walk_n = min(nvec_pair, 2**31 - 1)
                            walk_rounds = max(8, rounds * 2) if compute_on else max(24, rounds * 8)
                            held.extend(cuda.launch(
                                fns["walk"],
                                (max(1, mem_grid), 1, 1),
                                (block, 1, 1),
                                [c_uint64(src), c_uint32(int(walk_n)), c_uint32(walk_rounds)],
                                stream_mem,
                            ))
                            mem_launched = True

                if stream_alu:
                    cuda.sync_stream(stream_alu)
                if stream_ten:
                    cuda.sync_stream(stream_ten)
                if stream_ten2:
                    cuda.sync_stream(stream_ten2)
                tick += 1
                if stream_mem and mem_launched and (not compute_on or tick % 3 == 0):
                    cuda.sync_stream(stream_mem)
                    mem_launched = False

                now = time.perf_counter()
                dt = max(1e-4, now - last)
                last = now
                self._set(
                    alu_tflops=alu_flops / dt / 1e12,
                    tensor_tflops=ten_flops / dt / 1e12,
                    vram_gbs=bytes_moved / dt / 1e9,
                    elapsed=now - started,
                )
                alu_flops = ten_flops = bytes_moved = 0.0
                held.clear()
        finally:
            if stream_alu:
                cuda.destroy_stream(stream_alu)
            if stream_ten:
                cuda.destroy_stream(stream_ten)
            if stream_ten2:
                cuda.destroy_stream(stream_ten2)
            if stream_mem:
                cuda.destroy_stream(stream_mem)

    def _start_shaders(self, cfg: BurnConfig) -> None:
        if self._stop.is_set():
            return
        if self._gl_thread and self._gl_thread.is_alive():
            return
        from coreforge.gl_stress import ShaderStress

        self._shader = ShaderStress(
            width=cfg.shader_width,
            height=cfg.shader_height,
            intensity=cfg.intensity,
            stop_event=self._stop,
            on_fps=self._on_shader_fps,
            on_hud=self.hud_lines,
            on_scene=self._on_shader_scene,
            on_log=self.log,
            fsr_mode=cfg.fsr_mode,
            render_choice=cfg.render_choice,
            vsync=cfg.vsync,
            fps_cap=0.0 if cfg.vsync else (90.0 if (cfg.cuda_cores or cfg.tensor) else 0.0),
            on_native_fps=self._on_shader_native_fps,
            on_fsr_info=self._on_fsr_info,
        )
        if cfg.cuda_cores or cfg.tensor:
            if not cfg.vsync:
                self.log("шейдеры: потолок 90 FPS, чтобы CUDA и тензоры не голодали (или включите VSync)")
        if cfg.fsr_mode and cfg.fsr_mode != "Выкл":
            self.log(f"шейдеры: FSR 1 {cfg.fsr_mode}, рендер {cfg.render_choice}")
        self._gl_thread = threading.Thread(target=self._shader.run, daemon=True, name="coreforge-gl")
        self._gl_thread.start()

    def set_vsync(self, enabled: bool) -> None:
        sh = getattr(self, "_shader", None)
        if sh is not None:
            sh.vsync = bool(enabled)
            if enabled:
                sh.fps_cap = 0.0

    def _on_shader_fps(self, fps: float) -> None:
        self._shader_fps = fps

    def _on_shader_native_fps(self, fps: float) -> None:
        self._shader_native_fps = fps

    def _on_fsr_info(self, text: str) -> None:
        self._fsr_info = text

    def _on_shader_scene(self, name: str) -> None:
        self._shader_scene = name

    def hud_lines(self) -> list[str]:
        snap = self.snapshot()
        s = snap.sample
        used_gb = s.vram_used / 1024**3 if s.vram_used else 0.0
        total_gb = s.vram_total / 1024**3 if s.vram_total else 0.0
        vram_pct = (s.vram_used / s.vram_total * 100.0) if s.vram_total else 0.0
        volt = f"{s.voltage_v:.2f} В" if s.voltage_v else "—"
        limit = "бесконечно" if getattr(self, "_duration_sec", 0) <= 0 else f"лимит {self._duration_sec} с"
        return [
            f"GPU {s.clock_graphics} МГц    MEM {s.clock_mem} МГц    {volt}",
            f"GPU {s.temp}°C    Hotspot {s.temp_hotspot or '—'}°C    VRAM {s.temp_vram or '—'}°C",
            f"{s.power_w:.0f} Вт    GPU {s.gpu_util}%    память {s.mem_util}%",
            f"VRAM {used_gb:.2f} / {total_gb:.1f} ГБ    ({vram_pct:.0f}%)",
            (
                f"Сцена: {self._shader_scene}    натив {self._shader_native_fps:.0f} / FSR {snap.shader_fps:.0f} FPS    "
                f"RT {snap.rt_mrays:.1f} Mrays/с"
                if self._fsr_info
                else f"Сцена: {self._shader_scene}    FPS {snap.shader_fps:.0f}    RT {snap.rt_mrays:.1f} Mrays/с"
            ),
            f"Нагрузка: {self._blocks_text}" + (f"    {self._fsr_info}" if self._fsr_info else ""),
            f"Режим: {snap.stage} · {snap.elapsed:.0f} с · {limit}",
        ]

    def _stop_shaders(self) -> None:
        shader = getattr(self, "_shader", None)
        if shader:
            shader.request_close()
        if self._gl_thread and self._gl_thread.is_alive():
            self._gl_thread.join(timeout=3.0)
        self._gl_thread = None
        self._shader = None
        self._shader_fps = 0.0
        self._shader_native_fps = 0.0
        self._fsr_info = ""

    def _start_rt(self, cfg: BurnConfig) -> None:
        if self._stop.is_set():
            return
        if not self.caps.has_vulkan:
            self.log("RT: нужен Vulkan Runtime")
            return
        if self._rt_thread and self._rt_thread.is_alive():
            return
        from coreforge.rt_stress import RayStress

        scale = 1.0 if (cfg.cuda_cores or cfg.tensor) else (1.25 if cfg.intensity < 70 else 1.5)
        self._rt = RayStress(
            width=max(1280, int(cfg.shader_width * scale)),
            height=max(720, int(cfg.shader_height * scale)),
            intensity=cfg.intensity,
            stop_event=self._stop,
            on_mrays=self._on_rt_mrays,
            on_log=self.log,
        )
        self._rt_thread = threading.Thread(target=self._rt.run, daemon=True, name="coreforge-rt")
        self._rt_thread.start()

    def _on_rt_mrays(self, mrays: float) -> None:
        self._rt_mrays = mrays

    def _stop_rt(self) -> None:
        rt = getattr(self, "_rt", None)
        if rt:
            rt.request_close()
        if self._rt_thread and self._rt_thread.is_alive():
            self._rt_thread.join(timeout=4.0)
        self._rt_thread = None
        self._rt = None
        self._rt_mrays = 0.0

    def _start_vk_compute(self, cfg: BurnConfig) -> None:
        if self._stop.is_set():
            return
        if self._vk_thread and self._vk_thread.is_alive():
            return
        from coreforge.vk_compute import VkComputeStress

        def on_alu(v: float) -> None:
            with self._lock:
                if cfg.cuda_cores:
                    self.stats.alu_tflops = v

        def on_mem(v: float) -> None:
            with self._lock:
                if cfg.vram or cfg.copy_engine:
                    self.stats.vram_gbs = v

        self._vk = VkComputeStress(
            stop_event=self._stop,
            intensity=cfg.intensity,
            vram_gb=cfg.vram_gb if cfg.vram else 0.35,
            do_alu=cfg.cuda_cores,
            do_vram=cfg.vram,
            do_copy=cfg.copy_engine or cfg.vram,
            on_log=self.log,
            on_tflops=on_alu,
            on_gbs=on_mem,
        )
        self._vk_thread = threading.Thread(target=self._vk.run, daemon=True, name="coreforge-vk")
        self._vk_thread.start()

    def _stop_vk_compute(self) -> None:
        vk = getattr(self, "_vk", None)
        if vk:
            vk.request_close()
        if self._vk_thread and self._vk_thread.is_alive():
            self._vk_thread.join(timeout=4.0)
        self._vk_thread = None
        self._vk = None

    def _start_copy(self, cfg: BurnConfig) -> None:
        if self._stop.is_set():
            return
        if not self.caps.has_cuda:
            return
        if self._copy_thread and self._copy_thread.is_alive():
            return
        from coreforge.copy_stress import CopyStress

        reserved = sum(self._sizes.values()) if cfg.vram else 0
        self._copy = CopyStress(
            stop_event=self._stop,
            intensity=cfg.intensity,
            on_log=self.log,
            reserved_bytes=reserved,
        )
        self._copy_thread = threading.Thread(target=self._copy.run, daemon=True, name="coreforge-copy")
        self._copy_thread.start()

    def _stop_copy(self) -> None:
        copy = getattr(self, "_copy", None)
        if copy:
            copy.request_close()
        if self._copy_thread and self._copy_thread.is_alive():
            self._copy_thread.join(timeout=4.0)
        self._copy_thread = None
        self._copy = None

    def _start_media(self, cfg: BurnConfig) -> None:
        if self._stop.is_set():
            return
        if not self.caps.is_nvidia or not (self.caps.has_nvenc or self.caps.has_nvdec):
            self.log("NVENC/NVDEC: нет на этой видеокарте (только NVIDIA)")
            return
        proc = getattr(self, "_media_proc", None)
        if proc is not None and proc.poll() is None:
            return
        import subprocess
        import sys
        from pathlib import Path

        exe = sys.executable
        if getattr(sys, "frozen", False):
            args = [exe, "--media-worker", f"--int={cfg.intensity}"]
        else:
            args = [exe, "-m", "coreforge.media_worker", f"--int={cfg.intensity}"]
        if cfg.nvenc:
            args.append("--enc")
        if cfg.nvdec:
            args.append("--dec")
        flags = 0
        if sys.platform == "win32":
            flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        log_path = Path(__file__).resolve().parent.parent / "_nvenc.log"
        logf = open(log_path, "w", encoding="utf-8", errors="replace")
        self._media_log = logf
        self._media_proc = subprocess.Popen(
            args,
            cwd=str(Path(__file__).resolve().parent.parent),
            stdin=subprocess.DEVNULL,
            stdout=logf,
            stderr=subprocess.STDOUT,
            creationflags=flags,
        )
        self.log("NVENC/NVDEC: отдельный процесс (Copy/Encode/Decode в диспетчере)")
        time.sleep(0.6)
        if self._media_proc.poll() is not None:
            try:
                logf.flush()
                text = log_path.read_text(encoding="utf-8", errors="replace").strip()
            except Exception:
                text = ""
            self.log(f"NVENC/NVDEC: процесс сразу вышел. {text or 'см. _nvenc.log'}")
        self._media_thread = None

    def _stop_media(self) -> None:
        proc = getattr(self, "_media_proc", None)
        if proc is not None:
            pid = getattr(proc, "pid", None)
            try:
                if pid and sys.platform == "win32":
                    import subprocess
                    subprocess.run(
                        ["taskkill", "/F", "/T", "/PID", str(pid)],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                    )
                else:
                    proc.terminate()
                    proc.wait(timeout=3.0)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        self._media_proc = None
        self._media_thread = None
        logf = getattr(self, "_media_log", None)
        if logf:
            try:
                logf.close()
            except Exception:
                pass
        self._media_log = None
