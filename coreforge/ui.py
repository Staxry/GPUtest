"""Тёмный интерфейс CoreForge."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from collections import deque

from coreforge.engine import BurnConfig, BurnEngine


BG = "#0d1016"
PANEL = "#161b24"
PANEL2 = "#1d2430"
LINE = "#2a3344"
TEXT = "#e8edf5"
MUTED = "#8b93a7"
ORANGE = "#ff6a3d"
BLUE = "#4aa3ff"
PURPLE = "#c084fc"
GREEN = "#34d399"
RED = "#fb7185"
YELLOW = "#fbbf24"
CYAN = "#22d3ee"


class Spark:
    def __init__(self, canvas: tk.Canvas, color: str, ymax: float) -> None:
        self.cv = canvas
        self.color = color
        self.ymax = ymax
        self.data: deque[float] = deque(maxlen=120)

    def add(self, value: float) -> None:
        self.data.append(value)
        self.draw()

    def draw(self) -> None:
        cv = self.cv
        w = int(cv.winfo_width() or 240)
        h = int(cv.winfo_height() or 48)
        if w < 8 or h < 8 or len(self.data) < 2:
            return
        mx = max(self.ymax, max(self.data) * 1.05, 1.0)
        n = len(self.data)
        pts = []
        for i, v in enumerate(self.data):
            x = 2 + (w - 4) * i / (n - 1)
            y = h - 3 - (h - 6) * min(v / mx, 1.0)
            pts.extend((x, y))
        last = self.data[-1]
        label = f"{last:.0f}"
        if not cv.find_withtag("spark"):
            cv.create_line(*pts, fill=self.color, width=2, tags="spark")
            cv.create_text(w - 6, 8, text=label, fill=self.color, anchor="e", font=("Segoe UI", 8), tags="sparkv")
            return
        cv.coords("spark", *pts)
        cv.coords("sparkv", w - 6, 8)
        cv.itemconfigure("sparkv", text=label)


class App(tk.Tk):
    def __init__(self, engine: BurnEngine, info: dict) -> None:
        super().__init__()
        self.engine = engine
        self.info = info
        self.title("CoreForge — стресс-тест GPU")
        self.configure(bg=BG)
        self._apply_icon()
        self._style()
        self._build()
        self._fit_screen()
        engine.set_logger(self._log)
        self._tick()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _apply_icon(self) -> None:
        from coreforge.assets import icon_ico, icon_png

        ico = icon_ico()
        if ico:
            try:
                self.iconbitmap(default=str(ico))
            except tk.TclError:
                try:
                    self.iconbitmap(str(ico))
                except tk.TclError:
                    pass
        png = icon_png()
        if png:
            try:
                self._icon_img = tk.PhotoImage(file=str(png))
                self.iconphoto(True, self._icon_img)
            except tk.TclError:
                pass

    def _fit_screen(self) -> None:
        self.update_idletasks()
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        w = min(1360, max(1100, sw - 24))
        h = min(900, max(720, sh - 60))
        self.geometry(f"{w}x{h}+0+0")
        self.minsize(980, 640)
        try:
            self.state("zoomed")
        except tk.TclError:
            pass

    def _style(self) -> None:
        st = ttk.Style(self)
        st.theme_use("clam")
        st.configure(".", background=BG, foreground=TEXT, fieldbackground=PANEL2, bordercolor=LINE)
        st.configure("TFrame", background=BG)
        st.configure("TLabel", background=BG, foreground=TEXT, font=("Segoe UI", 10))
        st.configure("Head.TLabel", background=BG, foreground=MUTED, font=("Segoe UI", 9))
        st.configure("Title.TLabel", background=BG, foreground=TEXT, font=("Segoe UI Semibold", 16))
        st.configure("TButton", background=PANEL2, foreground=TEXT, font=("Segoe UI Semibold", 10), padding=6)
        st.map("TButton", background=[("active", LINE)])
        st.configure("Accent.TButton", background=ORANGE, foreground="#1a0d08")
        st.map("Accent.TButton", background=[("active", "#ff825c")])
        st.configure("Stop.TButton", background="#3b1d24", foreground=RED)
        st.configure("Horizontal.TScale", background=PANEL, troughcolor=PANEL2)
        st.configure(
            "TCombobox",
            fieldbackground=PANEL2,
            background=PANEL2,
            foreground=TEXT,
            arrowcolor=TEXT,
            insertcolor=TEXT,
            selectbackground=LINE,
            selectforeground=TEXT,
            bordercolor=LINE,
            lightcolor=LINE,
            darkcolor=LINE,
        )
        st.map(
            "TCombobox",
            fieldbackground=[("readonly", PANEL2), ("disabled", PANEL)],
            foreground=[("readonly", TEXT), ("disabled", MUTED)],
            background=[("readonly", PANEL2)],
            arrowcolor=[("readonly", TEXT)],
        )
        self.option_add("*TCombobox*Listbox.background", PANEL2)
        self.option_add("*TCombobox*Listbox.foreground", TEXT)
        self.option_add("*TCombobox*Listbox.selectBackground", LINE)
        self.option_add("*TCombobox*Listbox.selectForeground", TEXT)
        self.option_add("*TCombobox*Listbox.font", "Segoe UI 10")

    def _card(self, parent, padx=0, pady=0) -> tk.Frame:
        f = tk.Frame(parent, bg=PANEL, highlightbackground=LINE, highlightthickness=1)
        f.pack(fill="x", padx=padx, pady=pady)
        return f

    def _build(self) -> None:
        vram_gb = self.info.get("vram_total", 12 * 1024**3) / 1024**3
        head = tk.Frame(self, bg=BG)
        head.pack(fill="x", padx=14, pady=(10, 4))
        ttk.Label(head, text="COREFORGE", style="Title.TLabel").pack(side="left")
        ttk.Label(head, text="  compute · тензоры · шейдеры · RT · память · Copy · видео", style="Head.TLabel").pack(side="left")
        self.hdr_stats = ttk.Label(head, text="", style="Head.TLabel")
        self.hdr_stats.pack(side="right")

        vendor = str(self.info.get("vendor", "")).upper() or "GPU"
        extra = []
        if self.info.get("sm"):
            extra.append(f"{self.info.get('sm')} SM")
        extra.append(str(self.info.get("cc", vendor)))
        extra.append(str(self.info.get("driver", "")))
        ttk.Label(
            self,
            text=f"{self.info.get('name', 'GPU')}   ·   {vram_gb:.1f} ГБ   ·   {vendor}   ·   " + "   ·   ".join(x for x in extra if x),
            style="Head.TLabel",
        ).pack(anchor="w", padx=16)

        body = tk.Frame(self, bg=BG)
        body.pack(fill="both", expand=True, padx=12, pady=8)
        left = tk.Frame(body, bg=BG, width=400)
        left.pack(side="left", fill="y")
        left.pack_propagate(False)
        right = tk.Frame(body, bg=BG)
        right.pack(side="left", fill="both", expand=True, padx=(10, 0))

        mode = self._card(left, pady=(0, 6))
        pad = tk.Frame(mode, bg=PANEL)
        pad.pack(fill="x", padx=10, pady=8)
        tk.Label(pad, text="РЕЖИМ", bg=PANEL, fg=MUTED, font=("Segoe UI", 8)).pack(anchor="w")
        self.together = tk.BooleanVar(value=True)
        row = tk.Frame(pad, bg=PANEL)
        row.pack(anchor="w", pady=(4, 0))
        tk.Radiobutton(row, text="Вместе", variable=self.together, value=True, bg=PANEL, fg=TEXT, selectcolor=PANEL2, activebackground=PANEL, activeforeground=TEXT, font=("Segoe UI", 10)).pack(side="left")
        tk.Radiobutton(row, text="По очереди", variable=self.together, value=False, bg=PANEL, fg=TEXT, selectcolor=PANEL2, activebackground=PANEL, activeforeground=TEXT, font=("Segoe UI", 10)).pack(side="left", padx=(14, 0))

        units = self._card(left, pady=(0, 6))
        pad = tk.Frame(units, bg=PANEL)
        pad.pack(fill="x", padx=10, pady=8)
        tk.Label(pad, text="ЧТО НАГРУЖАТЬ", bg=PANEL, fg=MUTED, font=("Segoe UI", 8)).pack(anchor="w")
        nvidia = bool(getattr(self.engine.caps, "is_nvidia", False) or self.info.get("vendor") == "nvidia")
        amd = bool(getattr(self.engine.caps, "is_amd", False) or self.info.get("vendor") == "amd")
        has_rt = bool(nvidia or self.info.get("has_ray_query"))
        self.var_cuda = tk.BooleanVar(value=bool(nvidia))
        self.var_tensor = tk.BooleanVar(value=bool(nvidia and self.info.get("has_tensor", True)))
        self.var_shader = tk.BooleanVar(value=True)
        self.var_rt = tk.BooleanVar(value=bool(has_rt))
        self.var_vram = tk.BooleanVar(value=True)
        self.var_copy = tk.BooleanVar(value=True)
        self.var_nvenc = tk.BooleanVar(value=bool(nvidia and self.info.get("has_nvenc")))
        self.var_nvdec = tk.BooleanVar(value=bool(nvidia and self.info.get("has_nvdec")))
        cuda_hint = "FP32 FMA, SFU, INT" if nvidia else "только NVIDIA"
        rt_hint = "Vulkan ray query" if has_rt else "нет RT на этой карте"
        if amd and not has_rt:
            rt_hint = "нужен RDNA 2+ и Vulkan"
        for var, title, hint, enable in (
            (self.var_cuda, "CUDA-ядра", cuda_hint, nvidia),
            (self.var_tensor, "Тензорные ядра", "только NVIDIA (MMA FP16/TF32/FP8)", nvidia),
            (self.var_shader, "Шейдеры", "OpenGL, NVIDIA и AMD", True),
            (self.var_rt, "RT-ядра", rt_hint, has_rt),
            (self.var_vram, "Видеопамять", "занятие + копирование", True),
            (self.var_copy, "Copy-движок", "DMA / Vulkan copy", True),
            (self.var_nvenc, "NVENC", "только NVIDIA", bool(nvidia and self.info.get("has_nvenc"))),
            (self.var_nvdec, "NVDEC", "только NVIDIA", bool(nvidia and self.info.get("has_nvdec"))),
        ):
            box = tk.Frame(pad, bg=PANEL)
            box.pack(fill="x", pady=1)
            fg = TEXT if enable else MUTED
            cb = tk.Checkbutton(box, text=title, variable=var, bg=PANEL, fg=fg, selectcolor=PANEL2, activebackground=PANEL, activeforeground=fg, font=("Segoe UI Semibold", 10))
            if not enable:
                var.set(False)
                cb.configure(state="disabled")
            cb.pack(side="left")
            tk.Label(box, text=hint, bg=PANEL, fg=MUTED, font=("Segoe UI", 8)).pack(side="right")

        mem = self._card(left, pady=(0, 6))
        pad = tk.Frame(mem, bg=PANEL)
        pad.pack(fill="x", padx=10, pady=8)
        top = tk.Frame(pad, bg=PANEL)
        top.pack(fill="x")
        tk.Label(top, text="ОБЪЁМ VRAM", bg=PANEL, fg=MUTED, font=("Segoe UI", 8)).pack(side="left")
        self.lbl_vram = tk.Label(top, text="6.0 ГБ", bg=PANEL, fg=GREEN, font=("Segoe UI Semibold", 10))
        self.lbl_vram.pack(side="right")
        max_gb = max(1.0, vram_gb - 0.9)
        default = min(8.0, max(1.0, max_gb * 0.65))
        self.var_vram_gb = tk.DoubleVar(value=round(default, 1))
        ttk.Scale(pad, from_=0.5, to=round(max_gb, 1), variable=self.var_vram_gb, command=lambda _v: self.lbl_vram.config(text=f"{self.var_vram_gb.get():.1f} ГБ")).pack(fill="x")
        tk.Label(pad, text=f"до {max_gb:.1f} ГБ, запас ~0.9 ГБ системе", bg=PANEL, fg=MUTED, font=("Segoe UI", 8)).pack(anchor="w")

        opts = self._card(left, pady=(0, 6))
        pad = tk.Frame(opts, bg=PANEL)
        pad.pack(fill="x", padx=10, pady=8)
        self.var_intensity = tk.IntVar(value=90)
        self.var_temp = tk.IntVar(value=83)
        self.var_infinite = tk.BooleanVar(value=True)
        self.var_duration = tk.IntVar(value=10)
        self.var_slice = tk.IntVar(value=30)
        self.var_tensor_mode = tk.StringVar(value="all")
        self.var_res = tk.StringVar(value="1920x1080")
        self.var_fsr = tk.StringVar(value="Выкл")
        self.var_render = tk.StringVar(value="Авто")
        self.var_vsync = tk.BooleanVar(value=False)
        self._slider(pad, "Интенсивность", self.var_intensity, 20, 100, "%")
        self._slider(pad, "Стоп по GPU °C (не hotspot)", self.var_temp, 70, 90, "°C")
        inf = tk.Frame(pad, bg=PANEL)
        inf.pack(fill="x", pady=(6, 0))
        tk.Checkbutton(
            inf, text="Бесконечно (пока не СТОП / лимит °C)",
            variable=self.var_infinite, command=self._toggle_infinite,
            bg=PANEL, fg=TEXT, selectcolor=PANEL2, activebackground=PANEL, activeforeground=TEXT,
            font=("Segoe UI Semibold", 10),
        ).pack(anchor="w")
        self.dur_box = tk.Frame(pad, bg=PANEL)
        self.dur_box.pack(fill="x")
        self._slider(self.dur_box, "Длительность, если не бесконечно", self.var_duration, 1, 30, " мин")
        self._toggle_infinite()
        self._slider(pad, "Шаг в раздельном режиме", self.var_slice, 10, 120, " с")
        row = tk.Frame(pad, bg=PANEL)
        row.pack(fill="x", pady=(6, 0))
        tk.Label(row, text="Тензоры", bg=PANEL, fg=MUTED, font=("Segoe UI", 8)).pack(side="left")
        self.cmb_tensor = self._dark_combo(row, self.var_tensor_mode, ("all", "f16", "tf32", "fp8"), width=8)
        self.cmb_tensor.pack(side="left", padx=(6, 16))
        if not nvidia:
            self.cmb_tensor.configure(state="disabled", fg=MUTED)
        tk.Label(row, text="Шейдеры/RT", bg=PANEL, fg=MUTED, font=("Segoe UI", 8)).pack(side="left")
        self._dark_combo(row, self.var_res, ("1280x720", "1920x1080", "2560x1440", "3840x2160"), width=12).pack(side="left", padx=6)
        fsr_row = tk.Frame(pad, bg=PANEL)
        fsr_row.pack(fill="x", pady=(6, 0))
        tk.Label(fsr_row, text="FSR 1", bg=PANEL, fg=MUTED, font=("Segoe UI", 8)).pack(side="left")
        self._dark_combo(
            fsr_row, self.var_fsr,
            ("Выкл", "Ultra Quality", "Quality", "Balanced", "Performance"),
            width=14,
        ).pack(side="left", padx=(6, 16))
        tk.Label(fsr_row, text="Рендер", bg=PANEL, fg=MUTED, font=("Segoe UI", 8)).pack(side="left")
        self.cmb_render = self._dark_combo(
            fsr_row, self.var_render,
            ("Авто", "960x540", "1280x720", "1600x900", "1920x1080"),
            width=10,
        )
        self.cmb_render.pack(side="left", padx=6)
        self.var_fsr.trace_add("write", lambda *_a: self._sync_fsr_ui())
        self._sync_fsr_ui()
        vs = tk.Frame(pad, bg=PANEL)
        vs.pack(fill="x", pady=(6, 0))
        tk.Checkbutton(
            vs, text="Вертикальная синхронизация (VSync)",
            variable=self.var_vsync, command=self._on_vsync,
            bg=PANEL, fg=TEXT, selectcolor=PANEL2, activebackground=PANEL, activeforeground=TEXT,
            font=("Segoe UI Semibold", 10),
        ).pack(side="left")
        tk.Label(vs, text="вкл — потолок герцовки, больше CUDA/тензор", bg=PANEL, fg=MUTED, font=("Segoe UI", 8)).pack(side="right")

        btns = tk.Frame(left, bg=BG)
        btns.pack(fill="x", pady=(4, 0))
        ttk.Button(btns, text="ЗАПУСК", style="Accent.TButton", command=self._start).pack(side="left", fill="x", expand=True, ipady=3)
        ttk.Button(btns, text="СТОП", style="Stop.TButton", command=lambda: self.engine.stop("остановлено")).pack(side="left", fill="x", expand=True, padx=(8, 0), ipady=3)

        metrics = tk.Frame(right, bg=BG)
        metrics.pack(fill="x")
        self.metric_labels = {}
        for key, title, color in (
            ("alu", "CUDA TFLOPS", ORANGE),
            ("tensor", "Tensor TFLOPS", PURPLE),
            ("vram", "VRAM ГБ/с", GREEN),
            ("fps", "Шейдеры FPS", BLUE),
            ("rt", "RT Mrays/с", CYAN),
        ):
            card = tk.Frame(metrics, bg=PANEL, highlightbackground=LINE, highlightthickness=1)
            card.pack(side="left", fill="both", expand=True, padx=(0, 6))
            tk.Label(card, text=title, bg=PANEL, fg=MUTED, font=("Segoe UI", 8)).pack(anchor="w", padx=8, pady=(6, 0))
            lbl = tk.Label(card, text="0.0", bg=PANEL, fg=color, font=("Segoe UI Semibold", 18))
            lbl.pack(anchor="w", padx=8, pady=(0, 6))
            self.metric_labels[key] = lbl

        charts = self._card(right, pady=(8, 0))
        charts.pack(fill="x")
        grid = tk.Frame(charts, bg=PANEL)
        grid.pack(fill="x", padx=8, pady=8)
        self.sparks = {}
        for i, (key, title, color, ymax) in enumerate((
            ("temp", "GPU °C", RED, 90),
            ("hot", "Hotspot °C", ORANGE, 105),
            ("vtemp", "VRAM °C", YELLOW, 110),
            ("power", "Мощность Вт", YELLOW, 300),
            ("util", "Загрузка GPU %", ORANGE, 100),
            ("mem", "VRAM занято ГБ", GREEN, vram_gb),
        )):
            cell = tk.Frame(grid, bg=PANEL)
            cell.grid(row=i // 3, column=i % 3, sticky="nsew", padx=5, pady=4)
            tk.Label(cell, text=title, bg=PANEL, fg=MUTED, font=("Segoe UI", 8)).pack(anchor="w")
            cv = tk.Canvas(cell, height=52, bg=PANEL2, highlightthickness=0)
            cv.pack(fill="x")
            self.sparks[key] = Spark(cv, color, ymax)
        for c in range(3):
            grid.columnconfigure(c, weight=1)

        status = self._card(right, pady=(8, 0))
        status.pack(fill="both", expand=True)
        pad = tk.Frame(status, bg=PANEL)
        pad.pack(fill="both", expand=True, padx=10, pady=8)
        self.lbl_stage = tk.Label(pad, text="ожидание", bg=PANEL, fg=TEXT, font=("Segoe UI Semibold", 11))
        self.lbl_stage.pack(anchor="w")
        self.lbl_detail = tk.Label(pad, text="выберите блоки и нажмите ЗАПУСК", bg=PANEL, fg=MUTED, font=("Segoe UI", 9))
        self.lbl_detail.pack(anchor="w", pady=(2, 6))
        self.log_box = tk.Text(pad, height=6, bg=PANEL2, fg=MUTED, insertbackground=TEXT, relief="flat", font=("Consolas", 9), wrap="word")
        self.log_box.pack(fill="both", expand=True)
        self.log_box.configure(state="disabled")

    def _on_vsync(self) -> None:
        self.engine.set_vsync(bool(self.var_vsync.get()))

    def _sync_fsr_ui(self) -> None:
        on = self.var_fsr.get() != "Выкл"
        if on:
            self.cmb_render.configure(state="normal", fg=TEXT)
        else:
            self.cmb_render.configure(state="disabled", fg=MUTED)

    def _toggle_infinite(self) -> None:
        state = "disabled" if self.var_infinite.get() else "normal"
        for child in self.dur_box.winfo_children():
            try:
                child.configure(state=state)
            except tk.TclError:
                for nested in child.winfo_children():
                    try:
                        nested.configure(state=state)
                    except tk.TclError:
                        pass

    def _dark_combo(self, parent, var: tk.StringVar, values: tuple[str, ...], width: int = 10) -> tk.Menubutton:
        mb = tk.Menubutton(
            parent,
            textvariable=var,
            bg=PANEL2,
            fg=TEXT,
            activebackground=LINE,
            activeforeground=TEXT,
            disabledforeground=MUTED,
            relief="flat",
            bd=0,
            highlightthickness=1,
            highlightbackground=LINE,
            highlightcolor=LINE,
            width=width,
            anchor="w",
            font=("Segoe UI", 9),
            indicatoron=True,
            direction="below",
        )
        menu = tk.Menu(mb, tearoff=0, bg=PANEL2, fg=TEXT, activebackground=LINE, activeforeground=TEXT, font=("Segoe UI", 9))
        for item in values:
            menu.add_command(label=item, command=lambda v=item: var.set(v))
        mb.configure(menu=menu)
        return mb

    def _slider(self, parent, title: str, var: tk.IntVar, a: int, b: int, suffix: str) -> None:
        row = tk.Frame(parent, bg=PANEL)
        row.pack(fill="x", pady=(4, 0))
        tk.Label(row, text=title, bg=PANEL, fg=MUTED, font=("Segoe UI", 8)).pack(side="left")
        val = tk.Label(row, text=f"{var.get()}{suffix}", bg=PANEL, fg=TEXT, font=("Segoe UI", 8))
        val.pack(side="right")
        ttk.Scale(
            parent, from_=a, to=b, variable=var,
            command=lambda v, lab=val, s=suffix, vr=var: (vr.set(int(float(v))), lab.config(text=f"{int(float(v))}{s}")),
        ).pack(fill="x")

    def _log(self, text: str) -> None:
        def append() -> None:
            self.log_box.configure(state="normal")
            self.log_box.insert("end", text + "\n")
            self.log_box.see("end")
            self.log_box.configure(state="disabled")

        self.after(0, append)

    def _config(self) -> BurnConfig:
        w, h = (int(x) for x in self.var_res.get().split("x"))
        return BurnConfig(
            together=bool(self.together.get()),
            cuda_cores=bool(self.var_cuda.get()),
            tensor=bool(self.var_tensor.get()),
            shaders=bool(self.var_shader.get()),
            rt_cores=bool(self.var_rt.get()),
            vram=bool(self.var_vram.get()),
            copy_engine=bool(self.var_copy.get()),
            nvenc=bool(self.var_nvenc.get()),
            nvdec=bool(self.var_nvdec.get()),
            vram_gb=float(self.var_vram_gb.get()),
            intensity=int(self.var_intensity.get()),
            temp_limit=int(self.var_temp.get()),
            duration_sec=0 if self.var_infinite.get() else int(self.var_duration.get()) * 60,
            sequential_sec=int(self.var_slice.get()),
            shader_width=w,
            shader_height=h,
            tensor_mode=self.var_tensor_mode.get(),
            fsr_mode=self.var_fsr.get(),
            render_choice=self.var_render.get(),
            vsync=bool(self.var_vsync.get()),
        )

    def _start(self) -> None:
        cfg = self._config()
        if not any((cfg.cuda_cores, cfg.tensor, cfg.shaders, cfg.rt_cores, cfg.vram, cfg.copy_engine, cfg.nvenc, cfg.nvdec)):
            self._log("включите хотя бы один блок нагрузки")
            return
        if cfg.duration_sec <= 0:
            self._log("старт: бесконечный прогон, остановите кнопкой СТОП")
        else:
            self._log(f"старт теста на {cfg.duration_sec // 60} мин…")
        self.engine.start(cfg)

    def _tick(self) -> None:
        snap = self.engine.snapshot()
        s = snap.sample
        self._stable_text(self.metric_labels["alu"], f"{snap.alu_tflops:.1f}")
        self._stable_text(self.metric_labels["tensor"], f"{snap.tensor_tflops:.1f}")
        self._stable_text(self.metric_labels["vram"], f"{snap.vram_gbs:.1f}")
        self._stable_text(self.metric_labels["fps"], f"{snap.shader_fps:.0f}")
        self._stable_text(self.metric_labels["rt"], f"{snap.rt_mrays:.1f}")
        if s.ok:
            self.sparks["temp"].add(s.temp)
            self.sparks["hot"].add(s.temp_hotspot)
            self.sparks["vtemp"].add(s.temp_vram)
            self.sparks["power"].add(s.power_w)
            self.sparks["util"].add(s.gpu_util)
            self.sparks["mem"].add(s.vram_used / 1024**3)
            parts = [f"GPU {s.temp}°C"]
            if s.temp_hotspot:
                parts.append(f"hot {s.temp_hotspot}°C")
            if s.temp_vram:
                parts.append(f"VRAM {s.temp_vram}°C")
            parts += [f"{s.power_w:.0f} Вт", f"GPU {s.gpu_util}%", f"{s.clock_graphics} МГц", f"вент {s.fan}%"]
            self._stable_text(self.hdr_stats, "   ".join(parts))
        extra = snap.message
        if snap.running:
            extra = f"{extra}   ·   {snap.elapsed:.0f} с"
            if snap.allocated_bytes:
                extra += f"   ·   занято {snap.allocated_bytes / 1024**3:.2f} ГБ"
        if snap.stopped_reason and not snap.running:
            extra = snap.stopped_reason
        self._stable_text(self.lbl_stage, snap.stage)
        self._stable_text(self.lbl_detail, extra)
        self.after(400, self._tick)

    def _stable_text(self, widget, text: str) -> None:
        if getattr(widget, "_cf_text", None) != text:
            widget._cf_text = text
            widget.config(text=text)

    def _on_close(self) -> None:
        try:
            self.engine.shutdown("закрытие")
        except Exception:
            pass
        self.destroy()
