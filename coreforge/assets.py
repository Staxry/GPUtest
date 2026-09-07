"""Пути к иконке: исходники и собранный exe."""

from __future__ import annotations

import sys
from pathlib import Path


def _root() -> Path:
    if getattr(sys, "frozen", False) and getattr(sys, "_MEIPASS", None):
        return Path(sys._MEIPASS)
    return Path(__file__).resolve().parent.parent


def icon_png() -> Path | None:
    p = _root() / "assets" / "coreforge.png"
    return p if p.is_file() else None


def icon_ico() -> Path | None:
    p = _root() / "assets" / "coreforge.ico"
    return p if p.is_file() else None


def apply_process_identity() -> None:
    if sys.platform != "win32":
        return
    try:
        import ctypes

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("CoreForge.GPUStress")
    except Exception:
        pass
