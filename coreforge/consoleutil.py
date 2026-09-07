"""Скрыть чёрную консоль, если приложение запущено через run.bat / ярлык."""

from __future__ import annotations

import ctypes
import sys


def hide_console() -> None:
    if sys.platform != "win32":
        return
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    kernel32.GetConsoleProcessList.argtypes = [ctypes.POINTER(ctypes.c_uint), ctypes.c_uint]
    kernel32.GetConsoleProcessList.restype = ctypes.c_uint
    kernel32.GetConsoleWindow.restype = ctypes.c_void_p
    user32.ShowWindow.argtypes = [ctypes.c_void_p, ctypes.c_int]
    buf = (ctypes.c_uint * 8)()
    attached = kernel32.GetConsoleProcessList(buf, 8)
    # 1 — только мы, 2 — cmd.exe + python из run.bat
    if attached == 0 or attached > 2:
        return
    hwnd = kernel32.GetConsoleWindow()
    if hwnd:
        user32.ShowWindow(hwnd, 0)
