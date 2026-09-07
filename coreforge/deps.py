"""Проверка и доустановка зависимостей CoreForge через командную строку."""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
import sys
from pathlib import Path


LINKS = {
    "python": "https://www.python.org/downloads/windows/",
    "driver": "https://www.nvidia.com/Download/index.aspx?lang=ru",
    "amd": "https://www.amd.com/en/support/download/drivers.html",
    "vulkan": "https://vulkan.lunarg.com/sdk/home#windows",
}

WINGET_PYTHON = "Python.Python.3.12"
WINGET_VULKAN = "KhronosGroup.VulkanRuntime"


def _ok(path: str) -> bool:
    if Path(path).is_file():
        return True
    try:
        ctypes.WinDLL(path)
        return True
    except OSError:
        return False


def _find_dll(*names: str) -> str:
    extra = [
        r"C:\Windows\System32",
        r"C:\Windows\SysWOW64",
        r"C:\Program Files\NVIDIA Corporation\NVSMI",
        os.environ.get("WINDIR", r"C:\Windows") + r"\System32",
    ]
    for name in names:
        if _ok(name):
            return name
        for folder in extra:
            cand = str(Path(folder) / name)
            if Path(cand).is_file():
                return cand
    return ""


def _run(cmd: list[str], timeout: int = 180) -> tuple[int, str]:
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding="utf-8",
            errors="replace",
        )
        out = (proc.stdout or "") + (proc.stderr or "")
        return proc.returncode, out.strip()
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 1, str(exc)


def _winget_install(package_id: str) -> bool:
    winget = shutil.which("winget")
    if not winget:
        print("  winget не найден — автоматическая установка недоступна.")
        return False
    print(f"  пробую winget install {package_id} …")
    code, out = _run(
        [
            winget,
            "install",
            "-e",
            "--id",
            package_id,
            "--accept-package-agreements",
            "--accept-source-agreements",
            "--disable-interactivity",
        ],
        timeout=300,
    )
    if out:
        for line in out.splitlines()[-8:]:
            print(f"    {line}")
    return code == 0


def check_python() -> bool:
    ver = sys.version_info
    print(f"Python {ver.major}.{ver.minor}.{ver.micro}  ({sys.executable})")
    if ver < (3, 10):
        print("  нужно Python 3.10 или новее.")
        if _winget_install(WINGET_PYTHON):
            print("  Python установлен через winget. Закройте окно и запустите setup.bat снова.")
            return False
        print(f"  скачайте вручную: {LINKS['python']}")
        return False
    return True


def check_dll(title: str, names: tuple[str, ...], link: str, critical: bool = True) -> bool:
    found = _find_dll(*names)
    if found:
        print(f"OK   {title}: {found}")
        return True
    print(f"НЕТ  {title}: {', '.join(names)}")
    print(f"     ссылка: {link}")
    return not critical


def required_ok() -> bool:
    if sys.version_info < (3, 10):
        return False
    nvidia = bool(_find_dll("nvcuda.dll"))
    amd = bool(_find_dll("atiadlxx.dll", "amdhip64.dll", "amdocl64.dll"))
    vulkan = bool(_find_dll("vulkan-1.dll"))
    return nvidia or amd or vulkan


def main() -> int:
    print("=== CoreForge: проверка зависимостей ===\n")
    bad = 0
    if not check_python():
        bad += 1

    print()
    nvidia = check_dll("NVIDIA CUDA (nvcuda.dll)", ("nvcuda.dll",), LINKS["driver"], critical=False)
    amd = check_dll("AMD ADL (atiadlxx.dll)", ("atiadlxx.dll",), LINKS["amd"], critical=False)
    if not nvidia and not amd:
        print("  не найден драйвер NVIDIA или AMD.")
        print(f"  NVIDIA: {LINKS['driver']}")
        print(f"  AMD:    {LINKS['amd']}")
        bad += 1
    check_dll("NVML (температуры NVIDIA)", ("nvml.dll", r"C:\Windows\System32\nvml.dll"), LINKS["driver"], critical=False)
    check_dll("NvAPI (hotspot / VRAM °C)", ("nvapi64.dll",), LINKS["driver"], critical=False)
    check_dll("NVENC (Video Encode, только NVIDIA)", ("nvEncodeAPI64.dll",), LINKS["driver"], critical=False)
    check_dll("NVDEC (Video Decode, только NVIDIA)", ("nvcuvid.dll",), LINKS["driver"], critical=False)
    if not check_dll("Vulkan (RT-ядра)", ("vulkan-1.dll",), LINKS["vulkan"], critical=False):
        if _winget_install(WINGET_VULKAN):
            print("  Vulkan Runtime установлен.")
        else:
            print(f"  runtime: {LINKS['vulkan']}")
    check_dll("OpenGL (окно шейдеров)", ("opengl32.dll",), LINKS["driver"], critical=False)

    print()
    if bad:
        print("Есть критические пропуски. Поставьте драйвер / Python по ссылкам выше и запустите setup.bat ещё раз.")
        return 1
    print("Всё необходимое на месте. Можно запускать run.bat")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
