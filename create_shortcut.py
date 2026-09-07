"""Ярлык CoreForge на рабочем столе и в папке проекта."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def _targets(root: Path) -> tuple[Path, Path]:
    ico = root / "assets" / "coreforge.ico"
    exe = root / "dist" / "CoreForge" / "CoreForge.exe"
    if exe.is_file():
        return exe, ico if ico.is_file() else exe
    bat = root / "run.bat"
    if bat.is_file():
        return bat, ico if ico.is_file() else bat
    return root / "main.py", ico


def make_shortcut(link: Path, target: Path, icon: Path, workdir: Path) -> None:
    def q(p: Path) -> str:
        return str(p).replace("'", "''")

    ps = (
        "$ws = New-Object -ComObject WScript.Shell; "
        f"$s = $ws.CreateShortcut('{q(link)}'); "
        f"$s.TargetPath = '{q(target)}'; "
        f"$s.WorkingDirectory = '{q(workdir)}'; "
        f"$s.IconLocation = '{q(icon)},0'; "
        "$s.Description = 'CoreForge - GPU stress'; "
        "$s.Save()"
    )
    subprocess.run(
        ["powershell", "-NoProfile", "-Command", ps],
        check=True,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )


def desktop_dir() -> Path:
    home = Path(os.environ.get("USERPROFILE", str(Path.home())))
    return home / "Desktop"


def main() -> int:
    root = Path(__file__).resolve().parent
    target, icon = _targets(root)
    if not target.is_file():
        print("нет цели для ярлыка (exe / run.bat)", file=sys.stderr)
        return 1
    workdir = target.parent if target.suffix.lower() == ".exe" else root
    icon_path = icon if icon.is_file() else target
    links = [root / "CoreForge.lnk", desktop_dir() / "CoreForge.lnk"]
    ok = 0
    for link in links:
        try:
            make_shortcut(link, target, icon_path, workdir)
            print(f"ярлык: {link}")
            ok += 1
        except Exception as exc:
            print(f"не создан {link}: {exc}", file=sys.stderr)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
