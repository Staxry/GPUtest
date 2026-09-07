"""Отдельный процесс NVENC/NVDEC: падение API не роняет окно CoreForge."""

from __future__ import annotations

import sys
import threading


def main() -> None:
    encode = "--enc" in sys.argv
    decode = "--dec" in sys.argv
    intensity = 70
    for arg in sys.argv:
        if arg.startswith("--int="):
            intensity = int(arg.split("=", 1)[1])
    if not encode and not decode:
        encode = decode = True

    from coreforge.media_stress import MediaStress

    stop = threading.Event()
    media = MediaStress(stop, encode, decode, intensity, lambda m: print(m, flush=True))
    try:
        media.run()
    except KeyboardInterrupt:
        media.request_close()
        stop.set()


if __name__ == "__main__":
    main()
