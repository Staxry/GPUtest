import sys

from coreforge.consoleutil import hide_console


def main() -> int:
    if "--media-worker" in sys.argv:
        from coreforge.media_worker import main as media_main

        media_main()
        return 0
    hide_console()
    from coreforge.assets import apply_process_identity
    from coreforge.engine import BurnEngine

    apply_process_identity()
    from coreforge.ui import App

    engine = BurnEngine()
    try:
        info = engine.probe()
    except Exception as exc:
        info = {"name": f"GPU: {exc}", "vram_total": 8 * 1024**3, "sm": 0, "cc": "?", "driver": "?", "vendor": "unknown"}
    app = App(engine, info)
    try:
        app.mainloop()
    finally:
        engine.shutdown("выход")
        engine.nvml.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
