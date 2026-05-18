"""
Auto-regenerate reports/report.pdf whenever reports/report.md is saved.

Usage:
    python src/watch_report.py

Press Ctrl+C to stop.
"""

import subprocess
import sys
import time
from pathlib import Path

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

ROOT = Path(__file__).resolve().parents[1]
REPORT_MD = ROOT / "reports" / "report.md"
GENERATE_SCRIPT = ROOT / "src" / "generate_pdf.py"

_last_run: float = 0
_DEBOUNCE = 1.5  # seconds — ignore rapid duplicate save events


class ReportHandler(FileSystemEventHandler):
    def on_modified(self, event):
        global _last_run
        if Path(event.src_path).resolve() != REPORT_MD:
            return
        now = time.monotonic()
        if now - _last_run < _DEBOUNCE:
            return
        _last_run = now
        print(f"\n  report.md changed — regenerating PDF…")
        result = subprocess.run(
            [sys.executable, str(GENERATE_SCRIPT)],
            cwd=str(ROOT),
        )
        if result.returncode == 0:
            print("  Done. reports/report.pdf updated.\n")
        else:
            print("  PDF generation failed — check output above.\n")


def main() -> None:
    print("=" * 50)
    print("  Watching reports/report.md for changes")
    print("  Press Ctrl+C to stop")
    print("=" * 50)

    handler = ReportHandler()
    observer = Observer()
    observer.schedule(handler, path=str(REPORT_MD.parent), recursive=False)
    observer.start()

    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        observer.stop()
        print("\n  Watcher stopped.")
    observer.join()


if __name__ == "__main__":
    main()
