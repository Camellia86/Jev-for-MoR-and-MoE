"""Run repository tests and write a compact, machine-readable validation receipt."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import platform
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).with_name("validation_receipt.json"))
    args = parser.parse_args()
    stream = io.StringIO()
    suite = unittest.defaultTestLoader.discover(
        start_dir=str(ROOT / "tests"), top_level_dir=str(ROOT)
    )
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    receipt = {
        "schema": "jev-routing-validation-receipt-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": ["python", "-m", "unittest", "discover", "-v"],
        "python": sys.version,
        "platform": platform.platform(),
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "skipped": len(result.skipped),
        "successful": result.wasSuccessful(),
        "notes": [
            "Optional torch tests are reported as skipped when torch is unavailable.",
            "No Canopy-Jev model was loaded; actual 27B inference requires its documented GPU environment.",
        ],
        "unittest_output": stream.getvalue(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: receipt[key] for key in ("tests_run", "failures", "errors", "skipped", "successful")},
                     ensure_ascii=False))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
