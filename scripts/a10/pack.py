"""Pack the code and the small inputs needed on the cluster into artifacts/v33_upload.tar with a SHA256 manifest.

    python scripts/a10/pack.py
The model is not included; it is streamed into the job's node-local $TMPDIR by push_model.py.
"""

from __future__ import annotations

import hashlib
import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INPUTS = ["artifacts/chunk-depth-v31/items_formal_train.pt", "artifacts/chunk-depth-v31/items_pilot_cal.pt",
          "artifacts/chunk-depth-v31/items_formal_cal.pt", "artifacts/chunk-depth-v31/items_confirm.pt",
          "artifacts/chunk-depth-v31/items_pilot_train.pt", "artifacts/chunk-depth-v31/pilot_SCHED-A_s17.pt",
          "artifacts/chunk-depth-v32/head.pt", "artifacts/chunk-depth-v32/R.pt",
          "artifacts/gsm8k/train.parquet", "artifacts/gsm8k/test.parquet", "results/chunk-depth-v31/kv.json",
          "docs/EXPERIMENT_PLAN.v3.3.zh-CN.md"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def files() -> list[Path]:
    out = [p for p in (ROOT / "experiments").rglob("*.py") if "__pycache__" not in p.parts]
    out += sorted((ROOT / "docs").glob("EXPERIMENT_CONFIG*.json"))
    out += [p for p in (ROOT / "scripts" / "a10").iterdir() if p.is_file() and p.name != "pack.py"]
    out += [ROOT / p for p in INPUTS]
    return sorted(set(out))


def main() -> int:
    fs = files()
    for p in fs:
        if p.suffix in (".py", ".sbatch", ".txt", ".json", ".md") and b"\r\n" in p.read_bytes() and p.suffix == ".sbatch":
            raise SystemExit(f"CRLF line endings in {p}")
    manifest = {str(p.relative_to(ROOT)).replace("\\", "/"): {"bytes": p.stat().st_size, "sha256": sha256(p)} for p in fs}
    (ROOT / "artifacts" / "v33_upload_manifest.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    tar_path = ROOT / "artifacts" / "v33_upload.tar"
    with tarfile.open(tar_path, "w") as tar:
        for p in fs:
            tar.add(p, arcname=str(p.relative_to(ROOT)).replace("\\", "/"))
        tar.add(ROOT / "artifacts" / "v33_upload_manifest.json", arcname="upload_manifest.json")
    print(len(fs), "files", round(tar_path.stat().st_size / 2**20, 1), "MiB ->", tar_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
