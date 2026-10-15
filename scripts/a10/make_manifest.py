"""Build scripts/a10/model_manifest.json from the local Huginn-0125 copy and cross-check it with the download record.

    python scripts/a10/make_manifest.py [--local-model DIR]
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

FILES = ["config.json", "generation_config.json", "model.safetensors.index.json",
         "model-00001-of-00004.safetensors", "model-00002-of-00004.safetensors",
         "model-00003-of-00004.safetensors", "model-00004-of-00004.safetensors",
         "raven_config_minimal.py", "raven_modeling_minimal.py", "special_tokens_map.json",
         "tokenizer.json", "tokenizer_config.json"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-model", default=r"C:\Users\z5885\git\loopjev\checkpoints\huginn-0125")
    args = parser.parse_args()
    root = Path(args.local_model)
    record = json.loads((root / "download-manifest.json").read_text(encoding="utf-8"))
    files, cross = {}, {}
    for name in FILES:
        p = root / name
        files[name] = {"bytes": p.stat().st_size, "sha256": sha256(p)}
        rec = record["files"].get(name)
        cross[name] = None if rec is None else (rec.get("sha256") == files[name]["sha256"])
        print(name, files[name]["bytes"], files[name]["sha256"][:16], "download record match:", cross[name], flush=True)
    if any(v is False for v in cross.values()):
        raise SystemExit("local files differ from the download record")
    out = {"repo": record["repo"], "revision": record["revision"], "files": files,
           "download_record_match": cross, "total_bytes": sum(f["bytes"] for f in files.values())}
    (Path(__file__).resolve().parent / "model_manifest.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print("total GiB", out["total_bytes"] / 2**30)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
