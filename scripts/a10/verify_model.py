"""Check every model file in the node-local directory against the manifest (SHA256 and size).

    python verify_model.py <model_dir> <model_manifest.json>
"""

import hashlib
import json
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    model, manifest = Path(sys.argv[1]), json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
    bad = []
    for name, entry in manifest["files"].items():
        p = model / name
        if not p.exists() or p.stat().st_size != entry["bytes"] or sha256(p) != entry["sha256"]:
            bad.append(name)
    if bad:
        print("model verification failed:", bad, flush=True)
        return 1
    print(f"model verified: {len(manifest['files'])} files", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
