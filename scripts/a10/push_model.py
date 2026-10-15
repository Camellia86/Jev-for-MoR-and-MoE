"""Stream the local FP32 Huginn files into the running job's node-local $TMPDIR (plan v3.3 section 7.2).

    python scripts/a10/push_model.py <jobid> [--local-model DIR]

One SSH connection carries everything: an uncompressed tar stream of the model files goes local -> ssh kudu-taught ->
`srun --jobid <job> --overlap` -> tar -x in the node's model directory; the same remote command then checks every file
against scripts/a10/model_manifest.json and only on success writes .upload_done, which releases the job.
Opening a new connection per file is avoided: the jump host drops bursts of new connections.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path

HOST = "kudu-taught"
HERE = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("job")
    parser.add_argument("--local-model", default=r"C:\Users\z5885\git\loopjev\checkpoints\huginn-0125")
    parser.add_argument("--model-dir", required=True, help="model_dir from ~/jev-v33/run/<job>.ready")
    args = parser.parse_args()
    files = sorted(json.loads((HERE / "model_manifest.json").read_text(encoding="utf-8"))["files"])
    d = shlex.quote(args.model_dir)
    inner = (f"set -e; tar -C {d} -xf -; python3 $HOME/jev-v33/code/scripts/a10/verify_model.py {d} "
             f"$HOME/jev-v33/code/scripts/a10/model_manifest.json && touch {d}/.upload_done && echo RELEASED")
    remote = (f"srun --jobid={args.job} --overlap --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=4G --gres=none "
              f"bash -c {shlex.quote(inner)}")
    t0 = time.perf_counter()
    tar = subprocess.Popen(["tar", "-C", args.local_model, "-cf", "-", *files], stdout=subprocess.PIPE)
    r = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ServerAliveInterval=30", HOST, remote], stdin=tar.stdout,
                       capture_output=True, text=True)
    tar.wait()
    print(r.stdout, r.stderr, f"seconds {time.perf_counter() - t0:.0f}", sep="\n", flush=True)
    return 0 if "RELEASED" in r.stdout else 1


if __name__ == "__main__":
    raise SystemExit(main())
