"""Run every v3.3 stage in order on one node (plan section 7.2), resuming from stage markers.

    python -m experiments.chunk_depth_v33.driver --gpus 3 --nshards 3            (A10 job)
    JEV_SMOKE=1 python -m experiments.chunk_depth_v33.driver --gpus 1 --nshards 3 --sequential   (local smoke)

A stage whose marker exists is skipped. A scratch stage (tables, features) is also skipped when the stage that
needs it is already done; otherwise, if its files are missing on this node, it is recomputed.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

from experiments.shared_depth_v2.common import write_json
from experiments.chunk_depth_v33 import common as C

PY = sys.executable


def log(*args):
    print(time.strftime("%Y-%m-%d %H:%M:%S"), *args, flush=True)


def worker_cmd(stage, shard, nshards):
    return [PY, "-m", "experiments.chunk_depth_v33.worker", stage, "--shard", str(shard), "--nshards", str(nshards)]


def run_parallel(stage, cmds, gpus, sequential, threads):
    """cmds: list of (label, argv). Worker i runs on GPU i % gpus."""
    logs = C.RESULTS / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    procs = []
    for i, (label, argv) in enumerate(cmds):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(i % gpus), OMP_NUM_THREADS=str(threads),
                   MKL_NUM_THREADS=str(threads), PYTHONUNBUFFERED="1")
        fh = open(logs / f"{label}.log", "a", encoding="utf-8")
        fh.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(argv[2:])}\n")
        fh.flush()
        p = subprocess.Popen(argv, env=env, stdout=fh, stderr=subprocess.STDOUT, cwd=str(C.ROOT))
        if sequential:
            p.wait()
            fh.close()
            if p.returncode:
                raise RuntimeError(f"{stage}: {label} exited with {p.returncode}")
        else:
            procs.append((label, p, fh))
    failed = []
    for label, p, fh in procs:
        p.wait()
        fh.close()
        if p.returncode:
            failed.append((label, p.returncode))
    if failed:
        raise RuntimeError(f"{stage}: workers failed {failed}")


def stage_table(nshards):
    w = lambda stage, n: [(f"{stage}_{i}of{n}", worker_cmd(stage, i, n)) for i in range(n)]
    one = lambda stage: [(f"{stage}_0of1", worker_cmd(stage, 0, 1))]
    py = lambda label, *args: [(label, [PY, "-m", *args])]
    return [
        ("A_tables", w("A_tables", nshards), {"scratch_of": ("A_fit", [("tables_fit", nshards), ("tables_calibration", nshards)])}),
        ("A_fit", py("A_fit", "experiments.chunk_depth_v33.lock", "A_fit", "--nshards", str(nshards)), {}),
        ("A_calcheck", w("A_calcheck", nshards), {}),
        ("A_online", w("A_online", nshards), {}),
        ("A_analyze", py("A_analyze", "experiments.chunk_depth_v33.analyze", "A", "--nshards", str(nshards)), {}),
        ("B_features", one("B_features"), {"scratch_of": ("B_train", [("features_btrain", 1)])}),
        ("B_train", w("B_train", nshards), {}),
        ("B_eval", w("B_eval", nshards), {}),
        ("B_deploy_tables", w("B_deploy_tables", nshards),
         {"scratch_of": ("B_deploy_lock", [(f"deploy_tables_{a}", 1) for a in C.ARMS])}),
        ("B_deploy_lock", py("B_deploy_lock", "experiments.chunk_depth_v33.lock", "B_deploy_lock"), {}),
        ("B_deploy_online", w("B_deploy_online", nshards), {}),
        ("B_analyze", py("B_analyze", "experiments.chunk_depth_v33.analyze", "B", "--nshards", str(nshards)), {}),
    ]


def verify():
    if C.stage_done("verify"):
        return
    cfg = C.load_config()
    reused = C.verify_reused(cfg)
    counts = {role: len(C.load_role(role)) for role in ("fit", "calibration", "confirm_train", "confirm_test", "btrain")}
    write_json(C.RESULTS / "verify.json", C.v33_receipt("v33_verify", reused_sha256=reused, role_documents=counts,
                                                       model_dir=os.environ.get("JEV_HUGINN_DIR")))
    C.mark_stage("verify", role_documents=counts)
    log("verify", json.dumps(counts))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", type=int, default=3)
    parser.add_argument("--nshards", type=int, default=3)
    parser.add_argument("--sequential", action="store_true")
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--only", nargs="*", default=None, help="run only these stages (debugging)")
    args = parser.parse_args()
    t_all = time.perf_counter()
    log("driver start", "smoke" if C.SMOKE else "full", f"gpus={args.gpus}", f"nshards={args.nshards}",
        f"scratch={C.SCRATCH}", f"results={C.RESULTS}")
    C.SCRATCH.mkdir(parents=True, exist_ok=True)
    verify()
    timing = {}
    for name, cmds, opts in stage_table(args.nshards):
        if args.only and name not in args.only:
            continue
        if C.stage_done(name):
            dep = opts.get("scratch_of")
            if dep is None or C.stage_done(dep[0]) or all(C.shards_complete(k, n, scratch=True) for k, n in dep[1]):
                log("skip", name)
                continue
            log("recompute scratch stage", name, "(files missing on this node)")
        log("stage", name, "start")
        t0 = time.perf_counter()
        gpus_used = min(len(cmds), args.gpus)
        run_parallel(name, cmds, args.gpus, args.sequential, args.threads)
        dt = time.perf_counter() - t0
        timing[name] = {"seconds": dt, "gpu_hours": dt * (1 if args.sequential else gpus_used) / 3600}
        C.mark_stage(name, seconds=dt, workers=len(cmds))
        log("stage", name, "done", f"{dt / 60:.1f} min")
    write_json(C.RESULTS / "driver_run.json", C.v33_receipt("v33_driver_run", timing=timing,
                                                           seconds=time.perf_counter() - t_all))
    log("driver end", f"{(time.perf_counter() - t_all) / 3600:.2f} h")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
