#!/usr/bin/env python3
"""
Multi-worker scaling benchmark for WAMOS pipeline.

Measures how throughput scales with number of parallel workers across
different backend configurations (NumPy-only, PyTorch GPU, etc.).

Each (config, n_workers) combination runs in a subprocess with the
appropriate WAMOS_NO_GPU / WAMOS_NO_NUMBA env vars so that module-level
detection picks up the correct settings.

Usage:
    python benchmarks/scaling_benchmark.py /path/to/POLAR
    python benchmarks/scaling_benchmark.py /path/to/POLAR -n 30 --workers 1,2,4,8,16
    python benchmarks/scaling_benchmark.py /path/to/POLAR --configs numpy pytorch both
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

# ── Configuration matrix ──

CONFIGS = {
    "numpy": {
        "name": "NumPy-only",
        "env": {"WAMOS_NO_GPU": "1", "WAMOS_NO_NUMBA": "1", "WAMOS_NO_CUPY": "1"},
    },
    "numba": {
        "name": "Numba",
        "env": {"WAMOS_NO_GPU": "1", "WAMOS_NO_NUMBA": "", "WAMOS_NO_CUPY": "1"},
    },
    "pytorch": {
        "name": "PyTorch GPU",
        "env": {"WAMOS_NO_GPU": "", "WAMOS_NO_NUMBA": "1", "WAMOS_NO_CUPY": "1"},
    },
    "both": {
        "name": "PyTorch + Numba",
        "env": {"WAMOS_NO_GPU": "", "WAMOS_NO_NUMBA": "", "WAMOS_NO_CUPY": "1"},
    },
    "cupy": {
        "name": "CuPy",
        "env": {"WAMOS_NO_GPU": "1", "WAMOS_NO_NUMBA": "1", "WAMOS_NO_CUPY": ""},
    },
    "cupy+numba": {
        "name": "CuPy + Numba",
        "env": {"WAMOS_NO_GPU": "1", "WAMOS_NO_NUMBA": "", "WAMOS_NO_CUPY": ""},
    },
}

# ── Worker script (run in subprocess) ──

SCALING_WORKER_SCRIPT = r'''
"""Worker subprocess for scaling benchmark."""
import gc
import json
import os
import resource
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, "{src_dir}")

# NOTE: Do NOT import wamos_tpw.backend here — it initializes CUDA, which
# must not happen before fork().  Workers import it via FramePipeline.


def process_one_file(filepath, config_path, frame_index):
    """Process a single file through FramePipeline. Returns timing + RSS."""
    import resource
    import time

    from wamos_tpw.config import Config
    from wamos_tpw.frame_pipeline import FramePipeline
    from wamos_tpw.polarfile import PolarFile

    config = Config(config_path) if config_path else Config()
    try:
        pf = PolarFile(filepath, config=config)
        frame = pf.frame()
        t0 = time.perf_counter()
        fp = FramePipeline(frame, config=config, qTiming=True)
        elapsed = time.perf_counter() - t0
        peak_rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return {{
            "filepath": filepath,
            "elapsed": elapsed,
            "timings": dict(fp.timings),
            "peak_rss_mb": peak_rss_kb / 1024,
            "success": True,
        }}
    except Exception as e:
        return {{
            "filepath": filepath,
            "elapsed": 0.0,
            "timings": {{}},
            "peak_rss_mb": 0.0,
            "success": False,
            "error": str(e),
        }}


def main():
    filepaths = {filepaths_repr}
    config_path = {config_repr}
    frame_index = {frame_index}
    n_workers = {n_workers}

    # Baseline RSS
    baseline_rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    results = []
    gc.collect()

    wall_start = time.perf_counter()

    if n_workers == 1:
        # Sequential — avoid pool overhead
        for fp in filepaths:
            results.append(process_one_file(fp, config_path, frame_index))
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {{
                pool.submit(process_one_file, fp, config_path, frame_index): fp
                for fp in filepaths
            }}
            for fut in as_completed(futures):
                results.append(fut.result())

    wall_elapsed = time.perf_counter() - wall_start

    # Aggregate
    n_success = sum(1 for r in results if r["success"])
    elapsed_list = [r["elapsed"] for r in results if r["success"]]
    rss_list = [r["peak_rss_mb"] for r in results if r["success"]]

    peak_rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    # Import backend AFTER workers finish (avoids CUDA init before fork)
    from wamos_tpw.backend import HAS_NUMBA, HAS_TORCH_GPU

    output = {{
        "HAS_TORCH_GPU": HAS_TORCH_GPU,
        "HAS_NUMBA": HAS_NUMBA,
        "n_workers": n_workers,
        "n_files": len(filepaths),
        "n_success": n_success,
        "wall_elapsed": wall_elapsed,
        "fps": n_success / wall_elapsed if wall_elapsed > 0 else 0,
        "per_frame_times": elapsed_list,
        "worker_rss_list": rss_list,
        "parent_baseline_rss_mb": baseline_rss_kb / 1024,
        "parent_peak_rss_mb": peak_rss_kb / 1024,
    }}

    # GPU memory stats (device-wide)
    if HAS_TORCH_GPU:
        import torch
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            output["gpu_free_mb"] = free / (1024 * 1024)
            output["gpu_total_mb"] = total / (1024 * 1024)

    print(json.dumps(output))


if __name__ == '__main__':
    main()
'''


def find_polar_files(polar_path: Path, n_files: int) -> list[str]:
    """Discover n_files evenly-spaced .pol files under polar_path."""
    all_files = sorted(polar_path.rglob("*.pol*"))
    if not all_files:
        return []

    if len(all_files) <= n_files:
        return [str(f) for f in all_files]

    # Evenly space files
    step = len(all_files) / n_files
    indices = [int(i * step) for i in range(n_files)]
    return [str(all_files[i]) for i in indices]


def run_scaling_config(
    cfg: dict,
    filepaths: list[str],
    config_path: str | None,
    n_workers: int,
    frame_index: int,
    src_dir: str,
) -> dict | None:
    """Launch one subprocess for a (config, n_workers) pair."""
    env = dict(os.environ)
    env.pop("WAMOS_NO_GPU", None)
    env.pop("WAMOS_NO_NUMBA", None)
    env.pop("WAMOS_NO_CUPY", None)
    for k, v in cfg["env"].items():
        if v:
            env[k] = v
        else:
            env.pop(k, None)

    script = SCALING_WORKER_SCRIPT.format(
        src_dir=src_dir,
        filepaths_repr=repr(filepaths),
        config_repr=repr(config_path) if config_path else "None",
        frame_index=frame_index,
        n_workers=n_workers,
    )

    # Generous timeout: up to 60s per file
    timeout = max(600, len(filepaths) * 60)

    # Write script to a temp file instead of using -c, so that
    # ProcessPoolExecutor (spawn) can pickle functions from __main__.
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tmp:
        tmp.write(script)
        tmp_path = tmp.name

    try:
        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            env=env,
            timeout=timeout,
        )
    finally:
        os.unlink(tmp_path)

    if result.returncode != 0:
        print(f"\n  ERROR running {cfg['name']} w={n_workers}:", file=sys.stderr)
        stderr = result.stderr
        print(stderr[-2000:] if len(stderr) > 2000 else stderr, file=sys.stderr)
        return None

    try:
        return json.loads(result.stdout.strip())
    except json.JSONDecodeError:
        print(f"\n  ERROR parsing output for {cfg['name']} w={n_workers}:", file=sys.stderr)
        print(f"  stdout: {result.stdout[:500]}", file=sys.stderr)
        return None


def print_scaling_table(config_results: dict[str, list[dict]]) -> None:
    """Print scaling results per config."""
    import statistics as st

    for cfg_name, results in config_results.items():
        if not results:
            continue

        print(f"\n{'=' * 100}")
        print(f"  SCALING: {cfg_name}")

        # Backend info from first result
        r0 = results[0]
        backends = []
        if r0.get("HAS_TORCH_GPU"):
            backends.append("PyTorch/CUDA")
        if r0.get("HAS_NUMBA"):
            backends.append("Numba")
        if not backends:
            backends.append("NumPy")
        print(f"  Backend: {' + '.join(backends)}")
        print(f"  Files: {r0['n_files']}")
        print(f"{'=' * 100}")

        header = (
            f"  {'Workers':>7}  {'Wall(s)':>8}  {'FPS':>8}  "
            f"{'Speedup':>8}  {'Effic%':>7}  "
            f"{'MeanRSS':>9}  {'MaxRSS':>9}"
        )
        has_gpu = any(r.get("gpu_total_mb") for r in results)
        if has_gpu:
            header += f"  {'GPU Used':>10}  {'GPU Total':>10}"
        print(header)
        print("  " + "-" * (len(header) - 2))

        # FPS at 1 worker for speedup calculation
        fps_1 = None
        for r in results:
            if r["n_workers"] == 1:
                fps_1 = r["fps"]
                break
        if fps_1 is None and results:
            fps_1 = results[0]["fps"]

        for r in results:
            n_w = r["n_workers"]
            wall = r["wall_elapsed"]
            fps = r["fps"]
            speedup = fps / fps_1 if fps_1 and fps_1 > 0 else 1.0
            efficiency = (speedup / n_w) * 100

            rss_list = r.get("worker_rss_list", [])
            mean_rss = st.mean(rss_list) if rss_list else 0
            max_rss = max(rss_list) if rss_list else 0

            row = (
                f"  {n_w:>7d}  {wall:>8.2f}  {fps:>8.2f}  "
                f"{speedup:>7.2f}x  {efficiency:>6.1f}%  "
                f"{mean_rss:>8.1f}M  {max_rss:>8.1f}M"
            )
            if has_gpu:
                gpu_total = r.get("gpu_total_mb", 0)
                gpu_free = r.get("gpu_free_mb", 0)
                gpu_used = gpu_total - gpu_free if gpu_total else 0
                row += f"  {gpu_used:>9.0f}M  {gpu_total:>9.0f}M"
            print(row)

        n_success = results[-1].get("n_success", 0)
        n_files = results[-1].get("n_files", 0)
        if n_success < n_files:
            print(f"\n  WARNING: {n_files - n_success} files failed processing")


def print_cross_config_summary(config_results: dict[str, list[dict]]) -> None:
    """Print cross-config comparison of optimal worker counts."""
    print(f"\n{'=' * 100}")
    print("  CROSS-CONFIG SUMMARY")
    print(f"{'=' * 100}")

    header = f"  {'Config':<20s}  {'Best Workers':>12}  {'Peak FPS':>9}  {'Peak Speedup':>12}  {'Wall(s) @best':>13}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for cfg_name, results in config_results.items():
        if not results:
            continue

        # Find best FPS
        best = max(results, key=lambda r: r["fps"])
        fps_1 = None
        for r in results:
            if r["n_workers"] == 1:
                fps_1 = r["fps"]
                break
        if fps_1 is None:
            fps_1 = results[0]["fps"]

        speedup = best["fps"] / fps_1 if fps_1 and fps_1 > 0 else 1.0

        print(
            f"  {cfg_name:<20s}  {best['n_workers']:>12d}  "
            f"{best['fps']:>9.2f}  {speedup:>11.2f}x  "
            f"{best['wall_elapsed']:>13.2f}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Multi-worker scaling benchmark for WAMOS pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("polar_path", help="Path to POLAR directory")
    parser.add_argument("--config", "-c", type=str, default=None, help="Config YAML")
    parser.add_argument(
        "-n", "--n-files", type=int, default=30, help="Number of files to process (default: 30)"
    )
    parser.add_argument(
        "--workers",
        type=str,
        default="1,2,4,8,16,20",
        help="Comma-separated worker counts (default: 1,2,4,8,16,20)",
    )
    parser.add_argument(
        "--frame", "-f", type=int, default=0, help="Frame index within each file (default: 0)"
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        default=["numpy", "both"],
        choices=["numpy", "numba", "pytorch", "both", "cupy", "cupy+numba"],
        help="Configs to test (default: numpy both)",
    )
    parser.add_argument("--json", type=str, default=None, help="Save results to JSON file")
    args = parser.parse_args()

    polar_path = Path(args.polar_path)
    if not polar_path.is_dir():
        print(f"ERROR: {polar_path} is not a directory")
        sys.exit(1)

    # Discover files
    filepaths = find_polar_files(polar_path, args.n_files)
    if not filepaths:
        print(f"ERROR: No .pol files found under {polar_path}")
        sys.exit(1)

    worker_counts = [int(w) for w in args.workers.split(",")]
    src_dir = str(Path(__file__).resolve().parents[1] / "src")

    selected = [CONFIGS[c] for c in args.configs]

    print(f"Scaling benchmark: {polar_path}")
    print(f"Files: {len(filepaths)}, Workers: {args.workers}")
    print(f"Configs: {', '.join(c['name'] for c in selected)}")
    print()

    config_results: dict[str, list[dict]] = {}

    for cfg in selected:
        cfg_name = cfg["name"]
        config_results[cfg_name] = []

        for n_w in worker_counts:
            label = f"{cfg_name} w={n_w}"
            print(f"  Running {label:<30s} ...", end="", flush=True)

            import time

            t0 = time.perf_counter()
            data = run_scaling_config(cfg, filepaths, args.config, n_w, args.frame, src_dir)
            elapsed = time.perf_counter() - t0

            if data is not None:
                config_results[cfg_name].append(data)
                fps = data.get("fps", 0)
                print(f"  done ({elapsed:.1f}s, {fps:.2f} FPS)")
            else:
                print(f"  FAILED ({elapsed:.1f}s)")

    # Print results
    print_scaling_table(config_results)
    print_cross_config_summary(config_results)

    if args.json:
        with open(args.json, "w") as f:
            json.dump(config_results, f, indent=2)
        print(f"\nResults saved to {args.json}")


if __name__ == "__main__":
    main()
