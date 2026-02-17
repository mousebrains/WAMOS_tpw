#!/usr/bin/env python3
"""
Backend comparison benchmark for WAMOS pipeline.

Tests 6 configurations of the actual pipeline code:
  1. NumPy-only       (no GPU, no CuPy, no Numba)
  2. Numba            (no GPU, no CuPy, Numba enabled)
  3. PyTorch GPU      (GPU enabled, no CuPy, no Numba)
  4. PyTorch + Numba  (GPU enabled, no CuPy, both enabled)
  5. CuPy             (no PyTorch GPU, CuPy enabled, no Numba)
  6. CuPy + Numba     (no PyTorch GPU, CuPy enabled, Numba enabled)

Each configuration runs in a subprocess with the appropriate env vars
so that module-level detection picks up the correct settings.

Usage:
    python benchmarks/backend_benchmark.py /path/to/POLAR
    python benchmarks/backend_benchmark.py /path/to/file.pol.gz
    python benchmarks/backend_benchmark.py /path/to/POLAR -n 30 --warmup 5
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

# ── Configuration matrix ──

CONFIGS = [
    {
        "name": "NumPy-only",
        "env": {"WAMOS_NO_GPU": "1", "WAMOS_NO_NUMBA": "1", "WAMOS_NO_CUPY": "1"},
    },
    {
        "name": "Numba",
        "env": {"WAMOS_NO_GPU": "1", "WAMOS_NO_NUMBA": "", "WAMOS_NO_CUPY": "1"},
    },
    {
        "name": "PyTorch GPU",
        "env": {"WAMOS_NO_GPU": "", "WAMOS_NO_NUMBA": "1", "WAMOS_NO_CUPY": "1"},
    },
    {
        "name": "PyTorch + Numba",
        "env": {"WAMOS_NO_GPU": "", "WAMOS_NO_NUMBA": "", "WAMOS_NO_CUPY": "1"},
    },
    {
        "name": "CuPy",
        "env": {"WAMOS_NO_GPU": "1", "WAMOS_NO_NUMBA": "1", "WAMOS_NO_CUPY": ""},
    },
    {
        "name": "CuPy + Numba",
        "env": {"WAMOS_NO_GPU": "1", "WAMOS_NO_NUMBA": "", "WAMOS_NO_CUPY": ""},
    },
]

# ── Worker script (run in subprocess) ──

WORKER_SCRIPT = r'''
"""Worker subprocess for backend benchmark."""
import gc
import json
import resource
import statistics
import sys
import time

import numpy as np

sys.path.insert(0, "{src_dir}")

from wamos_tpw.backend import HAS_CUPY_GPU, HAS_NUMBA, HAS_TORCH_GPU
from wamos_tpw.config import Config
from wamos_tpw.polarfile import PolarFile


def bench_pipeline(frame, config, n_warmup, n_iter):
    """Benchmark FramePipeline."""
    from wamos_tpw.frame_pipeline import FramePipeline

    # Warmup
    for _ in range(n_warmup):
        FramePipeline(frame, config=config, qTiming=True)
        gc.collect()

    results = []
    for _ in range(n_iter):
        gc.collect()
        t0 = time.perf_counter()
        fp = FramePipeline(frame, config=config, qTiming=True)
        elapsed = time.perf_counter() - t0
        timings = dict(fp.timings)
        timings["Pipeline_TOTAL"] = elapsed
        results.append(timings)

    return results


def bench_grid_projection(frame, config, n_warmup, n_iter):
    """Benchmark grid projection."""
    from wamos_tpw.grid import GridParams, project_frame_to_common_grid
    from wamos_tpw.range import Range
    from wamos_tpw.theta import Theta

    theta = Theta(frame)
    rng = Range(frame)
    gr = rng.ground_range
    n_bearings = frame.shape[0]

    # Create a test grid
    max_range = float(gr[-1])
    spacing = max_range * 2 / 512
    n_x = n_y = 512
    x_edges = np.linspace(-max_range, max_range, n_x + 1)
    y_edges = np.linspace(-max_range, max_range, n_y + 1)

    gp = GridParams(
        x_edges=x_edges,
        y_edges=y_edges,
        x_edges_abs=x_edges + 1e6,
        y_edges_abs=y_edges + 5e6,
        grid_spacing=spacing,
        utm_zone=10,
        hemisphere="north",
        center_lat=45.0,
        center_lon=-122.0,
        ref_lat=45.0,
        ref_lon=-122.0,
        m_per_deg_lon=78846.0,
        n_x=n_x,
        n_y=n_y,
    )

    intensity = frame.intensity.astype(np.float32)
    theta_arr = theta.theta
    lats = np.full(n_bearings, 45.0)
    lons = np.full(n_bearings, -122.0)
    headings = np.full(n_bearings, 90.0)

    # Warmup
    for _ in range(n_warmup):
        project_frame_to_common_grid(intensity, theta_arr, gr, lats, lons, headings, gp)
        gc.collect()

    results = []
    for _ in range(n_iter):
        gc.collect()
        t0 = time.perf_counter()
        project_frame_to_common_grid(intensity, theta_arr, gr, lats, lons, headings, gp)
        elapsed = time.perf_counter() - t0
        results.append({{"GridProjection": elapsed}})

    return results


def main():
    filepath = "{filepath}"
    config_path = {config_repr}
    n_warmup = {n_warmup}
    n_iter = {n_iter}

    config = Config(config_path) if config_path else Config()
    pf = PolarFile(filepath, config=config)
    frame = pf.frame()

    # Capture baseline RSS after imports and data loading
    baseline_rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    output = {{
        "HAS_CUPY_GPU": HAS_CUPY_GPU,
        "HAS_TORCH_GPU": HAS_TORCH_GPU,
        "HAS_NUMBA": HAS_NUMBA,
        "frame_shape": list(frame.shape),
    }}

    # Run pipeline benchmark
    pipeline_results = bench_pipeline(frame, config, n_warmup, n_iter)
    output["pipeline"] = pipeline_results

    # Run grid projection benchmark
    grid_results = bench_grid_projection(frame, config, n_warmup, n_iter)
    output["grid"] = grid_results

    # Capture peak RSS after all benchmarks
    peak_rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    memory = {{
        "baseline_rss_mb": baseline_rss_kb / 1024,
        "peak_rss_mb": peak_rss_kb / 1024,
        "delta_rss_mb": (peak_rss_kb - baseline_rss_kb) / 1024,
    }}

    # GPU memory stats
    if HAS_TORCH_GPU:
        import torch
        if torch.cuda.is_available():
            memory["gpu_peak_allocated_mb"] = torch.cuda.max_memory_allocated() / (1024 * 1024)
            memory["gpu_current_allocated_mb"] = torch.cuda.memory_allocated() / (1024 * 1024)
            free, total = torch.cuda.mem_get_info()
            memory["gpu_free_mb"] = free / (1024 * 1024)
            memory["gpu_total_mb"] = total / (1024 * 1024)

    output["memory"] = memory

    print(json.dumps(output))


main()
'''


def run_config(cfg, filepath, config_path, n_warmup, n_iter, src_dir):
    """Run one configuration in a subprocess."""
    env = dict(os.environ)
    # Clear first, then set
    env.pop("WAMOS_NO_GPU", None)
    env.pop("WAMOS_NO_NUMBA", None)
    env.pop("WAMOS_NO_CUPY", None)
    for k, v in cfg["env"].items():
        if v:
            env[k] = v
        else:
            env.pop(k, None)

    script = WORKER_SCRIPT.format(
        src_dir=src_dir,
        filepath=filepath,
        config_repr=repr(config_path) if config_path else "None",
        n_warmup=n_warmup,
        n_iter=n_iter,
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=env,
        timeout=600,
    )

    if result.returncode != 0:
        print(f"\n  ERROR running {cfg['name']}:", file=sys.stderr)
        print(
            result.stderr[-2000:] if len(result.stderr) > 2000 else result.stderr, file=sys.stderr
        )
        return None

    # Parse JSON from stdout (skip any warnings on stderr)
    try:
        return json.loads(result.stdout.strip())
    except json.JSONDecodeError:
        print(f"\n  ERROR parsing output for {cfg['name']}:", file=sys.stderr)
        print(f"  stdout: {result.stdout[:500]}", file=sys.stderr)
        return None


def aggregate_timings(results_list, key):
    """Aggregate timing results into statistics."""
    all_steps = set()
    for r in results_list:
        for entry in r.get(key, []):
            all_steps.update(entry.keys())

    stats = {}
    for step in sorted(all_steps):
        vals = [entry[step] * 1000 for entry in r.get(key, []) if step in entry]
        if vals:
            import statistics as st

            stats[step] = {
                "median_ms": st.median(vals),
                "mean_ms": st.mean(vals),
                "std_ms": st.stdev(vals) if len(vals) > 1 else 0.0,
                "min_ms": min(vals),
                "max_ms": max(vals),
            }
    return stats


def print_comparison_table(all_results):
    """Print a formatted comparison table."""
    import statistics as st

    # Gather all pipeline steps
    pipeline_steps = [
        "PPS",
        "Theta",
        "Range",
        "Destreak",
        "Shadow",
        "theta_bias",
        "MaskShadow",
        "Deramp",
        "Dewind",
        "Pipeline_TOTAL",
    ]
    grid_step = "GridProjection"

    config_names = [r["name"] for r in all_results]
    col_w = max(14, max(len(n) for n in config_names) + 2)

    # ── Pipeline table ──
    print(f"\n{'=' * (25 + (col_w + 2) * len(config_names) + col_w + 2)}")
    print("  FRAME PIPELINE BENCHMARK  (median ms, lower is better)")
    print(f"{'=' * (25 + (col_w + 2) * len(config_names) + col_w + 2)}")

    header = f"{'Step':<25}"
    for name in config_names:
        header += f"  {name:>{col_w}}"
    header += f"  {'Best speedup':>{col_w}}"
    print(header)
    print("-" * len(header))

    for step in pipeline_steps:
        if step == "Pipeline_TOTAL":
            print("-" * len(header))

        row = f"{step:<25}"
        medians = []
        for r in all_results:
            vals = [e.get(step, 0) * 1000 for e in r.get("pipeline", []) if step in e]
            if vals:
                med = st.median(vals)
                medians.append(med)
                row += f"  {med:>{col_w}.3f}"
            else:
                medians.append(None)
                row += f"  {'N/A':>{col_w}}"

        # Speedup (numpy-only / best)
        valid_medians = [m for m in medians if m is not None and m > 0]
        if valid_medians and medians[0] is not None and medians[0] > 0:
            best = min(valid_medians)
            speedup = medians[0] / best
            row += f"  {speedup:>{col_w}.2f}x"
        else:
            row += f"  {'':>{col_w}}"
        print(row)

    # ── Grid projection table ──
    print(f"\n{'=' * (25 + (col_w + 2) * len(config_names) + col_w + 2)}")
    print("  GRID PROJECTION BENCHMARK  (median ms, lower is better)")
    print(f"{'=' * (25 + (col_w + 2) * len(config_names) + col_w + 2)}")
    print(header)
    print("-" * len(header))

    row = f"{grid_step:<25}"
    medians = []
    for r in all_results:
        vals = [e.get(grid_step, 0) * 1000 for e in r.get("grid", []) if grid_step in e]
        if vals:
            med = st.median(vals)
            medians.append(med)
            row += f"  {med:>{col_w}.3f}"
        else:
            medians.append(None)
            row += f"  {'N/A':>{col_w}}"

    valid_medians = [m for m in medians if m is not None and m > 0]
    if valid_medians and medians[0] is not None and medians[0] > 0:
        best = min(valid_medians)
        speedup = medians[0] / best
        row += f"  {speedup:>{col_w}.2f}x"
    print(row)

    # ── Summary ──
    print(f"\n{'─' * 70}")
    print("  SUMMARY")
    print(f"{'─' * 70}")

    numpy_pipeline = None
    numpy_grid = None
    for r in all_results:
        pipe_vals = [
            e.get("Pipeline_TOTAL", 0) * 1000
            for e in r.get("pipeline", [])
            if "Pipeline_TOTAL" in e
        ]
        grid_vals = [
            e.get("GridProjection", 0) * 1000 for e in r.get("grid", []) if "GridProjection" in e
        ]
        pipe_med = st.median(pipe_vals) if pipe_vals else 0
        grid_med = st.median(grid_vals) if grid_vals else 0

        if r["name"] == "NumPy-only":
            numpy_pipeline = pipe_med
            numpy_grid = grid_med

        pipe_speedup = numpy_pipeline / pipe_med if numpy_pipeline and pipe_med > 0 else 1.0
        grid_speedup = numpy_grid / grid_med if numpy_grid and grid_med > 0 else 1.0

        backends = []
        if r.get("HAS_CUPY_GPU"):
            backends.append("CuPy/CUDA")
        if r.get("HAS_TORCH_GPU"):
            backends.append("PyTorch/CUDA")
        if r.get("HAS_NUMBA"):
            backends.append("Numba")
        if not backends:
            backends.append("NumPy")
        backend_str = " + ".join(backends)

        print(
            f"  {r['name']:<20s}  Pipeline: {pipe_med:7.2f} ms ({pipe_speedup:.2f}x)  "
            f"Grid: {grid_med:7.2f} ms ({grid_speedup:.2f}x)  [{backend_str}]"
        )

    frame_shape = all_results[0].get("frame_shape", [0, 0]) if all_results else [0, 0]
    print(f"\n  Frame: {frame_shape[0]} x {frame_shape[1]}")

    # ── Memory comparison ──
    has_any_memory = any(r.get("memory") for r in all_results)
    if has_any_memory:
        print(f"\n{'=' * (25 + (col_w + 2) * len(config_names))}")
        print("  MEMORY USAGE  (MB)")
        print(f"{'=' * (25 + (col_w + 2) * len(config_names))}")

        mem_header = f"{'Metric':<25}"
        for name in config_names:
            mem_header += f"  {name:>{col_w}}"
        print(mem_header)
        print("-" * len(mem_header))

        mem_rows = [
            ("Baseline RSS", "baseline_rss_mb"),
            ("Peak RSS", "peak_rss_mb"),
            ("Delta RSS", "delta_rss_mb"),
            ("GPU Peak Alloc", "gpu_peak_allocated_mb"),
            ("GPU Current Alloc", "gpu_current_allocated_mb"),
            ("GPU Free", "gpu_free_mb"),
            ("GPU Total", "gpu_total_mb"),
        ]

        for label, key in mem_rows:
            row = f"{label:<25}"
            for r in all_results:
                mem = r.get("memory", {})
                val = mem.get(key)
                if val is not None:
                    row += f"  {val:>{col_w}.1f}"
                else:
                    row += f"  {'N/A':>{col_w}}"
            print(row)


def main():
    parser = argparse.ArgumentParser(
        description="Backend comparison benchmark for WAMOS pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("polar_path", help="Path to POLAR directory or .pol file")
    parser.add_argument("--config", "-c", type=str, default=None, help="Config YAML")
    parser.add_argument("--warmup", type=int, default=3, help="Warmup iterations (default: 3)")
    parser.add_argument("-n", "--iterations", type=int, default=20, help="Iterations (default: 20)")
    parser.add_argument("--json", type=str, default=None, help="Save results to JSON file")
    parser.add_argument(
        "--configs",
        nargs="+",
        default=None,
        choices=["numpy", "numba", "pytorch", "both", "cupy", "cupy+numba"],
        help="Configs to test (default: all)",
    )
    args = parser.parse_args()

    # Find polar file
    polar_path = Path(args.polar_path)
    if polar_path.is_file():
        filepath = str(polar_path)
    else:
        files = sorted(polar_path.rglob("*.pol*"))
        if not files:
            print(f"ERROR: No .pol files found under {polar_path}")
            sys.exit(1)
        filepath = str(files[0])

    src_dir = str(Path(__file__).resolve().parents[1] / "src")

    # Filter configs
    config_map = {"numpy": 0, "numba": 1, "pytorch": 2, "both": 3, "cupy": 4, "cupy+numba": 5}
    if args.configs:
        selected = [CONFIGS[config_map[c]] for c in args.configs]
    else:
        selected = list(CONFIGS)

    print(f"Benchmark: {filepath}")
    print(f"Warmup: {args.warmup}, Iterations: {args.iterations}")
    print(f"Configs: {', '.join(c['name'] for c in selected)}")
    print()

    all_results = []
    for cfg in selected:
        print(f"  Running {cfg['name']:<20s} ...", end="", flush=True)
        t0 = __import__("time").perf_counter()
        data = run_config(cfg, filepath, args.config, args.warmup, args.iterations, src_dir)
        elapsed = __import__("time").perf_counter() - t0
        if data is not None:
            data["name"] = cfg["name"]
            all_results.append(data)
            print(f"  done ({elapsed:.1f}s)")
        else:
            print(f"  FAILED ({elapsed:.1f}s)")

    if not all_results:
        print("ERROR: No successful runs")
        sys.exit(1)

    print_comparison_table(all_results)

    if args.json:
        with open(args.json, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to {args.json}")


if __name__ == "__main__":
    main()
