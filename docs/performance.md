# Performance Guide

This document covers performance optimization strategies, benchmarking, and profiling guidance for wamos_tpw.

## Running Benchmarks

The package includes benchmarks using pytest-benchmark:

```bash
# Run all benchmarks
pytest tests/test_benchmarks.py -v

# Run with comparison to previous results
pytest tests/test_benchmarks.py --benchmark-compare

# Save benchmark results
pytest tests/test_benchmarks.py --benchmark-save=baseline

# Compare against saved baseline
pytest tests/test_benchmarks.py --benchmark-compare=baseline
```

## Profiling Large Datasets

### Using cProfile

```python
import cProfile
import pstats
from wamos_tpw import ProcessedFrames

# Profile the main processing loop
profiler = cProfile.Profile()
profiler.enable()

with ProcessedFrames(
    stime='2022-04-05 14:00',
    etime='2022-04-05 15:00',
    polar_path='/path/to/POLAR'
) as pf:
    for period, frames in pf.itergroups():
        frames = list(frames)
        pf.process(frames)

profiler.disable()

# Print top 20 functions by cumulative time
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(20)
```

### Using line_profiler

For line-by-line profiling of specific functions:

```bash
pip install line_profiler

# Add @profile decorator to functions of interest, then:
kernprof -l -v your_script.py
```

### Using memory_profiler

For memory usage analysis:

```bash
pip install memory_profiler

# Add @profile decorator to functions, then:
python -m memory_profiler your_script.py
```

## Key Optimization Strategies

### 1. Lazy Evaluation

Frame properties are computed on first access and cached:

```python
# In frame.py
@property
def intensity(self) -> np.ndarray:
    if self._intensity is None:
        self._intensity = self._data & _MASK_DATA  # Only computed once
    return self._intensity
```

### 2. Memory-Efficient Processing

The combine module uses streaming/chunked processing:

- **Metadata-only loading**: First pass loads only file headers to compute grid bounds
- **Chunked processing**: Frames processed in chunks of 50 (configurable)
- **In-place accumulation**: Grid sum/count arrays updated in-place
- **Aggressive cleanup**: `del` statements and `gc.collect()` after each chunk

```python
# Memory-efficient percentile estimation using reservoir sampling
max_samples = 100_000
sample_rate = min(1.0, max_samples / total_values)
```

### 3. Parallel Processing

The package uses different parallelization strategies:

- **File discovery**: `ProcessPoolExecutor` for parallel glob matching
- **Metadata loading**: `ThreadPoolExecutor` for I/O-bound operations
- **Frame processing**: `ProcessPoolExecutor` for CPU-bound processing

```python
# GIL detection for optimal threading
def _is_free_threaded() -> bool:
    if sys.version_info < (3, 13):
        return False
    return sys._is_gil_enabled() is False
```

### 4. Efficient Algorithms

- **O(n) quantile calculation**: Uses `np.partition` instead of full sorting
- **Circular statistics**: Single pass for both mean and std
- **Vectorized operations**: NumPy broadcasting instead of Python loops

```python
# Fast quantile using partition (O(n) vs O(n log n) for sort)
partitioned = np.partition(data_t, [k_low, k_high], axis=1)
```

## Memory Usage Guidelines

### Typical Memory Requirements

| Operation | Memory per Frame | Notes |
|-----------|-----------------|-------|
| Raw data loading | ~1.5 MB | uint16 array (720 x 752 x 2 bytes) |
| Intensity extraction | +1.5 MB | Cached property |
| Coordinate calculation | +12 MB | x/y arrays (720 x 752 x 8 bytes x 2) |
| Combined grid | ~7 MB | float64 (1200 x 1600 x 8 bytes) |

### Reducing Memory Usage

1. **Clear caches when done**:
   ```python
   frame.clear_cache()
   theta.clear_shadow_data()
   bearing.clear_cache()
   ```

2. **Use `metadata_only=True` for scanning**:
   ```python
   pf = PolarFile(path, metadata_only=True)
   # Access pf.header and pf.frame_metadata without loading frame data
   ```

3. **Process in smaller chunks**:
   ```bash
   wamos combine ... --max-frames=100
   ```

4. **Limit parallel workers**:
   ```bash
   wamos combine ... --workers=2
   ```

## Network File System Considerations

### Local vs NFS/SMB Performance

| File System | Discovery | Sequential Read | Random Access |
|-------------|-----------|-----------------|---------------|
| Local SSD | Fast | Fast | Fast |
| Local HDD | Fast | Moderate | Slow |
| NFS | Moderate | Moderate | Slow |
| SMB | Slow | Slow | Very Slow |

### Recommendations for Network Storage

1. **Increase file discovery parallelism**:
   ```python
   # For NFS/SMB, more threads help mask latency
   filenames = Filenames(stime, etime, path, workers=16)
   ```

2. **Use larger chunks for movie generation**:
   ```bash
   # Reduces number of I/O operations
   wamos combine ... --groupby=1h
   ```

3. **Consider local caching**:
   ```bash
   # Copy data locally before processing
   rsync -av server:/path/to/POLAR/ ./local_cache/
   wamos combine ... ./local_cache/
   ```

4. **Adjust timeouts**:
   Network latency can cause operations to take longer. The default 2-minute
   timeout for shell commands may need adjustment for very slow connections.

## GPU Acceleration

The pipeline supports optional PyTorch GPU acceleration for compute-heavy steps.
Install with `pip install wamos_tpw[gpu]` or `pip install torch>=2.0` separately.

GPU-accelerated modules: `destreak.py`, `deramp.py`, `bearing.py`, `grid.py`,
`hard_returns.py`, `interpolator_tasks.py`. Control via environment variables
or CLI flags:

```bash
# Environment variables
export WAMOS_NO_GPU=1    # Force CPU-only
export WAMOS_NO_NUMBA=1  # Disable Numba JIT

# CLI flags
wamos frame-pipeline ... --no-gpu --no-numba
wamos stream-pipeline ... --no-gpu
```

## Benchmark Results

### Running Benchmarks

```bash
# Single-frame timing + memory across 4 backend configs
python benchmarks/backend_benchmark.py /path/to/POLAR -n 30 --warmup 5

# Multi-worker scaling
python benchmarks/scaling_benchmark.py /path/to/POLAR -n 30 --workers 1,2,4,8,16

# Detailed per-step GPU vs CPU comparison
python benchmarks/gpu_comparison.py /path/to/POLAR
```

### NVIDIA DGX Spark (GB10)

Machine: NVIDIA Grace Blackwell GB10, 20 ARM cores, 119 GB unified memory, 128 GB GPU VRAM
Software: PyTorch 2.10.0+cu128, Numba 0.63.1, NumPy 2.3.5

#### Large Frame (2514 x 1552) -- 20 iterations, 5 warmup

**Frame Pipeline (median ms, lower is better)**

| Step | NumPy-only | Numba | PyTorch GPU | PyTorch+Numba | Best speedup |
|---|---|---|---|---|---|
| PPS | 0.034 | 0.045 | 0.046 | 0.046 | 1.0x |
| Theta | 0.251 | 0.281 | 0.281 | 0.286 | 1.0x |
| Range | 0.029 | 0.034 | 0.034 | 0.034 | 1.0x |
| Destreak | 31.13 | 30.68 | 17.97 | **10.40** | **2.99x** |
| Shadow | 4.15 | 4.27 | 4.18 | 4.13 | 1.0x |
| MaskShadow | 0.62 | 0.69 | **0.47** | 0.63 | 1.31x |
| Deramp | 6.98 | 7.15 | 5.57 | **2.88** | **2.43x** |
| Dewind | 5.88 | 6.33 | 6.89 | 6.83 | 1.0x |
| **Pipeline TOTAL** | 49.16 | 49.51 | 37.72 | **25.54** | **1.92x** |

**Grid Projection (median ms)**

| NumPy-only | Numba | PyTorch GPU | PyTorch+Numba | Best speedup |
|---|---|---|---|---|
| 55.09 | 55.33 | 5.02 | **3.81** | **14.48x** |

**Memory Usage (MB)**

| Metric | NumPy-only | Numba | PyTorch GPU | PyTorch+Numba |
|---|---|---|---|---|
| Baseline RSS | 89.3 | 146.5 | 582.9 | 640.1 |
| Peak RSS | 253.0 | 316.1 | 1216.5 | 1279.3 |
| Delta RSS | 163.6 | 169.6 | 633.6 | 639.2 |
| GPU Peak Alloc | N/A | N/A | 119.4 | 119.4 |

#### Small Frame (808 x 752) -- 30 iterations, 5 warmup

**Frame Pipeline (median ms)**

| Step | NumPy-only | Numba | PyTorch GPU | PyTorch+Numba | Best speedup |
|---|---|---|---|---|---|
| Destreak | 5.22 | 5.47 | **2.98** | 3.15 | **1.75x** |
| Shadow | 4.17 | 4.11 | 4.16 | 4.17 | 1.0x |
| Deramp | 1.17 | 1.09 | 1.26 | 1.54 | 1.1x |
| Dewind | 1.09 | 1.04 | 1.26 | 1.32 | 1.1x |
| **Pipeline TOTAL** | 12.09 | 12.00 | **10.78** | 11.00 | **1.12x** |

**Grid Projection (median ms)**

| NumPy-only | Numba | PyTorch GPU | PyTorch+Numba | Best speedup |
|---|---|---|---|---|
| 8.63 | 7.86 | 1.68 | **1.56** | **5.52x** |

**Memory Usage (MB)**

| Metric | NumPy-only | Numba | PyTorch GPU | PyTorch+Numba |
|---|---|---|---|---|
| Baseline RSS | 76.7 | 134.3 | 569.7 | 627.3 |
| Peak RSS | 124.1 | 187.5 | 1124.7 | 1187.5 |
| Delta RSS | 47.4 | 53.2 | 554.9 | 560.2 |
| GPU Peak Alloc | N/A | N/A | 18.6 | 18.6 |

#### Multi-Worker Scaling -- 200 large frames (2514 x 1552)

**NumPy-only**

| Workers | Wall(s) | FPS | Speedup | Efficiency |
|---|---|---|---|---|
| 1 | 15.05 | 13.29 | 1.00x | 100.0% |
| 2 | 7.86 | 25.46 | 1.92x | 95.8% |
| 4 | 4.19 | 47.76 | 3.59x | 89.8% |
| 8 | 2.71 | 73.72 | 5.55x | 69.3% |
| 12 | 2.40 | 83.39 | 6.27x | 52.3% |
| 16 | 2.26 | 88.61 | 6.67x | 41.7% |
| 20 | 2.28 | 87.62 | 6.59x | 33.0% |

**PyTorch GPU**

| Workers | Wall(s) | FPS | Speedup | Efficiency |
|---|---|---|---|---|
| 1 | 13.03 | 15.35 | 1.00x | 100.0% |
| 2 | 7.61 | 26.29 | 1.71x | 85.7% |
| 4 | 5.69 | 35.18 | 2.29x | 57.3% |
| 8 | 5.74 | 34.82 | 2.27x | 28.4% |
| 12 | 6.84 | 29.24 | 1.91x | 15.9% |
| 16 | 8.46 | 23.64 | 1.54x | 9.6% |

**Summary**

| Config | Best Workers | Peak FPS | Est. time for 100k files |
|---|---|---|---|
| **NumPy-only** | **16** | **88.61** | **~19 min** |
| Numba | 12 | 81.78 | ~20 min |
| PyTorch GPU | 4 | 35.18 | ~47 min |
| PyTorch + Numba | 4 | 35.29 | ~47 min |

For bulk processing of large datasets, use `--no-gpu --workers 16`:

```bash
wamos stream-pipeline ... --no-gpu --workers 16
```

#### Key Observations

1. NumPy-only with 16 workers is the fastest config for bulk throughput (88.6 FPS), 2.5x faster than the best GPU config
2. GPU scaling saturates at 4 workers then degrades -- all workers contend on the single GPU
3. CPU scaling is near-linear to 4 workers (90% efficiency), useful to 12-16 workers
4. GPU wins per-frame latency (1.92x pipeline, 14.5x grid projection) but can't parallelize
5. PyTorch adds ~940 MB RSS per worker vs ~200 MB for NumPy-only
6. For 100,000 files: `--no-gpu --workers 16` recommended (~19 min vs ~47 min with GPU)

### Apple M4 Max

Machine: Apple M4 Max, 16 CPU cores, 40 GPU cores (Metal 4), 128 GB unified memory
Software: PyTorch 2.10.0 (MPS), Numba 0.63.1, NumPy 2.3.5, Python 3.14.3

#### Large Frame (2514 x 1552) -- 20 iterations, 5 warmup

**Frame Pipeline (median ms, lower is better)**

| Step | NumPy-only | Numba | PyTorch MPS | PyTorch+Numba | Best speedup |
|---|---|---|---|---|---|
| PPS | 0.017 | 0.019 | 0.024 | 0.024 | 1.0x |
| Theta | 0.145 | 0.157 | 0.178 | 0.179 | 1.0x |
| Range | 0.017 | 0.018 | 0.022 | 0.022 | 1.0x |
| Destreak | 11.12 | 11.16 | **10.85** | 10.99 | **1.02x** |
| Shadow | 3.69 | 3.73 | 3.79 | 3.76 | 1.0x |
| MaskShadow | 0.25 | 0.25 | 0.26 | 0.26 | 1.0x |
| Deramp | 5.06 | 5.11 | **3.85** | 4.01 | **1.31x** |
| Dewind | 4.66 | 4.65 | 4.71 | 4.74 | 1.0x |
| **Pipeline TOTAL** | 25.07 | 25.20 | **23.80** | 24.00 | **1.05x** |

**Grid Projection (median ms)**

| NumPy-only | Numba | PyTorch MPS | PyTorch+Numba | Best speedup |
|---|---|---|---|---|
| 9.74 | 9.80 | **7.87** | 8.43 | **1.24x** |

**Memory Usage (MB)**

| Metric | NumPy-only | Numba | PyTorch MPS | PyTorch+Numba |
|---|---|---|---|---|
| Baseline RSS | 100.8 | 141.9 | 273.2 | 311.5 |
| Peak RSS | 331.7 | 373.9 | 435.8 | 471.8 |
| Delta RSS | 230.9 | 232.0 | 162.6 | 160.3 |

#### Small Frame (808 x 752) -- 30 iterations, 5 warmup

**Frame Pipeline (median ms)**

| Step | NumPy-only | Numba | PyTorch MPS | PyTorch+Numba | Best speedup |
|---|---|---|---|---|---|
| Destreak | **1.71** | 1.74 | 3.36 | 4.02 | 1.0x |
| Shadow | **3.51** | 3.62 | 3.69 | 3.71 | 1.0x |
| Deramp | 0.91 | **0.90** | 1.70 | 1.73 | 1.0x |
| Dewind | **0.84** | 0.85 | 0.90 | 0.91 | 1.0x |
| **Pipeline TOTAL** | **7.12** | 7.28 | 10.14 | 10.64 | **1.0x** |

**Grid Projection (median ms)**

| NumPy-only | Numba | PyTorch MPS | PyTorch+Numba | Best speedup |
|---|---|---|---|---|
| **1.48** | 1.55 | 3.54 | 3.84 | **1.0x** |

**Memory Usage (MB)**

| Metric | NumPy-only | Numba | PyTorch MPS | PyTorch+Numba |
|---|---|---|---|---|
| Baseline RSS | 84.3 | 125.3 | 255.4 | 294.8 |
| Peak RSS | 139.7 | 182.0 | 349.9 | 387.7 |
| Delta RSS | 55.4 | 56.8 | 94.5 | 92.9 |

#### Multi-Worker Scaling -- 30 files, Small Frame

| Config | Best Workers | Peak FPS | Peak Speedup |
|---|---|---|---|
| NumPy-only | 8 | 41.18 | 1.14x |
| Numba | 4 | 32.49 | 1.04x |
| PyTorch MPS | 1 | 16.63 | 1.00x |
| PyTorch + Numba | 2 | 12.18 | 1.00x |

NumPy-only scales modestly to 8 workers on this workload.  PyTorch MPS
degrades with multiple workers -- MPS serializes GPU commands from
separate processes, adding overhead without parallel execution benefit.

#### Key Observations

1. The M4 Max CPU is ~2x faster than DGX Spark ARM cores: NumPy-only pipeline is 25 ms vs 49 ms on identical large frames
2. PyTorch MPS provides only modest gains on large frames (1.05x pipeline, 1.24x grid) -- Apple's fast CPU and unified memory reduce the GPU advantage compared to discrete CUDA GPUs
3. On small frames, PyTorch MPS is slower than CPU (0.70x) -- MPS dispatch overhead exceeds the compute savings at this scale
4. Deramp sees the largest MPS benefit (1.31x) via GPU nanmean reduction; Destreak gain is marginal (1.02x) because OpenCV `filter2D` is already highly optimized on ARM NEON
5. Memory overhead: PyTorch MPS adds ~170-210 MB baseline RSS (Metal runtime), lower than CUDA's ~500 MB overhead on DGX
6. For Apple Silicon, **NumPy-only is the recommended configuration** -- it has the fastest small-frame throughput, lowest memory, and best multi-worker scaling

## Troubleshooting Performance Issues

### Slow File Discovery

```bash
# Check file count
find /path/to/POLAR -name "*.pol*" | wc -l

# Profile discovery
python -c "
from wamos_tpw import Filenames
import time
start = time.time()
fn = Filenames('2022040400', '2022040600', '/path/to/POLAR')
print(f'Found {len(fn)} files in {time.time()-start:.2f}s')
"
```

### High Memory Usage

```bash
# Monitor memory during processing
watch -n 1 'ps aux | grep wamos | grep -v grep'

# Or use memory_profiler
mprof run wamos combine ...
mprof plot
```

### Slow Processing

```bash
# Enable debug logging
wamos --verbose combine ...

# Or profile specific operations
python -m cProfile -s cumulative -m wamos_tpw.cli combine ...
```
