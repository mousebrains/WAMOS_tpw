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

The pipeline uses a **hybrid approach**: CPU (NumPy/Numba) for the per-frame
pipeline (destreak, deramp, dewind) which scales well across workers, and
**CuPy GPU** for grid projection and hard-return sweeps where GPU provides
12-16x speedup.

```bash
pip install wamos_tpw[cupy]      # CuPy (cupy-cuda13x>=14.0)
```

CuPy-accelerated modules: `grid.py`, `hard_returns.py`, `interpolator_tasks.py`.

**Backend priority**: CuPy > Numba > NumPy (pure CPU fallback).

Control via environment variables or CLI flags:

```bash
# Environment variables
export WAMOS_NO_CUPY=1   # Disable CuPy GPU
export WAMOS_NO_NUMBA=1  # Disable Numba JIT

# CLI flags
wamos stream-pipeline ... --no-cupy --no-numba
```

## Benchmark Results

### Running Benchmarks

```bash
# Single-frame timing + memory across backend configs
python benchmarks/backend_benchmark.py /path/to/POLAR -n 20 --warmup 3

# Select specific configs (numpy, numba, cupy, cupy+numba)
python benchmarks/backend_benchmark.py /path/to/POLAR --configs numpy cupy cupy+numba

# Multi-worker scaling
python benchmarks/scaling_benchmark.py /path/to/POLAR -n 200 --workers 1,2,4,8,16,20

# Multi-worker with specific configs
python benchmarks/scaling_benchmark.py /path/to/POLAR --configs numpy numba cupy cupy+numba

# Detailed per-step GPU vs CPU comparison
python benchmarks/gpu_comparison.py /path/to/POLAR
```

### NVIDIA DGX Spark (GB10)

Machine: NVIDIA Grace Blackwell GB10, 20 ARM cores, 119 GB unified memory, 128 GB GPU VRAM
Software: CuPy 14.0.0 (CUDA 13.0), PyTorch 2.10.0+cu128, Numba 0.63.1, NumPy 2.3.5

#### Large Frame (2514 x 1552) -- 20 iterations, 3 warmup

**Frame Pipeline (median ms, lower is better)**

Pipeline steps run on CPU only (hybrid approach). CuPy is used for grid projection.

| Step | NumPy-only | Numba | Best speedup |
|---|---|---|---|
| Destreak | 37.71 | **37.42** | 1.01x |
| Shadow | 4.28 | **4.27** | 1.0x |
| Deramp | **7.61** | 7.78 | 1.0x |
| Dewind | **6.37** | 6.47 | 1.0x |
| **Pipeline TOTAL** | 57.32 | **56.88** | **1.01x** |

**Grid Projection (median ms)** -- CuPy GPU

| NumPy-only | Numba | CuPy | CuPy+Numba | Best speedup |
|---|---|---|---|---|
| 61.62 | 61.96 | **4.73** | 4.76 | **13.02x** |

**Memory Usage (MB)**

| Metric | NumPy-only | Numba | CuPy+Numba |
|---|---|---|---|
| Baseline RSS | 89.4 | 146.7 | 444.4 |
| Peak RSS | 254.7 | 316.2 | 602.4 |
| Delta RSS | 165.3 | 169.5 | 158.0 |

#### Small Frame (838 x 752) -- 20 iterations, 3 warmup

**Frame Pipeline (median ms)**

| Step | NumPy-only | Numba | Best speedup |
|---|---|---|---|
| Destreak | **6.05** | 6.28 | 1.0x |
| Shadow | 4.18 | **4.16** | 1.0x |
| Deramp | 1.06 | **1.05** | 1.0x |
| Dewind | 1.17 | **1.16** | 1.0x |
| **Pipeline TOTAL** | **12.89** | 13.06 | **1.0x** |

**Grid Projection (median ms)** -- CuPy GPU

| NumPy-only | Numba | CuPy | CuPy+Numba | Best speedup |
|---|---|---|---|---|
| 8.70 | 8.66 | **1.12** | 1.25 | **7.78x** |

**Memory Usage (MB)**

| Metric | NumPy-only | Numba | CuPy+Numba |
|---|---|---|---|
| Baseline RSS | 76.8 | 133.8 | 432.0 |
| Peak RSS | 125.0 | 187.9 | 513.2 |
| Delta RSS | 48.2 | 54.1 | 81.1 |

#### Multi-Worker Scaling -- 200 large frames (2514 x 1552)

These benchmarks measure per-frame pipeline throughput only (no grid projection / merging).

**NumPy-only**

| Workers | Wall(s) | FPS | Speedup | Efficiency | Mean RSS |
|---|---|---|---|---|---|
| 1 | 14.91 | 13.42 | 1.00x | 100.0% | 200 MB |
| 2 | 7.96 | 25.14 | 1.87x | 93.7% | 200 MB |
| 4 | 4.44 | 45.05 | 3.36x | 83.9% | 198 MB |
| 8 | 2.79 | 71.79 | 5.35x | 66.9% | 198 MB |
| 16 | 2.33 | 85.97 | 6.41x | 40.1% | 197 MB |
| 20 | 2.30 | 87.12 | 6.49x | 32.5% | 196 MB |

**Numba**

| Workers | Wall(s) | FPS | Speedup | Efficiency | Mean RSS |
|---|---|---|---|---|---|
| 1 | 16.50 | 12.12 | 1.00x | 100.0% | 259 MB |
| 2 | 8.06 | 24.80 | 2.05x | 102.3% | 258 MB |
| 4 | 4.47 | 44.70 | 3.69x | 92.2% | 258 MB |
| 8 | 2.90 | 69.08 | 5.70x | 71.3% | 256 MB |
| 16 | 2.41 | 83.03 | 6.85x | 42.8% | 255 MB |
| 20 | 2.48 | 80.80 | 6.67x | 33.3% | 254 MB |

**Cross-Config Summary (Large Frames)**

| Config | Best Workers | Peak FPS | Peak Speedup | Wall(s) @best |
|---|---|---|---|---|
| **NumPy-only** | **20** | **87.12** | **6.49x** | **2.30** |
| Numba | 16 | 83.03 | 6.85x | 2.41 |

#### Streaming Pipeline -- 1 hour, 5-min windows, 50% overlap

Full end-to-end benchmark: 2529 large frames (2514 x 1552), 22 merged images,
5-minute windows with 50% overlap. Includes frame processing, grid projection,
and NetCDF output. Auto worker count (20 cores).

| Config | Total Time | Merged Images | Speedup |
|---|---|---|---|
| **CuPy+Numba** (hybrid) | **80.7s** | 22 | **1.73x** |
| Numba | 128.3s | 22 | 1.09x |
| NumPy-only | 139.5s | 22 | 1.00x |

The hybrid approach (CPU pipeline + CuPy grid projection) is **1.73x faster**
than pure NumPy for the complete streaming pipeline because CuPy's 13x grid
projection speedup (4.7ms vs 62ms) compounds across 2529 frames being projected
into 22 overlapping windows.

#### Key Observations

1. **CuPy+Numba hybrid wins end-to-end streaming** (80.7s vs 139.5s NumPy-only for 1 hour of data)
2. **CuPy grid projection provides 13x speedup** (4.7ms vs 62ms) which is the key GPU benefit
3. Pipeline steps (destreak, deramp, dewind) run on CPU with good multi-worker scaling
4. CPU scaling is near-linear to 4 workers (84-93% efficiency), useful to 16-20 workers
5. **Memory per worker**: NumPy ~200 MB, Numba ~260 MB, CuPy+Numba ~600 MB
6. CuPy is fork-safe: CUDA context creation is deferred to first GPU operation in each worker process

### Apple M4 Max

Machine: Apple M4 Max, 16 CPU cores, 40 GPU cores (Metal 4), 128 GB unified memory
Software: Numba 0.63.1, NumPy 2.3.5, Python 3.14.3

Note: CuPy does not support Apple Silicon (no Metal/MPS backend). NumPy-only is
the recommended configuration for Apple Silicon.

#### Large Frame (2514 x 1552) -- 20 iterations, 5 warmup

**Frame Pipeline (median ms)**

| Step | NumPy-only | Numba |
|---|---|---|
| Destreak | **11.12** | 11.16 |
| Shadow | **3.69** | 3.73 |
| Deramp | **5.06** | 5.11 |
| Dewind | 4.66 | **4.65** |
| **Pipeline TOTAL** | **25.07** | 25.20 |

**Grid Projection (median ms)**

| NumPy-only | Numba |
|---|---|
| **9.74** | 9.80 |

**Memory Usage (MB)**

| Metric | NumPy-only | Numba |
|---|---|---|
| Baseline RSS | 100.8 | 141.9 |
| Peak RSS | 331.7 | 373.9 |
| Delta RSS | 230.9 | 232.0 |

#### Key Observations

1. The M4 Max CPU is ~2x faster than DGX Spark ARM cores: NumPy-only pipeline is 25 ms vs 57 ms on identical large frames
2. Numba provides no measurable benefit on Apple Silicon for this workload
3. For Apple Silicon, **NumPy-only is the recommended configuration**

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
