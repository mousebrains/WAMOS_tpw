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

## Benchmark Results

Typical performance on a MacBook Pro M1 with local SSD:

| Operation | Time | Notes |
|-----------|------|-------|
| File discovery (1000 files) | ~0.5s | Parallel glob |
| Load single .pol file | ~50ms | Including decompression |
| Deramp single frame | ~10ms | |
| Destreak single frame | ~15ms | |
| Combine 100 frames | ~2s | With processing |
| Grid single frame | ~100ms | Earth coordinates |

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
