#! /usr/bin/env python3
#
# Play with scipy's ndimage label in the context of streak removal
#
# Jan-2026, Pat Welch, pat@mousebrains.com

from scipy.ndimage import label, convolve1d
from scipy.signal import convolve2d
import numpy as np
import sys
import time

# Horizontal connectivity kernel
kernel = np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]], dtype=np.uint8)

# frame intensity data
dt = 0
t0 = time.perf_counter()
a = np.random.randint(0, 4095, (5, 100)).astype("uint16")
t1 = time.perf_counter()
dt += t1 - t0
print("a", a.shape, a.dtype, (t1 - t0) * 1e3)
print(a)

t0 = time.perf_counter()
q = a > 2048
t1 = time.perf_counter()
dt += t1 - t0
print("q", q.shape, q.dtype, (t1 - t0) * 1e3)
print(q.astype(np.uint8))

# Remove short non-streaks

t0 = time.perf_counter()
[b, nFeatures] = label(~q, structure=kernel)
cnt = np.bincount(b.ravel())
RL = cnt[b] # Run Length image
t1 = time.perf_counter()
dt += t1 - t0
print("not nFeatures", nFeatures)
print("b", b.shape, b.dtype, (t1 - t0) * 1e3)
print(b)
print("cnt", cnt.shape, cnt.dtype)
print(cnt)
print("RL", RL.shape, RL.dtype)
print(RL)

# Convert short non-streaks to streaks
t0 = time.perf_counter()
q1 = q.copy()
q1[RL < 2] = True
t1 = time.perf_counter()
dt += t1 - t0
print("q1", q1.shape, q1.dtype, (t1 - t0) * 1e3)
print(q1.astype(np.uint8))
print(q1.astype(np.int8) - q.astype(np.int8))

t0 = time.perf_counter()
[b, nFeatures] = label(q1, structure=kernel)
cnt = np.bincount(b.ravel())
RL = cnt[b] # Run Length image
t1 = time.perf_counter()
dt += t1 - t0
print("nFeatures", nFeatures)
print("b", b.shape, b.dtype, (t1 - t0) * 1e3)
print(b)
print("cnt", cnt.shape, cnt.dtype)
print(cnt)
print("RL", RL.shape, RL.dtype)
print(RL)

print("Total time (ms):", dt * 1e3)
