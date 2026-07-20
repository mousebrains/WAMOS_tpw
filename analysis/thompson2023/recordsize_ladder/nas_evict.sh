#!/bin/bash
# Stream 165 GB (> ~100 GB ARC) through ARC to evict test data.
# Runs ON the NAS (via ssh 'bash -s'). Prints the eviction read rate.
F="/mnt/HDD/SeaChest/goflow/data/goes_2023_5min.nc"   # 569 GB
t0=$SECONDS
dd if="$F" of=/dev/null bs=8M count=20600 2>/dev/null
dt=$((SECONDS - t0))
echo "evicted: 165 GB in ${dt}s ($((165000 / (dt + 1))) MB/s)"
