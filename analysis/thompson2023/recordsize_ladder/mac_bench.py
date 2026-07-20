"""Three-way SMB read benchmark: 128K loose vs 1M loose vs hour-tar.

Two repeats per arm with disjoint file/member selections so every read
is cold on both the client page cache and (post-eviction) the ARC.
Serial and 8-way parallel for the loose arms; ranged-serial and a 2 GB
mid-file stream for the tar arm.
"""

import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

W = Path("/Volumes/SeaChest/ARCTERX/2023/WAMOS")
DAY = "2023/05/15"


def t_serial(files):
    t0 = time.perf_counter()
    n = sum(len(p.read_bytes()) for p in files)
    return n, time.perf_counter() - t0


def t_parallel(files, workers=8):
    t0 = time.perf_counter()
    with ThreadPoolExecutor(workers) as ex:
        n = sum(ex.map(lambda p: len(p.read_bytes()), files))
    return n, time.perf_counter() - t0


def t_tar_ranged(tar, members):
    t0 = time.perf_counter()
    n = 0
    with open(tar, "rb") as f:
        for e in members:
            f.seek(e["offset"])
            n += len(f.read(e["size"]))
    return n, time.perf_counter() - t0


def t_tar_stream(tar, skip, nbytes):
    t0 = time.perf_counter()
    n = 0
    with open(tar, "rb") as f:
        f.seek(skip)
        while n < nbytes:
            b = f.read(8 << 20)
            if not b:
                break
            n += len(b)
    return n, time.perf_counter() - t0


def report(label, n, dt):
    print(f"{label:34s} {n/1e6:7.1f} MB {dt:7.2f} s {n/dt/1e6:6.0f} MB/s",
          flush=True)


def main():
    # repeat -> (loose slice, member slice): disjoint, never touched before
    # (200-432 and members 600-932 were consumed by the invalidated
    # reflink-clone run of 2026-07-20 morning)
    slices = [(slice(600, 616), slice(1200, 1216)),
              (slice(800, 816), slice(1500, 1516))]
    for rep, (fs, ms) in enumerate(slices, 1):
        print(f"--- repeat {rep} ---", flush=True)
        for arm, tree, hour in [("128K loose", "POLAR", "12"),
                                ("1M loose", "POLAR_1M", "13")]:
            files = sorted((W / tree / DAY / hour).glob("*.pol"))
            report(f"{arm} serial", *t_serial(files[fs]))
            shift = slice(fs.start + 16, fs.stop + 16)
            report(f"{arm} 8-way", *t_parallel(files[shift]))
        tar = W / "POLAR_HR" / DAY / "2023051514.pol.tar"
        idx = json.loads(
            (W / "POLAR_HR" / DAY / "2023051514.pol.tar.index.json").read_text())
        report("tar ranged serial", *t_tar_ranged(tar, idx["files"][ms]))
    report("tar stream 2GB (hour 12 tar, mid)",
           *t_tar_stream(W / "POLAR_HR" / DAY / "2023051512.pol.tar",
                         13 << 29, 2 << 30))  # bytes 6.5-8.5 GB: untouched


if __name__ == "__main__":
    main()
