#!/usr/bin/env python3
#
# Bundle per-hour WAMOS .pol directories into indexed tar chunks
#
# Turns YYYY/MM/DD/HH/*.pol* trees (~1-2k files/hour) into one
# uncompressed tar per hour plus a JSON byte-offset index, so a reader
# can open a single file and seek directly to any .pol member. Designed
# to run ON the TrueNAS host (no network in the rewrite path); stdlib
# only. Originals are never modified or deleted.
#
# Each hour is atomic and idempotent: data is written to <tar>.partial,
# the finished tar is re-scanned to build/verify the index, then both
# files are renamed into place. Re-running skips completed hours, so
# the job can be interrupted and resumed at any time.
#
# Usage (on TrueNAS, in tmux or under nohup):
#   python3 polchunk.py /mnt/pool/.../POLAR /mnt/pool/.../POLAR_HR \
#       --start 2023051500 --end 2023051700
#
# Jul-2026, Pat Welch, pat@mousebrains.com

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tarfile
import time
from pathlib import Path

INDEX_VERSION = 1
COPY_BUFSIZE = 4 * 1024 * 1024


class _HashReader:
    """File-object wrapper that sha256-hashes everything read through it."""

    def __init__(self, fobj):
        self._f = fobj
        self.digest = hashlib.sha256()

    def read(self, n: int = -1) -> bytes:
        buf = self._f.read(n)
        self.digest.update(buf)
        return buf


def find_hours(src: Path, start: str, end: str) -> list[Path]:
    """Hour directories YYYY/MM/DD/HH under *src* with start <= key <= end."""
    hours = []
    for p in sorted(src.glob("[0-9]" * 4 + "/[0-9][0-9]/[0-9][0-9]/[0-9][0-9]")):
        if not p.is_dir():
            continue
        key = "".join(p.parts[-4:])  # YYYYMMDDHH
        if start <= key <= end:
            hours.append(p)
    return hours


def chunk_hour(hour_dir: Path, dst_root: Path, src_root: Path, checksum: bool) -> dict:
    """Bundle one hour directory; returns a stats dict. Raises on failure."""
    y, m, d, h = hour_dir.parts[-4:]
    out_dir = dst_root / y / m / d
    tar_path = out_dir / f"{y}{m}{d}{h}.pol.tar"
    idx_path = tar_path.with_suffix(".tar.index.json")

    files = sorted(f for f in hour_dir.iterdir() if f.is_file() and ".pol" in f.name)
    if not files:
        return {"skipped": "empty"}
    if tar_path.exists() and idx_path.exists():
        return {"skipped": "done"}

    out_dir.mkdir(parents=True, exist_ok=True)
    part = tar_path.with_name(tar_path.name + ".partial")
    hashes: dict[str, str] = {}
    t0 = time.perf_counter()
    n_bytes = 0
    try:
        with tarfile.open(part, "w", format=tarfile.USTAR_FORMAT, bufsize=COPY_BUFSIZE) as tf:
            for f in files:
                st = f.stat()
                ti = tarfile.TarInfo(f.name)
                ti.size = st.st_size
                ti.mtime = int(st.st_mtime)
                ti.mode = 0o644
                with open(f, "rb") as fobj:
                    if checksum:
                        hr = _HashReader(fobj)
                        tf.addfile(ti, hr)
                        hashes[f.name] = hr.digest.hexdigest()
                    else:
                        tf.addfile(ti, fobj)
                n_bytes += st.st_size

        # authoritative index + verification: re-scan the tar we wrote
        members = []
        with tarfile.open(part, "r:") as tf:
            for mem in tf:
                entry = {
                    "name": mem.name,
                    "offset": mem.offset_data,
                    "size": mem.size,
                    "mtime": mem.mtime,
                }
                if checksum:
                    entry["sha256"] = hashes[mem.name]
                members.append(entry)
        if len(members) != len(files):
            raise RuntimeError(f"verify failed: {len(members)} members != {len(files)} files")
        by_name = {e["name"]: e["size"] for e in members}
        for f in files:
            if by_name.get(f.name) != f.stat().st_size:
                raise RuntimeError(f"verify failed: size mismatch for {f.name}")

        idx = {
            "version": INDEX_VERSION,
            "source": str(hour_dir.relative_to(src_root)),
            "n_files": len(members),
            "bytes": n_bytes,
            "files": members,
        }
        idx_tmp = idx_path.with_name(idx_path.name + ".partial")
        idx_tmp.write_text(json.dumps(idx))
        part.replace(tar_path)  # tar first, then index: skip-check
        idx_tmp.replace(idx_path)  # requires BOTH, so a crash here redoes
    except BaseException:
        part.unlink(missing_ok=True)
        raise
    dt = time.perf_counter() - t0
    return {"n": len(files), "bytes": n_bytes, "secs": dt}


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Bundle hourly .pol directories into indexed tar chunks"
    )
    ap.add_argument("src", type=Path, help="POLAR tree root (YYYY/MM/DD/HH)")
    ap.add_argument("dst", type=Path, help="output tree root (created)")
    ap.add_argument("--start", default="0000000000", help="first hour, YYYYMMDDHH (default: all)")
    ap.add_argument(
        "--end", default="9999999999", help="last hour, YYYYMMDDHH inclusive (default: all)"
    )
    ap.add_argument(
        "--checksum", action="store_true", help="store per-member sha256 in the index (slower)"
    )
    ap.add_argument("--dry-run", action="store_true", help="list hours and sizes, write nothing")
    args = ap.parse_args()

    hours = find_hours(args.src.resolve(), args.start, args.end)
    if not hours:
        print(f"no hour directories in {args.src} for [{args.start}, {args.end}]", flush=True)
        return 1

    tot_bytes = tot_files = tot_secs = 0.0
    n_done = n_skip = n_fail = 0
    for hd in hours:
        tag = "/".join(hd.parts[-4:])
        if args.dry_run:
            files = [f for f in hd.iterdir() if f.is_file() and ".pol" in f.name]
            gb = sum(f.stat().st_size for f in files) / 1e9
            print(f"{tag}: {len(files)} files, {gb:.2f} GB", flush=True)
            continue
        try:
            r = chunk_hour(hd, args.dst.resolve(), args.src.resolve(), args.checksum)
        except KeyboardInterrupt:
            print(f"{tag}: interrupted (partial removed); rerun to resume", flush=True)
            return 130
        except Exception as e:  # noqa: BLE001 keep the batch going
            print(f"{tag}: FAILED {e}", flush=True)
            n_fail += 1
            continue
        if "skipped" in r:
            n_skip += 1
            print(f"{tag}: skipped ({r['skipped']})", flush=True)
        else:
            n_done += 1
            tot_bytes += r["bytes"]
            tot_files += r["n"]
            tot_secs += r["secs"]
            print(
                f"{tag}: {r['n']} files, {r['bytes'] / 1e9:.2f} GB in "
                f"{r['secs']:.1f} s ({r['bytes'] / r['secs'] / 1e6:.0f} MB/s)",
                flush=True,
            )
    if not args.dry_run:
        rate = tot_bytes / tot_secs / 1e6 if tot_secs else 0.0
        print(
            f"TOTAL: {n_done} hours ({int(tot_files)} files, "
            f"{tot_bytes / 1e9:.2f} GB) at {rate:.0f} MB/s aggregate; "
            f"{n_skip} skipped, {n_fail} failed",
            flush=True,
        )
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
