#!/bin/zsh
# Full ship x mooring overlap: May 13 22:00 -> May 23 01:00, per-day chunks.
# Each day writes to bank_sweep_may/dayDD/ and drops a .done marker;
# re-running skips completed days.
VENV=/Users/pat/.local/pipx/venvs/wamos-tpw/bin/python
SRC=/Users/pat/Desktop/WAMOS/wamos_tpw/src
BASE=/Users/pat/Desktop/WAMOS/thompson2023
run_day() {  # $1 = day, $2 = stime, $3 = etime
  OUT=$BASE/bank_sweep_may/day$1
  if [[ -e $OUT/.done ]]; then echo "DAY $1 already done, skip"; return; fi
  mkdir -p "$OUT"
  echo "=== DAY $1: $2 -> $3  ($(date -u '+%H:%M:%S')) ==="
  PYTHONPATH=$SRC $VENV -m wamos_tpw.cli current "$2" "$3" \
    /Volumes/SeaChest/ARCTERX/2023/WAMOS/POLAR \
    --config $BASE/tn2023_wamos.yaml \
    --ship-data /Users/pat/Desktop/WAMOS/ship_tn417 \
    --min-snr 1.1 --max-tile-range 3000 \
    --land-mask $BASE/palau_landmask.nc \
    --depth-grid "/Volumes/SeaChest/ARCTERX/2023/Wake/bathy/Angaur_Peleliu_25m.nc" \
    --depth-adjust 1.2 \
    --field --window-sizes 2000,1000 \
    --composite-minutes 15 \
    --output-dir "$OUT" --format netcdf --no-progress
  rc=$?
  if [[ $rc -eq 0 ]]; then touch "$OUT/.done"; echo "DAY $1 DONE rc=0"
  else echo "DAY $1 FAILED rc=$rc"; fi
}
run_day 13 20230513T2200 20230514T0000
for d in 14 15 16 17 18 19 20 21 22; do
  run_day $d 202305${d}T0000 202305$((d+1))T0000
done
run_day 23 20230523T0000 20230523T0200
echo "ALL DAYS COMPLETE $(date -u '+%Y-%m-%d %H:%M:%S')"
