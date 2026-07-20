#!/bin/zsh
# Fine-step (0.02 s) hard-returns time_shift refinement, 3 turn windows
VENV=/Users/pat/.local/pipx/venvs/wamos-tpw/bin/python
SRC=/Users/pat/Desktop/WAMOS/wamos_tpw/src
POLAR=/Volumes/SeaChest/ARCTERX/2023/WAMOS/POLAR
SHIP=/Users/pat/Desktop/WAMOS/ship_tn417
run_one() {
  echo "=== WINDOW $1 - $2 ==="
  PYTHONPATH=$SRC $VENV -m wamos_tpw.cli hard-returns "$1" "$2" "$POLAR" \
    --ship-data "$SHIP" --sweep-range 1.4 --sweep-step 0.02 \
    --no-plot --no-progress 2>&1
}
run_one 20230502T183300 20230502T184300
run_one 20230502T223800 20230502T224700
run_one 20230503T121800 20230503T122700
echo "FINESTEP DONE"
