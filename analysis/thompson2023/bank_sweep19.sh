#!/bin/zsh
VENV=/Users/pat/.local/pipx/venvs/wamos-tpw/bin/python
SRC=/Users/pat/Desktop/WAMOS/wamos_tpw/src
cd /Users/pat/Desktop/WAMOS/thompson2023
PYTHONPATH=$SRC $VENV -m wamos_tpw.cli current \
  20230519T1400 20230519T1600 \
  /Volumes/SeaChest/ARCTERX/2023/WAMOS/POLAR \
  --config tn2023_wamos.yaml \
  --ship-data /Users/pat/Desktop/WAMOS/ship_tn417 \
  --min-snr 1.1 --max-tile-range 3000 \
  --land-mask palau_landmask.nc \
  --depth-grid "/Volumes/SeaChest/ARCTERX/2023/Wake/bathy/Angaur_Peleliu_25m.nc" \
  --depth-adjust 1.2 \
  --field --window-sizes 2000,1000 \
  --composite-minutes 15 \
  --output-dir bank_sweep_20230519 --format netcdf --no-progress
echo "BANK SWEEP DONE rc=$?"
