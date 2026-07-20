#!/bin/zsh
VENV=/Users/pat/.local/pipx/venvs/wamos-tpw/bin/python
cd /Users/pat/Desktop/WAMOS/thompson2023
run_stage() { local label=$1; shift
  for attempt in 1 2; do
    "$@" && { echo "CHAIN: $label done"; return 0; }
    echo "CHAIN: $label attempt $attempt FAILED rc=$?"; sleep 60
  done
  echo "CHAIN: $label GAVE UP"; return 1
}
rm -rf scene_data/movie/frames_ship
run_stage anchored $VENV movie_driver_ship.py 2023-05-13T22:00 2023-05-18T00:00 30 TGT_anchored.mp4 --fixed || exit 1
rm -rf scene_data/movie/frames_fold
run_stage fold $VENV movie_driver_fold.py 2023-05-13T22:00 2023-05-18T00:00 15 TGT_fold.mp4 --fixed || exit 1
rm -rf scene_data/movie/frames
run_stage eddy $VENV movie_driver.py 2023-05-18T00:00 2023-05-23T00:00 60 EDDY_joint.mp4 || exit 1
echo "REELS CHAIN COMPLETE"
