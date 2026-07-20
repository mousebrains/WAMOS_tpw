#!/bin/zsh
# Resume the movie campaign after the 2026-07-19 SMB service bounce.
# joint75 restarts at scene time 2023-05-22T05:45 with frame counter
# seeded at 763 (frames 0000-0762 already rendered, inputs cached).
VENV=/Users/pat/.local/pipx/venvs/wamos-tpw/bin/python
cd /Users/pat/Desktop/WAMOS/thompson2023
run_stage() { local label=$1; shift
  for attempt in 1 2; do
    "$@" && { echo "CAMPAIGN: $label done"; return 0; }
    echo "CAMPAIGN: $label attempt $attempt FAILED rc=$?"; sleep 120
  done
  echo "CAMPAIGN: $label GAVE UP"; return 1
}
run_stage joint75 $VENV movie_driver.py 2023-05-22T05:45 2023-05-23T00:00 7.5 EDDY_joint75.mp4 763 || exit 1
rm -rf scene_data/movie/frames_shore
run_stage angaur75 $VENV movie_driver_shore.py 2023-05-18T05:00 2023-05-30T05:00 7.5 ANGAUR.mp4 || exit 1
rm -rf scene_data/movie/frames_fold
run_stage tgt75 $VENV movie_driver_fold.py 2023-05-13T22:00 2023-05-23T01:00 7.5 TGT_fold75.mp4 --fixed || exit 1
echo "CAMPAIGN COMPLETE"
