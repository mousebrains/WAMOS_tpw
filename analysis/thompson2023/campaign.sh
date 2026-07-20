#!/bin/zsh
VENV=/Users/pat/.local/pipx/venvs/wamos-tpw/bin/python
cd /Users/pat/Desktop/WAMOS/thompson2023
while ! grep -qE "MOVIE DONE|no movie" movie_demo.log 2>/dev/null; do sleep 120; done
# stage 0: re-render demo with drifter arrows (render-only, cached inputs)
rm -rf scene_data/movie/frames
$VENV movie_driver.py 2023-05-19T14:00 2023-05-19T17:00 7.5 DEMO_smooth.mp4 \
  && echo "CAMPAIGN: demo done"
run_stage() { local label=$1; shift
  for attempt in 1 2; do
    "$@" && { echo "CAMPAIGN: $label done"; return 0; }
    echo "CAMPAIGN: $label attempt $attempt FAILED rc=$?"; sleep 120
  done
  echo "CAMPAIGN: $label GAVE UP"; return 1
}
rm -rf scene_data/movie/frames
run_stage joint75 $VENV movie_driver.py 2023-05-18T05:00 2023-05-23T00:00 7.5 EDDY_joint75.mp4 || exit 1
rm -rf scene_data/movie/frames_shore
run_stage angaur75 $VENV movie_driver_shore.py 2023-05-18T05:00 2023-05-30T05:00 7.5 ANGAUR.mp4 || exit 1
rm -rf scene_data/movie/frames_fold
run_stage tgt75 $VENV movie_driver_fold.py 2023-05-13T22:00 2023-05-23T01:00 7.5 TGT_fold75.mp4 --fixed || exit 1
echo "CAMPAIGN COMPLETE"
