#!/bin/zsh
cd /Users/pat/Desktop/WAMOS/thompson2023
while ! grep -q "REELS CHAIN COMPLETE" reels_chain.log 2>/dev/null; do
  kill -0 "$(cat reels_chain.pid)" 2>/dev/null || grep -q "REELS CHAIN COMPLETE" reels_chain.log || { echo "chain dead, running shore anyway"; break; }
  sleep 300
done
/Users/pat/.local/pipx/venvs/wamos-tpw/bin/python movie_driver_shore.py \
  2023-05-18T05:00 2023-05-30T05:00 60 ANGAUR.mp4
echo "SHORE REEL DONE rc=$?"
