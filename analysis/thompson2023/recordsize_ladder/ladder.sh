#!/bin/zsh
# Recordsize/bundling/protocol disentangling ladder (2026-07-19 plan):
#   cp1: POLAR (128K records) -> POLAR_1M      = cold 128K server read
#   cp2: POLAR_1M            -> POLAR_1M_B     = cold 1M   server read
#   Mac bench: 128K loose vs 1M loose vs tar (ranged + stream) over SMB
# ARC (~100 GB) is flushed with ~160 GB of unrelated reads between steps.
# Run detached: nohup caffeinate -i ./ladder.sh > ladder.log 2>&1 &
NAS=pat@nas0.local
W=/mnt/HDD/SeaChest/ARCTERX/2023/WAMOS
HERE=${0:a:h}
log() { echo "$(date +%H:%M:%S) $*" }

evict() { log "eviction pass..."; ssh $NAS 'bash -s' < $HERE/nas_evict.sh; }

server_cp() { # $1 src-tree $2 dst-tree ; copies hours 12-14 of 2023/05/15
    ssh $NAS "mkdir -p $W/$2/2023/05/15
        for h in 12 13 14; do
            t0=\$SECONDS
            cp -r --reflink=never $W/$1/2023/05/15/\$h $W/$2/2023/05/15/
            dt=\$((SECONDS - t0))
            b=\$(du -sb $W/$2/2023/05/15/\$h | cut -f1)
            echo \"cp $1->$2 hour \$h: \$((b/1000000)) MB in \${dt}s (\$((b/1000000/(dt+1))) MB/s)\"
        done"
}

log "=== LADDER START ==="
evict
log "--- cp1: 128K -> 1M ---"
server_cp POLAR POLAR_1M
evict
log "--- cp2: 1M -> 1M ---"
server_cp POLAR_1M POLAR_1M_B
evict
log "--- Mac three-way bench over SMB ---"
/Users/pat/.local/pipx/venvs/wamos-tpw/bin/python $HERE/mac_bench.py
log "=== LADDER DONE ==="
