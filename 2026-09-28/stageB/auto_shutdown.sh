#!/bin/bash
# Server-side safety net: shut the instance down when today's jobs are finished (+45 min grace for local sync),
# or unconditionally after a hard cap of 4 hours from launch. Data disk (/root/autodl-tmp) is kept by AutoDL.
cd /root/autodl-tmp/E01
log() { echo "$(date '+%F %T') [auto_shutdown] $*" >> logs/chain.log; }
START=$(date +%s); DEADLINE=${DEADLINE:-$((START+4*3600))}
finished() {
  { [ -f runs/CHAIN_DONE ] || [ -f runs/CHAIN_FAILED ]; } && { [ -f runs/PRE_DONE ] || [ -f runs/PRE_FAILED ]; } \
  && [ -f runs/CENSUSB_DONE ] && ! ls runs/BUSY_* >/dev/null 2>&1
}
log "armed (deadline $(date -d @$DEADLINE "+%F %T"))"
while ! finished; do
  [ $(date +%s) -gt $DEADLINE ] && { log "hard cap reached, shutting down"; sync; shutdown -h now; exit; }
  sleep 60
done
log "all jobs finished; grace 45 min for local sync"
for i in $(seq 45); do [ -f runs/SYNCED ] && break; sleep 60; done
log "shutting down"; sync; shutdown -h now
