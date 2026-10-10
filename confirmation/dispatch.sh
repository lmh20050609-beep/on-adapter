#!/usr/bin/env bash
# Train primary (cv) then control (ta) on seeds 2025-2027 (GPU, serial); test trajectories as cells finish (CPU); score.
set -u
B=/mnt/c/Users/18452/Documents/ChatGPT/FedNoise/iadapter-baselines; D=$B/results/confirmation; PY=./.venv/Scripts/python.exe; W=8
cd "$B" || exit 1; mkdir -p "$D/running" "$D/logs"; echo $$ > "$D/dispatcher.pid"; log() { echo "$(date '+%F %T') $*" >> "$D/dispatcher.log"; }
( for t in cv ta; do mkdir -p "$D/campaign_$t"; [ -f "$D/campaign_$t/control.json" ] || echo '{"pause": false}' > "$D/campaign_$t/control.json"
    $PY -m confirmation.runner run --config configs/confirmation_${t}.json < /dev/null >> "$D/logs/train_$t.log" 2>&1; echo "train $t exit=$?" >> "$D/dispatcher.log"; touch "$D/trained_$t"
    if [ "$t" = cv ]; then touch "$D/cv_trained"; fi; done ) &
log start; scored_cv=0
while true; do
  for f in "$D"/running/*.pid; do [ -e "$f" ] || continue; kill -0 "$(cat "$f")" 2>/dev/null || rm -f "$f"; done
  { $PY confirmation/confirm.py --list cv; $PY confirmation/confirm.py --list ta; } < /dev/null 2>>"$D/dispatcher.log" | tr -d '\r' > "$D/todo.txt"
  idle=0; [ ! -s "$D/todo.txt" ] && [ "$(ls "$D"/running/*.pid 2>/dev/null | wc -l)" -eq 0 ] && idle=1
  if [ $idle -eq 1 ] && [ -f "$D/trained_cv" ] && [ $scored_cv -eq 0 ]; then $PY confirmation/confirm.py --score < /dev/null > "$D/score_primary.log" 2>&1; log "primary score exit=$?"; scored_cv=1; fi
  if [ $idle -eq 1 ] && [ -f "$D/trained_ta" ]; then $PY confirmation/confirm.py --score < /dev/null > "$D/score_all.log" 2>&1; log "final score exit=$?"; break; fi
  while read -r tg d h s; do n="${tg}_${d}_${h}_${s}"; [ -e "$D/running/$n.pid" ] && continue; [ "$(ls "$D"/running/*.pid 2>/dev/null | wc -l)" -ge $W ] && break
    ( $PY confirmation/confirm.py --traj $tg $d $h $s < /dev/null >> "$D/logs/$n.log" 2>&1; echo "exit=$?" >> "$D/logs/$n.log" ) &
    echo $! > "$D/running/$n.pid"; log "started $n"; done < "$D/todo.txt"
  sleep 30
done
rm -f "$D/dispatcher.pid"
