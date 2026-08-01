#!/bin/bash
set -uo pipefail

echo "=== Sentra Stopper ==="
echo

# 8000 is the app; 8001 is the retired side-by-side port, still cleaned up here
# so a stale server from an older launcher can't linger.
for port in 8000 8001; do
    pids="$(lsof -ti:"$port" 2>/dev/null)"
    if [ -n "$pids" ]; then
        echo "[Sentra] Stopping server on port $port..."
        kill $pids 2>/dev/null
    else
        echo "[Sentra] Nothing running on port $port."
    fi
done

# The engine gets started two different ways and both must be caught here:
#   * this launcher / the dashboard restart button -> "python3 face_recognition.py"
#   * sentra_app.py, which is what the packaged build uses -> "sentra_app.py --engine"
# Matching only the first left an orphaned engine still holding the RTSP
# stream while this script cheerfully reported "Not running".
ENGINE_PATTERN="face_recognition\.py|sentra_app\.py --engine"

if pgrep -f "$ENGINE_PATTERN" >/dev/null 2>&1; then
    echo "[Camera engine] Stopping..."
    pkill -f "$ENGINE_PATTERN"
else
    echo "[Camera engine] Not running."
fi
# Stale pid file would otherwise make the restart endpoint try to kill a pid
# that no longer belongs to us.
rm -f "$(cd "$(dirname "$0")" && pwd)/.run_logs/engine.pid"

sleep 1

# Anything still holding those ports after a plain kill gets a harder nudge.
for port in 8000 8001; do
    pids="$(lsof -ti:"$port" 2>/dev/null)"
    if [ -n "$pids" ]; then
        kill -9 $pids 2>/dev/null
    fi
done

echo
echo "Done."
read -n 1 -s -r -p "Press any key to close this window..."
echo
