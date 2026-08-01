#!/bin/bash
set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="$PROJECT_ROOT/.run_logs"
mkdir -p "$LOG_DIR"

# EVERYTHING runs under the `sharktank` conda env — engine and backend alike.
# A bare `python3` resolves to base anaconda when launched from Finder, and the
# two envs are not interchangeable here:
#   * base has numpy 2.x and no ultralytics -> fight detection silently off
#   * the backend writes Database/face_embeddings.pkl during registration, so
#     if it runs under numpy 2 while the engine runs under numpy 1, the engine
#     can no longer load that pickle (ModuleNotFoundError numpy._core.numeric)
# Keeping both on one interpreter is what prevents that split.
SENTRA_PYTHON="/opt/anaconda3/envs/sharktank/bin/python3"
if [ ! -x "$SENTRA_PYTHON" ]; then
    echo "WARNING: $SENTRA_PYTHON not found — falling back to PATH python3."
    echo "         Fight detection and face registration may not work correctly."
    SENTRA_PYTHON="python3"
fi

# There is exactly ONE app now: backend_v2 (the current UI), served on 8000.
# The old backend/ UI is retired — it is kept on disk as a reference only and
# is deliberately never started. 8001 was the old side-by-side comparison port.
APP_DIR="$PROJECT_ROOT/backend_v2"
APP_PORT=8000
LEGACY_PORT=8001

echo "=== Sentra Launcher ==="
echo

# A previous version of this launcher started two backends (8000 + 8001). If any
# of those are still alive from an earlier session, they are stale — clear them
# out so only one app is ever running and the user never sees two windows again.
for stale_port in "$APP_PORT" "$LEGACY_PORT"; do
    stale_pids="$(lsof -ti:"$stale_port" 2>/dev/null)"
    if [ -n "$stale_pids" ]; then
        echo "[Cleanup] Stopping existing server on port $stale_port..."
        kill $stale_pids 2>/dev/null
    fi
done
# Also catch an old backend/ uvicorn that somehow isn't holding a listed port.
# The bracket keeps the pattern from matching pkill's own command line — without
# it pkill kills itself and the shell prints a "Terminated" line at every launch.
pkill -f "[u]vicorn main:app" >/dev/null 2>&1
sleep 1
for stale_port in "$APP_PORT" "$LEGACY_PORT"; do
    stale_pids="$(lsof -ti:"$stale_port" 2>/dev/null)"
    if [ -n "$stale_pids" ]; then
        kill -9 $stale_pids 2>/dev/null
    fi
done

echo "[Sentra] Starting on port $APP_PORT..."
(cd "$APP_DIR" && nohup "$SENTRA_PYTHON" -m uvicorn main:app --host 0.0.0.0 --port "$APP_PORT" > "$LOG_DIR/backend.log" 2>&1 &)
for i in $(seq 1 20); do
    if curl -s -o /dev/null "http://127.0.0.1:$APP_PORT/login"; then
        echo "[Sentra] Up."
        break
    fi
    sleep 0.5
done

# Sessions are in-memory (main.py), so every restart invalidates old cookies —
# a stale browser tab then sees 401s and e.g. the Alerts tab hangs on
# "Loading...". Log in fresh here and hit a real API route so a broken backend
# is caught now, not discovered later as a stuck UI.
COOKIE_JAR="/tmp/sentra_check_$APP_PORT.txt"
curl -s -c "$COOKIE_JAR" -X POST "http://127.0.0.1:$APP_PORT/api/login" \
    -H 'Content-Type: application/json' \
    -d '{"username":"sharktanktest","password":"demo"}' >/dev/null
# Captured into a variable rather than piped straight into grep -q: with
# pipefail on, grep -q's early exit on match sends curl a SIGPIPE, and pipefail
# then reports the whole pipeline as failed even though it matched.
ANOMALIES_RESP="$(curl -s -b "$COOKIE_JAR" "http://127.0.0.1:$APP_PORT/api/anomalies")"
if [[ "$ANOMALIES_RESP" == "[]" || "$ANOMALIES_RESP" == "[{"* ]]; then
    echo "[Sentra] API check passed (login + /api/anomalies)."
else
    echo "[Sentra] WARNING: API check failed — see $LOG_DIR/backend.log"
fi
rm -f "$COOKIE_JAR"

# --- AI engine (camera) ---
ENGINE_LOG="$LOG_DIR/face_recognition.log"
# The engine gets started two different ways and the "already running" check
# must catch both, otherwise this starts a SECOND engine and the two fight over
# the same RTSP stream:
#   * this launcher / the dashboard restart button -> "python3 face_recognition.py"
#   * sentra_app.py, which is what the packaged build uses -> "sentra_app.py --engine"
ENGINE_PATTERN="face_recognition\.py|sentra_app\.py --engine"

if pgrep -f "$ENGINE_PATTERN" >/dev/null 2>&1; then
    echo "[Camera engine] Already running — skipping."
else
    echo "[Camera engine] Starting..."
    # -u (unbuffered) is required, not cosmetic: with stdout redirected to a
    # file Python buffers it, so a HEALTHY engine writes nothing for a long
    # time. Everything below that greps this log — and anyone tailing it —
    # would see an empty file and conclude the engine is broken. Ironically the
    # log only ever filled up before because a crash flushes the buffer.
    (cd "$PROJECT_ROOT/Formal_Code" && nohup "$SENTRA_PYTHON" -u face_recognition.py > "$ENGINE_LOG" 2>&1 &)
    sleep 3
    if pgrep -f "$ENGINE_PATTERN" >/dev/null 2>&1; then
        echo "[Camera engine] Started. (Whether it actually reaches the camera shows up"
        echo "                 on the Live Monitor tab — this only confirms the process is up.)"
    else
        echo "[Camera engine] FAILED to start. Check: $ENGINE_LOG"
    fi

    # Fight detection degrades gracefully by design (face recognition keeps
    # working without it), which means a failure here is otherwise invisible.
    # Surface it at launch instead of at the moment someone needs an alert.
    # The RTSP connect attempt can take ~30s to time out, so wait past that
    # before judging — otherwise a dead camera just reads as "unknown".
    for i in $(seq 1 20); do
        if grep -qE "Pose model loaded|Fight detection disabled|Could not open the RTSP" "$ENGINE_LOG" 2>/dev/null; then
            break
        fi
        sleep 2
    done

    # Reported independently of the camera: the pose model is loaded before the
    # camera is opened, so its status is known even when the camera is down.
    # Chaining these would let a dead camera mask whether alerting works.
    if grep -q "Fight detection disabled" "$ENGINE_LOG" 2>/dev/null; then
        echo "[Camera engine] WARNING: fight detection is DISABLED —"
        grep -m1 "Fight detection disabled" "$ENGINE_LOG" | sed 's/^/                 /'
    elif grep -q "Pose model loaded" "$ENGINE_LOG" 2>/dev/null; then
        echo "[Camera engine] Fight detection active."
    else
        echo "[Camera engine] Fight detection status unknown yet — check: $ENGINE_LOG"
    fi

    if grep -q "Could not open the RTSP" "$ENGINE_LOG" 2>/dev/null; then
        echo "[Camera engine] ERROR: could not reach the camera — engine exited."
        echo "                 Check the camera is powered on and this Mac is on the"
        echo "                 same network as it. Configured URL:"
        grep -m1 rtsp_url "$PROJECT_ROOT/Database/camera_config.json" | sed 's/^/                 /'
    fi
fi

echo
echo "Opening Sentra..."
sleep 1
open "http://localhost:$APP_PORT"

echo
echo "Done — log in with sharktanktest / demo."
echo "  http://localhost:$APP_PORT"
echo "You can close this window; everything keeps running in the background."
echo "Logs: $LOG_DIR/"
echo
read -n 1 -s -r -p "Press any key to close this window..."
echo
