"""FastAPI backend for the Sentra dashboard.

Serves real data only:
- Detections come from the real Database/detections.db (written by
  Formal_Code/event_logger.py, called from face_recognition.py).
- Registered people come from the real Database/face_embeddings.pkl.
- The live feed is relayed from Formal_Code/face_recognition.py over a
  WebSocket (/ws/engine) to dashboard viewers (/ws/live). If the AI engine
  script isn't running, the dashboard says so — it never fabricates a feed.
- There is exactly one camera. Its name and RTSP URL live in
  Database/camera_config.json (Formal_Code/camera_config.py), editable from
  the Cameras tab. No other cameras, people, or locations are invented
  anywhere in this backend.
- Every page and API route requires a logged-in session (demo credentials
  only — see DEMO_USERNAME/DEMO_PASSWORD below).
"""

from __future__ import annotations

import secrets
import subprocess
import sys
import time
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from pydantic import BaseModel

import database
import people_store

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "Formal_Code"))
import camera_config  # noqa: E402  (path must be set up first)

app = FastAPI(title="Sentra Backend")

# --- Auth (demo-grade: fixed credentials, in-memory sessions) --------------
# This gates a local pitch demo, not a production deployment — see
# known_issues memory for what a real login would still need (password
# hashing, persistent sessions, etc.).

DEMO_USERNAME = "sharktanktest"
DEMO_PASSWORD = "demo"
SESSION_COOKIE = "sentra_session"

active_sessions: set[str] = set()


def is_authenticated(request: Request) -> bool:
    token = request.cookies.get(SESSION_COOKIE)
    return token is not None and token in active_sessions


@app.middleware("http")
async def require_login(request: Request, call_next):
    path = request.url.path
    # Only /api/* routes are gated here; "/" enforces its own redirect
    # below so browsers get a proper redirect instead of raw JSON.
    if path.startswith("/api/") and path != "/api/login" and not is_authenticated(request):
        return JSONResponse({"detail": "Not authenticated"}, status_code=401)
    return await call_next(request)


class LoginRequest(BaseModel):
    username: str
    password: str


@app.post("/api/login")
def login(payload: LoginRequest):
    if payload.username != DEMO_USERNAME or payload.password != DEMO_PASSWORD:
        raise HTTPException(status_code=401, detail="Invalid username or password.")
    token = secrets.token_urlsafe(24)
    active_sessions.add(token)
    resp = JSONResponse({"ok": True})
    resp.set_cookie(SESSION_COOKIE, token, httponly=True, samesite="lax")
    return resp


@app.get("/login")
def serve_login():
    return FileResponse(Path(__file__).parent / "login.html")


# --- Engine / live-viewer relay state -------------------------------------

engine_connected = False
live_viewers: set[WebSocket] = set()


@app.websocket("/ws/engine")
async def ws_engine(websocket: WebSocket) -> None:
    """face_recognition.py connects here as the single frame producer.

    This is a trusted local process, not a browser — it does not go through
    the login session (the dashboard viewers on /ws/live do).
    """
    global engine_connected
    await websocket.accept()
    engine_connected = True
    try:
        while True:
            message = await websocket.receive_text()
            dead = []
            for viewer in live_viewers:
                try:
                    await viewer.send_text(message)
                except Exception:
                    dead.append(viewer)
            for viewer in dead:
                live_viewers.discard(viewer)
    except WebSocketDisconnect:
        pass
    finally:
        engine_connected = False


@app.websocket("/ws/live")
async def ws_live(websocket: WebSocket) -> None:
    """Dashboard clients connect here to watch the relayed live feed."""
    token = websocket.cookies.get(SESSION_COOKIE)
    if token not in active_sessions:
        await websocket.close(code=4401)
        return
    await websocket.accept()
    live_viewers.add(websocket)
    try:
        while True:
            await websocket.receive_text()  # viewers don't send anything meaningful
    except WebSocketDisconnect:
        pass
    finally:
        live_viewers.discard(websocket)


# --- Stats / detections -----------------------------------------------------

@app.get("/api/stats")
def get_stats():
    return {
        "cameras_online": 1 if engine_connected else 0,
        "cameras_total": 1,
        "registered_individuals": len(people_store.list_names()),
        "detections_today": database.detections_today_count(),
        "alerts_today": database.anomalies_today_count(),
    }


@app.get("/api/detections")
def get_detections(limit: int = 10):
    limit = max(1, min(limit, 200))
    return database.recent_detections(limit)


# --- Anomalies (fight detection) -------------------------------------------
# Rows are written by the AI engine itself (Formal_Code/anomaly_detection.py
# via event_logger.log_anomaly), the same path detections take. The backend
# only reads them here — the /ws/engine relay stays a dumb, fast passthrough.


@app.get("/api/anomalies")
def get_anomalies(limit: int = 20, type: str | None = None):
    limit = max(1, min(limit, 200))
    return database.recent_anomalies(limit, anomaly_type=type)


# --- People -------------------------------------------------------------

@app.get("/api/people")
def get_people():
    return {"names": people_store.list_names()}


@app.get("/api/people/search")
def search_people(name: str = ""):
    query = name.strip().lower()
    if not query:
        return {"match": None}

    names = people_store.list_names()
    match = next((n for n in names if query in n.lower()), None)
    if not match:
        return {"match": None}

    history = database.detections_for_person(match, limit=50)
    return {
        "match": match,
        "last_seen": history[0] if history else None,
        "history": history,
    }


class DeleteResult(BaseModel):
    deleted: bool


@app.post("/api/people/register")
async def register_person(name: str = Form(...), images: list[UploadFile] = File(...)):
    try:
        image_bytes = [await f.read() for f in images]
        result = people_store.register_person(name, image_bytes)
        return result
    except (ValueError, people_store.InvalidNameError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.delete("/api/people/{name}", response_model=DeleteResult)
def remove_person(name: str):
    deleted = people_store.delete_person(name)
    return {"deleted": deleted}


# --- Cameras --------------------------------------------------------------

class CameraUpdate(BaseModel):
    name: str
    rtsp_url: str


@app.get("/api/cameras")
def get_cameras():
    cfg = camera_config.load_camera_config()
    return [
        {
            "name": cfg["name"],
            "rtsp_url": cfg["rtsp_url"],
            "online": engine_connected,
        }
    ]


@app.put("/api/cameras")
def update_camera(update: CameraUpdate):
    try:
        cfg = camera_config.save_camera_config(update.name, update.rtsp_url)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"name": cfg["name"], "rtsp_url": cfg["rtsp_url"], "online": engine_connected}


# --- Engine control ---------------------------------------------------------
# Restarts the one shared Formal_Code/face_recognition.py process: reloads the
# InsightFace model, reconnects the RTSP camera, and re-establishes the
# WebSocket stream to every backend listed in its DASHBOARD_WS_URLS (both
# this one and any sibling UI on another port). There is only ever one engine
# process regardless of which dashboard's button triggers the restart.

FORMAL_CODE_DIR = PROJECT_ROOT / "Formal_Code"
RUN_LOGS_DIR = PROJECT_ROOT / ".run_logs"

# The engine must run under the `sharktank` conda env, not whatever `python3`
# happens to resolve to. This backend itself runs under base anaconda, so
# sys.executable is the wrong interpreter here: base has numpy 2.x and no
# ultralytics, which silently disables fight detection while leaving face
# recognition working — a failure that is very easy to miss.
ENGINE_PYTHON = "/opt/anaconda3/envs/sharktank/bin/python3"
if not Path(ENGINE_PYTHON).is_file():
    ENGINE_PYTHON = "python3"


@app.post("/api/engine/restart")
def restart_engine():
    RUN_LOGS_DIR.mkdir(parents=True, exist_ok=True)
    # Keep the interpreter in the pattern: matching the bare filename also
    # matches any shell whose command line merely mentions it (a tail, a grep),
    # which would kill the wrong thing. An absolute interpreter path still
    # contains "python3 face_recognition.py" as a substring, so this matches.
    subprocess.run(["pkill", "-f", r"python3 face_recognition\.py"], check=False)
    time.sleep(1)
    log_path = RUN_LOGS_DIR / "face_recognition.log"
    with open(log_path, "a") as logf:
        subprocess.Popen(
            [ENGINE_PYTHON, "face_recognition.py"],
            cwd=str(FORMAL_CODE_DIR),
            stdout=logf,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    return {"restarted": True}


# --- Static dashboard -------------------------------------------------------

DASHBOARD_PATH = Path(__file__).parent / "dashboard.html"


@app.get("/")
def serve_dashboard(request: Request):
    if not is_authenticated(request):
        return RedirectResponse(url="/login")
    return FileResponse(DASHBOARD_PATH)
