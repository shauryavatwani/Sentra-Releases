"""FastAPI backend for the Sentra dashboard.

Serves real data only:
- Detections come from the real Database/detections.db (written by
  Formal_Code/event_logger.py, called from face_recognition.py).
- Registered people come from the real Database/face_embeddings.pkl.
- The live feed is relayed from Formal_Code/face_recognition.py over a
  WebSocket (/ws/engine) to dashboard viewers (/ws/live). If the AI engine
  script isn't running, the dashboard says so — it never fabricates a feed.
- Any number of cameras can be configured (Formal_Code/camera_store.py,
  Database/camera_config.json), managed from the Cameras tab. Each camera
  always streams to Live Monitor; face recognition and fight detection
  additionally run only on cameras with ai_enabled set, since each one costs
  a full detection pass per frame. No cameras, people, or locations are
  invented anywhere in this backend.
- Temporary Pass (Formal_Code/visitor_store.py) issues time-limited passes to
  gate visitors: the guard photographs them, management approves, and the same
  InsightFace pipeline recognises them until their window closes. Overstay
  alerts are raised by the engine, never simulated here.
- Every page and API route requires a logged-in session, and the management-only
  actions additionally require an admin role (see ACCOUNTS below).
"""

from __future__ import annotations

import json
import os
import secrets
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from pydantic import BaseModel

import database
import people_store
import visitor_service

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "Formal_Code"))
import camera_store  # noqa: E402  (path must be set up first)
import data_pack  # noqa: E402
import onvif_discovery  # noqa: E402
import sentra_paths  # noqa: E402
import sentra_version  # noqa: E402
import updater  # noqa: E402
import visitor_store  # noqa: E402

app = FastAPI(title="Sentra Backend")

# --- Auth (demo-grade: fixed credentials, in-memory sessions) --------------
# This gates a local pitch demo, not a production deployment — see
# known_issues memory for what a real login would still need (password
# hashing, persistent sessions, etc.).
#
# Two roles exist because the Temporary Pass feature genuinely needs them: the
# guard on the gate raises a visit request, and someone in management decides
# on it. One account that can do both would make the approval step theatre.
#
#   admin — full access, including approving/extending/ending visits
#   guard — raises visit requests and sees who is on the premises, but every
#           approval control is refused by the API, not merely hidden in the UI
ROLE_ADMIN = "admin"
ROLE_GUARD = "guard"

ACCOUNTS = {
    # The shared demo login the pitch runs on. Deliberately an admin so a judge
    # picking it up is never blocked mid-demo.
    "sharktanktest": {"password": "demo", "role": ROLE_ADMIN, "display": "Demo Account"},
    "shauryavatwani": {"password": "shauryav", "role": ROLE_ADMIN, "display": "Shaurya Vatwani"},
    "guard": {"password": "testing", "role": ROLE_GUARD, "display": "Security Guard"},
}

SESSION_COOKIE = "sentra_session"

# token -> {"username", "role", "display"}
active_sessions: dict[str, dict] = {}


def session_for(request: Request) -> dict | None:
    token = request.cookies.get(SESSION_COOKIE)
    return active_sessions.get(token) if token else None


def is_authenticated(request: Request) -> bool:
    return session_for(request) is not None


def require_admin(request: Request) -> dict:
    """Guard the management-only actions.

    Enforced server-side on purpose: the dashboard also hides these controls
    from a guard, but a hidden button is a suggestion, not a permission.
    """
    session = session_for(request)
    if session is None:
        raise HTTPException(status_code=401, detail="Not authenticated")
    if session["role"] != ROLE_ADMIN:
        raise HTTPException(
            status_code=403,
            detail="Only the security management team can approve or change a visitor pass.",
        )
    return session


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
    account = ACCOUNTS.get(payload.username.strip().lower())
    # secrets.compare_digest keeps the check constant-time. The credentials are
    # printed on the login page, so this buys nothing here — it is simply the
    # right shape for the day these stop being demo accounts.
    if account is None or not secrets.compare_digest(payload.password, account["password"]):
        raise HTTPException(status_code=401, detail="Invalid username or password.")

    token = secrets.token_urlsafe(24)
    active_sessions[token] = {
        "username": payload.username.strip().lower(),
        "role": account["role"],
        "display": account["display"],
    }
    resp = JSONResponse({"ok": True, "role": account["role"]})
    resp.set_cookie(SESSION_COOKIE, token, httponly=True, samesite="lax")
    return resp


@app.post("/api/logout")
def logout(request: Request):
    token = request.cookies.get(SESSION_COOKIE)
    if token:
        active_sessions.pop(token, None)
    resp = JSONResponse({"ok": True})
    resp.delete_cookie(SESSION_COOKIE)
    return resp


@app.get("/api/me")
def whoami(request: Request):
    """Who is signed in, so the dashboard shows the real operator and can hide
    the controls their role cannot use."""
    session = session_for(request)
    if session is None:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return session


@app.get("/login")
def serve_login():
    return FileResponse(sentra_paths.app_dir() / "login.html")


# --- Engine / live-viewer relay state -------------------------------------

engine_connected = False
live_viewers: set[WebSocket] = set()

# One engine process streams every camera over this single WebSocket, each
# frame tagged with its camera_id. `engine_connected` only says the process
# itself is connected; a camera can still be individually offline (RTSP down)
# while the connection carrying other cameras' frames stays up, so per-camera
# status is tracked separately here rather than folded into one bool.
camera_last_seen: dict[str, float] = {}
# A little more than a couple of stream intervals (STREAM_INTERVAL_SECONDS in
# face_recognition.py) so one slow frame doesn't flap a camera to "offline".
CAMERA_ONLINE_TIMEOUT_SECONDS = 6.0


def is_camera_online(camera_id: str) -> bool:
    last_seen = camera_last_seen.get(camera_id)
    return last_seen is not None and (time.time() - last_seen) < CAMERA_ONLINE_TIMEOUT_SECONDS


@app.websocket("/ws/engine")
async def ws_engine(websocket: WebSocket) -> None:
    """face_recognition.py connects here as the single frame producer.

    This is a trusted local process, not a browser — it does not go through
    the login session (the dashboard viewers on /ws/live do). One connection
    carries frames for every camera the engine is running, distinguished by
    the camera_id in each message.
    """
    global engine_connected
    await websocket.accept()
    engine_connected = True
    try:
        while True:
            message = await websocket.receive_text()
            # Peeked at, not consumed: the original text is still relayed
            # unchanged below so this stays a fast passthrough rather than a
            # re-serialize of every frame.
            try:
                camera_id = json.loads(message).get("camera_id")
                if camera_id:
                    camera_last_seen[camera_id] = time.time()
            except (json.JSONDecodeError, AttributeError):
                pass

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
    cameras = camera_store.load_cameras()
    online_count = sum(1 for c in cameras if is_camera_online(c["id"]))
    visitors = visitor_store.counts_by_state()
    return {
        "cameras_online": online_count,
        "cameras_total": len(cameras),
        "registered_individuals": len(people_store.list_names()),
        "detections_today": database.detections_today_count(),
        # Both kinds of alert: fights, and visitors seen after their pass ran
        # out. One number, because the operator only has one attention span.
        "alerts_today": database.anomalies_today_count(),
        "visitors_pending": visitors["pending"],
        "visitors_on_premises": visitors["on_premises"],
        "visitors_overdue": visitors["overdue"],
    }


@app.get("/api/detections")
def get_detections(limit: int = 10):
    limit = max(1, min(limit, 200))
    return database.recent_detections(limit)


# --- Anomalies (fight detection + visitor overstays) -----------------------
# Rows are written by the AI engine itself (Formal_Code/anomaly_detection.py
# via event_logger.log_anomaly, and visitor_store.log_overstay), the same path
# detections take. The backend only reads them here — the /ws/engine relay
# stays a dumb, fast passthrough.


@app.get("/api/anomalies")
def get_anomalies(limit: int = 20, type: str | None = None):
    limit = max(1, min(limit, 200))
    return database.recent_anomalies(limit, anomaly_type=type)


# --- Temporary Pass (visitors) ---------------------------------------------
# The gate workflow: a guard captures a photo and requests a visit, management
# approves it, and from that moment the engine recognises the visitor until the
# window closes. Everything that changes a decision is admin-only, enforced
# here rather than only hidden in the dashboard.


class VisitorDecision(BaseModel):
    reason: str = ""


class VisitorExtension(BaseModel):
    minutes: int


@app.get("/api/visitors")
def get_visitors(status: str | None = None, limit: int = 200):
    limit = max(1, min(limit, 500))
    rows = visitor_store.list_visitors(status=status, limit=limit)
    return [visitor_store.public_visitor(row) for row in rows]


@app.post("/api/visitors")
async def create_visitor(
    request: Request,
    name: str = Form(...),
    duration_minutes: int = Form(...),
    photo: UploadFile = File(...),
    purpose: str = Form(""),
    host: str = Form(""),
):
    """Raise a visit request. Any signed-in user may do this — it is the guard's
    job, and it grants nothing on its own until an admin approves it."""
    session = session_for(request)
    try:
        visitor = visitor_service.create_visit_request(
            name=name,
            image_bytes=await photo.read(),
            duration_minutes=duration_minutes,
            requested_by=session["username"],
            purpose=purpose,
            host=host,
        )
    except (visitor_service.CaptureError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return visitor_store.public_visitor(visitor)


@app.get("/api/visitors/alerts")
def get_visitor_alerts(limit: int = 50):
    """Overstay alerts with their full timing detail.

    /api/anomalies also carries these so the Smart Alerts tab shows one unified
    feed; this route keeps the extra columns (approved-until, how long overdue,
    the photo) that the Temporary Pass tab shows and a fight alert has no
    equivalent of.
    """
    limit = max(1, min(limit, 200))
    return visitor_store.recent_visitor_alerts(limit)


@app.get("/api/visitors/{visitor_id}")
def get_visitor(visitor_id: str):
    visitor = visitor_store.get_visitor(visitor_id)
    if visitor is None:
        raise HTTPException(status_code=404, detail="Visitor not found.")
    return visitor_store.public_visitor(visitor)


@app.get("/api/visitors/{visitor_id}/photo")
def get_visitor_photo(visitor_id: str):
    visitor = visitor_store.get_visitor(visitor_id)
    if visitor is None:
        raise HTTPException(status_code=404, detail="Visitor not found.")
    path = visitor_service.photo_path(visitor.get("photo_file") or "")
    if path is None:
        raise HTTPException(status_code=404, detail="No photo stored for this visitor.")
    return FileResponse(path, media_type="image/jpeg")


def _visitor_action(action, visitor_id: str, *args) -> dict:
    """Run one visitor_store mutation, mapping its errors onto HTTP codes."""
    try:
        return visitor_store.public_visitor(action(visitor_id, *args))
    except KeyError:
        raise HTTPException(status_code=404, detail="Visitor not found.") from None
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/visitors/{visitor_id}/approve")
def approve_visitor(visitor_id: str, request: Request):
    session = require_admin(request)
    return _visitor_action(visitor_store.approve_visitor, visitor_id, session["display"])


@app.post("/api/visitors/{visitor_id}/reject")
def reject_visitor(visitor_id: str, request: Request, payload: VisitorDecision):
    session = require_admin(request)
    return _visitor_action(
        visitor_store.reject_visitor, visitor_id, session["display"], payload.reason
    )


@app.post("/api/visitors/{visitor_id}/extend")
def extend_visitor(visitor_id: str, request: Request, payload: VisitorExtension):
    session = require_admin(request)
    return _visitor_action(
        visitor_store.extend_visitor, visitor_id, payload.minutes, session["display"]
    )


@app.post("/api/visitors/{visitor_id}/revoke")
def revoke_visitor(visitor_id: str, request: Request):
    """End a visit now. The visitor stays under watch — see
    visitor_store.revoke_visitor for why that is the point, not an oversight."""
    session = require_admin(request)
    return _visitor_action(visitor_store.revoke_visitor, visitor_id, session["display"])


# Gate photos are deleted 24h after the visit's own pass expired — see
# visitor_store.PHOTO_RETENTION_HOURS for why the photo goes but the visit
# record does not. Run from the backend rather than the AI engine: the
# backend is the process guaranteed to be up whenever anyone can reach the
# dashboard at all, whereas the engine may be stopped (no camera attached yet,
# mid-restart) while photos still need to age out on schedule.
PHOTO_PURGE_INTERVAL_SECONDS = 3600


def _photo_purge_loop() -> None:
    while True:
        try:
            purged = visitor_store.purge_expired_photos()
            if purged:
                print(f"Temporary Pass: purged {purged} visitor photo(s) past the 24h retention window.")
        except Exception as exc:
            # A purge failure must never take the backend down with it —
            # try again on the next tick.
            print(f"Temporary Pass: photo purge failed: {exc}")
        time.sleep(PHOTO_PURGE_INTERVAL_SECONDS)


# Run once immediately (so a server that was down doesn't leave last week's
# expired photos sitting around for up to an hour after it comes back), then
# on the regular interval from a daemon thread.
visitor_store.purge_expired_photos()
threading.Thread(target=_photo_purge_loop, daemon=True).start()


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


# --- Data pack ------------------------------------------------------------
# Moves a populated system onto a fresh install as one uploaded file. The
# public installer ships empty of real people's biometric data, so this is how
# a new machine receives the registered staff, their photos and the detection
# history without anyone re-registering everyone by hand. Import is admin-only:
# it merges face embeddings and can replace the local database.


@app.get("/api/data-pack/export")
def export_data_pack(request: Request):
    require_admin(request)
    tmp = sentra_paths.DATA_ROOT / ".updates" / "Sentra-DataPack.sentra"
    info = data_pack.export_pack(tmp)
    return FileResponse(
        tmp,
        media_type="application/octet-stream",
        filename="Sentra-DataPack.sentra",
        headers={"X-Sentra-People": str(len(info["people"]))},
    )


@app.post("/api/data-pack/import")
async def import_data_pack(request: Request, file: UploadFile = File(...)):
    require_admin(request)
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    # Guard rail: a stray multi-GB upload should be rejected, not buffered whole.
    if len(raw) > 500 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="That file is too large to be a data pack.")
    try:
        return data_pack.import_pack(raw)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


# --- Cameras --------------------------------------------------------------
# Any number of cameras (camera_store.py). Passwords are write-only: a client
# never receives a stored password back, only `has_password` — see
# camera_store.public_camera. A blank password on an update means "leave the
# stored one alone" (camera_store.update_camera), so the edit form can be
# rendered without ever holding the real value.


class OnvifCredentials(BaseModel):
    ip: str = ""
    port: int = camera_store.DEFAULT_ONVIF_PORT
    username: str = onvif_discovery.DEFAULT_USERNAME
    password: str = ""


class CameraCreate(BaseModel):
    name: str
    location: str = ""
    description: str = ""
    connection_method: str = "manual"  # "manual" | "onvif"
    rtsp_url: str = ""
    onvif: OnvifCredentials | None = None
    ai_enabled: bool = True


class CameraPatch(BaseModel):
    name: str | None = None
    location: str | None = None
    description: str | None = None
    rtsp_url: str | None = None
    connection_method: str | None = None
    onvif: OnvifCredentials | None = None
    ai_enabled: bool | None = None


class DiscoverRequest(BaseModel):
    ip: str
    port: int = camera_store.DEFAULT_ONVIF_PORT
    username: str = onvif_discovery.DEFAULT_USERNAME
    password: str = ""


def _camera_with_status(camera: dict) -> dict:
    public = camera_store.public_camera(camera)
    public["online"] = is_camera_online(camera["id"])
    return public


@app.get("/api/cameras")
def get_cameras():
    return [_camera_with_status(c) for c in camera_store.load_cameras()]


@app.post("/api/cameras")
def create_camera(payload: CameraCreate):
    try:
        camera = camera_store.add_camera(
            payload.name,
            payload.rtsp_url,
            location=payload.location,
            description=payload.description,
            connection_method=payload.connection_method,
            onvif=payload.onvif.model_dump() if payload.onvif else None,
            ai_enabled=payload.ai_enabled,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _camera_with_status(camera)


@app.put("/api/cameras/{camera_id}")
def edit_camera(camera_id: str, payload: CameraPatch):
    try:
        camera = camera_store.update_camera(
            camera_id,
            name=payload.name,
            location=payload.location,
            description=payload.description,
            rtsp_url=payload.rtsp_url,
            connection_method=payload.connection_method,
            onvif=payload.onvif.model_dump() if payload.onvif is not None else None,
            ai_enabled=payload.ai_enabled,
        )
    except KeyError:
        raise HTTPException(status_code=404, detail="Camera not found.") from None
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _camera_with_status(camera)


@app.delete("/api/cameras/{camera_id}", response_model=DeleteResult)
def remove_camera(camera_id: str):
    try:
        deleted = camera_store.delete_camera(camera_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"deleted": deleted}


@app.post("/api/cameras/discover")
def discover_camera(payload: DiscoverRequest):
    """ONVIF auto-discovery: given an IP + credentials, find the camera's
    actual RTSP URL(s) rather than making the user hunt for one. Read-only —
    saving the result is a separate POST/PUT to /api/cameras."""
    try:
        profiles, device = onvif_discovery.discover_rtsp_urls(
            payload.ip, payload.username, payload.password, payload.port
        )
    except onvif_discovery.DiscoveryError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    recommended = onvif_discovery.best_profile(profiles)
    return {
        "device": device,
        "profiles": [p.to_dict() for p in profiles],
        "recommended_url": recommended.url,
    }


# --- Engine control ---------------------------------------------------------
# Restarts the one shared Formal_Code/face_recognition.py process: reloads the
# InsightFace model, reconnects the RTSP camera, and re-establishes the
# WebSocket stream to every backend listed in its DASHBOARD_WS_URLS (both
# this one and any sibling UI on another port). There is only ever one engine
# process regardless of which dashboard's button triggers the restart.

FORMAL_CODE_DIR = PROJECT_ROOT / "Formal_Code"
RUN_LOGS_DIR = sentra_paths.RUN_LOGS_DIR


def _engine_command() -> tuple[list[str], str | None]:
    """How to start the engine, as (argv, cwd).

    Packaged: one exe serves both roles, so re-invoke ourselves with --engine.
    From source: the engine must run under the same interpreter as this backend
    — the two conda envs on the dev Mac are not interchangeable (base has
    numpy 2.x and no ultralytics, which silently disables fight detection while
    leaving face recognition working, a failure that is very easy to miss).
    """
    if getattr(sys, "frozen", False):
        return [sys.executable, "--engine"], None
    # -u so the engine's log is written as it happens. Redirected to a file,
    # Python buffers stdout, and a healthy engine can go a long time writing
    # nothing at all — which makes the log useless for telling "running fine"
    # apart from "hung", exactly when you most need to know.
    return [sys.executable, "-u", "face_recognition.py"], str(FORMAL_CODE_DIR)


def _process_is_alive(pid: int) -> bool:
    """Is this pid still running? Read-only on both platforms.

    The obvious ``os.kill(pid, 0)`` is a harmless existence probe on POSIX and
    a *destructive* call on Windows: CPython maps every signal except the two
    console-control events onto ``TerminateProcess``, so signal 0 does not ask
    whether the process is alive, it kills it with exit code 0. Against a
    recycled pid — and Windows recycles pids briskly — that would terminate
    whatever unrelated program now holds the number.

    Windows therefore gets OpenProcess with the most limited access right that
    can answer the question, and never a signal.
    """
    if sys.platform == "win32":
        import ctypes

        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        STILL_ACTIVE = 259

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            return False  # gone, or not ours to inspect
        try:
            code = ctypes.c_ulong()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return False
            return code.value == STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)

    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _stop_running_engine() -> None:
    """Terminate the engine using the pid it recorded at startup.

    This used to shell out to `pkill`, which does not exist on Windows.
    os.kill with SIGTERM works on both platforms (on Windows it maps to
    TerminateProcess — a hard kill, so the engine's own cleanup does not run;
    the OS releases the RTSP socket regardless, which is what matters here).
    """
    pid_file = sentra_paths.ENGINE_PID_FILE
    try:
        pid = int(pid_file.read_text().strip())
    except (OSError, ValueError):
        return  # never started, or already cleaned up

    if pid == os.getpid():
        return  # refuse to kill the server itself

    try:
        os.kill(pid, signal.SIGTERM)
    except (OSError, ProcessLookupError):
        pass  # already gone

    # Give it a moment to exit before the replacement grabs the camera; two
    # processes holding the same RTSP stream is the failure this avoids.
    for _ in range(20):
        if not _process_is_alive(pid):
            break
        time.sleep(0.1)

    pid_file.unlink(missing_ok=True)


@app.post("/api/engine/restart")
def restart_engine():
    RUN_LOGS_DIR.mkdir(parents=True, exist_ok=True)
    _stop_running_engine()

    argv, cwd = _engine_command()
    # Detach so the engine outlives this request, and on Windows so it does not
    # pop up a console window each restart.
    kwargs: dict = {}
    if sys.platform == "win32":
        kwargs["creationflags"] = (
            subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
        )
    else:
        kwargs["start_new_session"] = True

    with open(sentra_paths.ENGINE_LOG_FILE, "a") as logf:
        subprocess.Popen(
            argv, cwd=cwd, stdout=logf, stderr=subprocess.STDOUT, **kwargs
        )
    return {"restarted": True}


# --- Version and updates ----------------------------------------------------
# The check runs by itself shortly after launch and the dashboard polls the
# result, but nothing installs on its own: downloading and applying are both
# explicit clicks. An update replaces the program files and never touches
# ProgramData\Sentra, so the database, faces, visitors and camera config all
# carry across (see windows/sentra.iss).


@app.get("/api/version")
def get_version():
    return sentra_version.describe()


@app.get("/api/update/state")
def get_update_state():
    """Poll-friendly: reports the cached state without touching the network."""
    return updater.state()


@app.post("/api/update/check")
def post_update_check(force: bool = True):
    return updater.check(force=force)


@app.post("/api/update/download")
def post_update_download(request: Request):
    # Downloading fetches an executable that will later run elevated, so this is
    # management's call — the same bar as approving a visitor.
    require_admin(request)
    return updater.download()


@app.post("/api/update/install")
def post_update_install(request: Request):
    require_admin(request)
    return updater.install()


@app.post("/api/update/dismiss")
def post_update_dismiss(request: Request):
    require_admin(request)
    return updater.clear_staged()


@app.on_event("startup")
def _schedule_update_check() -> None:
    updater.start_background_check()


# --- Static dashboard -------------------------------------------------------

DASHBOARD_PATH = sentra_paths.app_dir() / "dashboard.html"


@app.get("/")
def serve_dashboard(request: Request):
    if not is_authenticated(request):
        return RedirectResponse(url="/login")
    return FileResponse(DASHBOARD_PATH)
