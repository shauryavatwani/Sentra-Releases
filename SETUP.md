# Sentra — Setup

A real, working CCTV face-recognition dashboard (not a demo). Includes the real
registered-people data for Dhaya, Ishan, Pranav, Veer, and Vishwa — treat this
like the personal/biometric data it is.

## 1. Requirements

- Python 3.10+
- ~1.5GB free disk space (InsightFace model weights download on first run)
- Internet access the first time you run it (to fetch those model weights)

## 2. Install dependencies

```bash
python3 -m pip install -r backend/requirements.txt
```

## 3. Run the dashboard

```bash
cd backend
python3 -m uvicorn main:app --host 0.0.0.0 --port 8000
```

Open **http://localhost:8000** (or `http://<your-machine's-LAN-IP>:8000` for
other devices on the same Wi-Fi).

Log in with:
- Username: `sharktanktest`
- Password: `demo`

(These are also shown directly on the login page.)

## 4. Connect a real camera (optional)

The dashboard works without a camera — it'll show real historical detections
and let you search/register people. To light up the **Live Monitor** tab with
an actual feed:

1. Go to the **Cameras** tab and set the RTSP URL to a camera reachable from
   this machine.
2. In a separate terminal, run:
   ```bash
   python3 Formal_Code/face_recognition.py
   ```
   This connects to that RTSP URL, runs face recognition, and streams to the
   dashboard. It also opens its own local preview window.

## 5. If InsightFace throws a numpy ABI error

Something like `ValueError: numpy.dtype size changed, may indicate binary
incompatibility...`. This is a compiled-extension mismatch, not a bug in this
code — fix it with:

```bash
python3 -m pip install --force-reinstall --no-deps scikit-image scipy
```

Then restart the `uvicorn` command from step 3.

## What's real vs. not

- Dashboard stats, person search, and the live feed are all real — nothing is
  simulated. If the AI engine isn't running, the UI says so honestly rather
  than faking a feed.
- Investigations and Smart Alerts are intentionally unbuilt ("Coming Soon" in
  the UI) — don't expect those tabs to do anything yet.
- Login is demo-grade: fixed credentials, sessions reset if the server
  restarts. Fine for a local demo, not for anything beyond that.
