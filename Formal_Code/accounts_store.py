"""Storage for login accounts.

The account list started as a hardcoded dict in ``backend_v2/main.py`` (demo
credentials for a pitch). That dict is now only the *seed* — accounts and any
password change are persisted here so a "change password" click survives a
restart and an update, the same way ``camera_store.py`` persists cameras
instead of keeping them as code.

File format (``Database/accounts.json``)::

    {
      "version": 1,
      "accounts": {
        "sharktanktest": {"password": "demo", "role": "admin", "display": "Demo Account"}
      }
    }

Still demo-grade: passwords are plaintext, not hashed (see the ``ACCOUNTS``
docstring in ``backend_v2/main.py``). Persisting them changes nothing about
that — it only makes a password change durable instead of reverting on
restart. The file is chmod 0600 on POSIX, same reasoning as
``camera_store.py``'s ONVIF credentials.
"""

from __future__ import annotations

import json
import os
import threading

from sentra_paths import ACCOUNTS_FILE as ACCOUNTS_FILE

ROLE_ADMIN = "admin"
ROLE_GUARD = "guard"

# The seed every install starts from. Only used the first time the store is
# read (no accounts.json yet, or one that fails to parse) — after that, the
# file on disk is the source of truth and this is never consulted again, so
# an account added here post-install does not retroactively appear for
# someone who already has a populated accounts.json.
DEFAULT_ACCOUNTS: dict[str, dict] = {
    "sharktanktest": {"password": "demo", "role": ROLE_ADMIN, "display": "Demo Account"},
    "shauryavatwani": {"password": "shauryav", "role": ROLE_ADMIN, "display": "Shaurya Vatwani"},
    "guard": {"password": "testing", "role": ROLE_GUARD, "display": "Security Guard"},
    # One-time setup passwords, deliberately neutral: this repo is public and
    # the installer ships this file, so anything written here is world-readable
    # forever. Each person signs in with their setup password and immediately
    # sets a real one in Settings -> Change your password, which is stored per
    # install and never travels back here.
    "rowanbandi": {"password": "rowan-setup-01", "role": ROLE_ADMIN, "display": "Rowan Bandi"},
    "veerpandeyy": {"password": "veer-setup-01", "role": ROLE_ADMIN, "display": "Veer Pandey"},
    "aarnagupta": {"password": "aarna-setup-01", "role": ROLE_ADMIN, "display": "Aarna Gupta"},
}

_lock = threading.RLock()


def _read_raw() -> dict:
    if not ACCOUNTS_FILE.is_file():
        return {"version": 1, "accounts": dict(DEFAULT_ACCOUNTS)}
    try:
        data = json.loads(ACCOUNTS_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {"version": 1, "accounts": dict(DEFAULT_ACCOUNTS)}

    if isinstance(data, dict) and isinstance(data.get("accounts"), dict):
        return {"version": 1, "accounts": data["accounts"]}
    return {"version": 1, "accounts": dict(DEFAULT_ACCOUNTS)}


def _write(accounts: dict[str, dict]) -> None:
    ACCOUNTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    payload = {"version": 1, "accounts": accounts}
    ACCOUNTS_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    try:
        os.chmod(ACCOUNTS_FILE, 0o600)
    except OSError:
        pass  # best effort; Windows ACLs don't map onto this


def load_accounts() -> dict[str, dict]:
    """Every account, seeding the file on first read so it exists on disk
    from then on rather than being silently re-derived every call."""
    with _lock:
        raw = _read_raw()
        if not ACCOUNTS_FILE.is_file():
            _write(raw["accounts"])
        return raw["accounts"]


def get_account(username: str) -> dict | None:
    return load_accounts().get(username.strip().lower())


def set_password(username: str, new_password: str) -> None:
    """Persist a new password for an existing account.

    Raises KeyError if the account does not exist and ValueError if the
    password is blank — callers should already have authenticated the
    session this username belongs to before calling this.
    """
    username = username.strip().lower()
    new_password = new_password or ""
    if not new_password:
        raise ValueError("Password cannot be blank.")

    with _lock:
        accounts = load_accounts()
        if username not in accounts:
            raise KeyError(username)
        accounts[username]["password"] = new_password
        _write(accounts)
