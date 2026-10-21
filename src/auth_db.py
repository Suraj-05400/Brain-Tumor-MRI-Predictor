"""SQLite persistence: user accounts (salted PBKDF2 password hashes) and per-user prediction history.

Demo-grade authentication for a college project: passwords are never stored in plain text, failed logins
are rate-limited, and every query is parameterised. For a real clinical product use a managed identity
provider, HTTPS and an audited database instead.
"""
import hashlib
import hmac
import json
import os
import re
import secrets
import shutil
import sqlite3
import time
import uuid
from contextlib import contextmanager
from datetime import datetime

DB_PATH = os.environ.get("BRAIN_APP_DB", "data/app.db")
UPLOAD_DIR = os.environ.get("BRAIN_APP_UPLOADS", "data/uploads")
ITERATIONS = 200_000                 # PBKDF2 work factor
MAX_FAILS, LOCK_SECONDS = 5, 300     # lock an account for 5 minutes after 5 wrong passwords
USERNAME_RE = re.compile(r"^[A-Za-z0-9_.-]{3,30}$")
_DUMMY_SALT = b"\x00" * 16


@contextmanager
def connect():
    """Open a connection, commit on success, always close."""
    os.makedirs(os.path.dirname(DB_PATH) or ".", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with connect() as c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE COLLATE NOCASE,
            full_name TEXT,
            salt BLOB NOT NULL,
            pw_hash BLOB NOT NULL,
            created_at TEXT NOT NULL,
            failed_attempts INTEGER NOT NULL DEFAULT 0,
            locked_until REAL NOT NULL DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            created_at TEXT NOT NULL,
            filename TEXT,
            model_name TEXT,
            predicted_class TEXT NOT NULL,
            confidence REAL NOT NULL,
            probs TEXT NOT NULL,
            image_path TEXT,
            note TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_pred_user ON predictions(user_id, created_at);
        """)


# ------------------------------------------------------------------ accounts
def _hash(password, salt):
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, ITERATIONS)


def check_password(password):
    if len(password) < 8:
        return False, "Password must be at least 8 characters."
    if len(password) > 128:
        return False, "Password is too long (max 128 characters)."
    if not (re.search(r"[A-Za-z]", password) and re.search(r"\d", password)):
        return False, "Password must contain at least one letter and one number."
    return True, "OK"


def register_user(username, password, full_name=""):
    """Create an account. Returns (ok, message)."""
    username = username.strip()
    if not USERNAME_RE.match(username):
        return False, "Username must be 3-30 characters: letters, numbers, dot, dash or underscore."
    ok, msg = check_password(password)
    if not ok:
        return False, msg
    salt = secrets.token_bytes(16)
    try:
        with connect() as c:
            c.execute("INSERT INTO users (username, full_name, salt, pw_hash, created_at) VALUES (?,?,?,?,?)",
                      (username, full_name.strip()[:60], salt, _hash(password, salt), datetime.now().isoformat(timespec="seconds")))
    except sqlite3.IntegrityError:
        return False, "That username is already taken."
    return True, "Account created. You can log in now."


def authenticate(username, password):
    """Check credentials. Returns (user_dict | None, message). Rate-limits repeated failures."""
    with connect() as c:
        row = c.execute("SELECT * FROM users WHERE username = ?", (username.strip(),)).fetchone()
        now = time.time()
        if row is None:
            _hash(password, _DUMMY_SALT)                       # same work as a real check (no user enumeration by timing)
            return None, "Invalid username or password."
        if row["locked_until"] > now:
            mins = int((row["locked_until"] - now) // 60) + 1
            return None, f"Too many failed attempts. Try again in about {mins} minute(s)."
        if hmac.compare_digest(_hash(password, row["salt"]), row["pw_hash"]):
            c.execute("UPDATE users SET failed_attempts = 0, locked_until = 0 WHERE id = ?", (row["id"],))
            return {"id": row["id"], "username": row["username"], "full_name": row["full_name"] or row["username"]}, "OK"
        fails = row["failed_attempts"] + 1
        locked = now + LOCK_SECONDS if fails >= MAX_FAILS else 0
        c.execute("UPDATE users SET failed_attempts = ?, locked_until = ? WHERE id = ?",
                  (0 if locked else fails, locked, row["id"]))
        return None, "Invalid username or password."


def delete_user(user_id):
    """Delete the account, all its predictions and its stored images."""
    with connect() as c:
        c.execute("DELETE FROM users WHERE id = ?", (user_id,))
    shutil.rmtree(os.path.join(UPLOAD_DIR, str(user_id)), ignore_errors=True)


# --------------------------------------------------------------- predictions
def save_prediction(user_id, filename, model_name, predicted_class, confidence, probs, image, note=""):
    """Store one prediction (and a <=512px copy of the image) for this user. Returns the new row id."""
    folder = os.path.join(UPLOAD_DIR, str(user_id))
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"{uuid.uuid4().hex}.png")
    img = image.convert("RGB")
    img.thumbnail((512, 512))
    img.save(path)
    with connect() as c:
        cur = c.execute(
            "INSERT INTO predictions (user_id, created_at, filename, model_name, predicted_class, confidence, probs, image_path, note)"
            " VALUES (?,?,?,?,?,?,?,?,?)",
            (user_id, datetime.now().isoformat(timespec="seconds"), os.path.basename(filename or "")[:120], model_name,
             predicted_class, float(confidence), json.dumps({k: float(v) for k, v in probs.items()}), path, (note or "").strip()[:300]))
        return cur.lastrowid


def list_predictions(user_id):
    """All predictions of this user, newest first, as dicts."""
    with connect() as c:
        rows = c.execute("SELECT * FROM predictions WHERE user_id = ? ORDER BY created_at DESC, id DESC", (user_id,)).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["probs"] = json.loads(d["probs"])
        out.append(d)
    return out


def delete_prediction(user_id, pred_id):
    """Delete one prediction, only if it belongs to this user. Returns True if something was deleted."""
    with connect() as c:
        row = c.execute("SELECT image_path FROM predictions WHERE id = ? AND user_id = ?", (pred_id, user_id)).fetchone()
        if row is None:
            return False
        c.execute("DELETE FROM predictions WHERE id = ? AND user_id = ?", (pred_id, user_id))
    if row["image_path"] and os.path.exists(row["image_path"]):
        os.remove(row["image_path"])
    return True
