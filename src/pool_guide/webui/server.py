"""Web control panel server (Python stdlib http.server -- no web framework).

One of these runs on each node. It serves a mobile-friendly single-page app and
a small JSON API to:
  * report node status + calibration state, and discover peer nodes,
  * read/write config.yaml through a generated form schema,
  * start / stop / restart the pool apps (calibration, games, drills, sensor),
  * grab a camera snapshot and finish calibration by tapping the table corners.

Kept dependency-free (stdlib + numpy/opencv already in the project) so it runs
happily on a Raspberry Pi.
"""
from __future__ import annotations

import dataclasses
import json
import socket
import subprocess
import sys
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import yaml

from .. import config as config_mod
from ..config import load_config

STATIC_DIR = Path(__file__).parent / "static"
PROJECT_ROOT = Path(config_mod.__file__).resolve().parents[2]

# --- catalogue of things a node can run --------------------------------------
# roles: which node role may launch it. group: how the UI buckets it.
ACTIONS: dict[str, dict] = {
    "aim_assist": {"label": "Aim Assist", "group": "play", "module": "aim_assist",
                   "roles": ["standalone", "brain"],
                   "desc": "Live projected aim line with cushion bounces."},
    "shot_predictor": {"label": "Shot Predictor", "group": "play", "module": "shot_predictor",
                       "roles": ["standalone", "brain"],
                       "desc": "Full physics prediction with strength + english."},
    "best_shot": {"label": "Best Shot", "group": "play", "module": "best_shot",
                  "roles": ["standalone", "brain"],
                  "desc": "Projects the recommended shot."},
    "drills": {"label": "Practice Drills", "group": "train", "module": "drills",
               "roles": ["standalone", "brain"], "arg": "--drill",
               "desc": "Guided drills with automatic scoring."},
    "calibrate": {"label": "Calibrate (auto)", "group": "calibrate", "module": "calibrate",
                  "roles": ["standalone", "brain"], "default_args": ["--skip-table"],
                  "desc": "Project markers and solve camera->projector."},
    "verify_calibration": {"label": "Verify Calibration", "group": "calibrate",
                           "module": "verify_calibration", "roles": ["standalone", "brain"],
                           "desc": "Live accuracy check."},
    "capture_background": {"label": "Capture Background", "group": "calibrate",
                           "module": "capture_background", "roles": ["standalone", "brain"],
                           "desc": "Empty-table reference for detection."},
    "sensor_node": {"label": "Sensor Streaming", "group": "system", "module": "sensor_node",
                    "roles": ["sensor"],
                    "desc": "Capture the Kinect and stream to the brain."},
}

_ENUM_CHOICES = {
    "mode": ["standalone", "distributed"],
    "capture.source": ["kinect_v1", "webcam", "synthetic", "network"],
    "display.sink": ["projector", "window", "network"],
    "network.role": ["sensor", "brain"],
    "physics.engine": ["simple", "pooltool"],
}


def get_ip() -> str:
    """Best-effort primary LAN IP."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def node_role(cfg) -> str:
    if cfg.mode == "standalone":
        return "standalone"
    return cfg.network.role            # "brain" | "sensor"


# --- config schema (drives the Setup form) -----------------------------------
def _field_type(type_str: str) -> str:
    t = (type_str or "").lower()
    if "bool" in t:
        return "bool"
    if "int" in t:
        return "int"
    if "float" in t:
        return "float"
    return "str"


def build_config_sections(cfg) -> list[dict]:
    """Return [{name, fields:[{key, label, type, value, choices?}]}] for the UI."""
    sections: list[dict] = []
    general = {"name": "general", "fields": []}
    for f in dataclasses.fields(cfg):
        value = getattr(cfg, f.name)
        if dataclasses.is_dataclass(value):
            fields = []
            for sf in dataclasses.fields(value):
                key = f"{f.name}.{sf.name}"
                fields.append({
                    "key": key, "label": sf.name, "type": _field_type(str(sf.type)),
                    "value": getattr(value, sf.name),
                    "choices": _ENUM_CHOICES.get(key),
                })
            sections.append({"name": f.name, "fields": fields})
        else:
            general["fields"].append({
                "key": f.name, "label": f.name, "type": _field_type(str(f.type)),
                "value": value, "choices": _ENUM_CHOICES.get(f.name),
            })
    if general["fields"]:
        sections.insert(0, general)
    return sections


def _coerce(value, type_str: str):
    t = _field_type(type_str)
    try:
        if t == "bool":
            return bool(value) if not isinstance(value, str) else value.lower() in ("1", "true", "yes", "on")
        if t == "int":
            return int(value)
        if t == "float":
            return float(value)
    except (TypeError, ValueError):
        return value
    return value


def apply_config_updates(cfg, updates: dict) -> dict:
    """Merge {key: value} (key = 'section.field' or 'field') into cfg's dict, coerced."""
    data = dataclasses.asdict(cfg)
    # Build a type lookup from the schema.
    types = {}
    for sec in build_config_sections(cfg):
        for fld in sec["fields"]:
            types[fld["key"]] = str(fld["type"])
    for key, raw in updates.items():
        val = _coerce(raw, types.get(key, "str"))
        if "." in key:
            sec, name = key.split(".", 1)
            data.setdefault(sec, {})[name] = val
        else:
            data[key] = val
    return data


# --- subprocess management ---------------------------------------------------
class ProcessManager:
    """Runs at most one pool app at a time on this node."""

    def __init__(self, config_path: Path):
        self.config_path = config_path
        self._proc: subprocess.Popen | None = None
        self._action: str | None = None
        self._started = 0.0
        self._log: deque[str] = deque(maxlen=200)
        self._lock = threading.Lock()

    @property
    def running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def start(self, action_key: str, extra_args: list[str] | None = None) -> None:
        spec = ACTIONS[action_key]
        with self._lock:
            self._stop_locked()
            cmd = [sys.executable, "-u", "-m", f"pool_guide.apps.{spec['module']}",
                   "--config", str(self.config_path)]
            cmd += spec.get("default_args", [])
            cmd += extra_args or []
            self._log.clear()
            self._log.append(f"$ {' '.join(cmd)}")
            self._proc = subprocess.Popen(
                cmd, cwd=str(PROJECT_ROOT), stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, bufsize=1)
            self._action = action_key
            self._started = time.time()
            threading.Thread(target=self._pump, args=(self._proc,), daemon=True).start()

    def _pump(self, proc):
        try:
            for line in proc.stdout:                       # type: ignore[union-attr]
                self._log.append(line.rstrip())
        except (ValueError, OSError):
            pass

    def stop(self) -> None:
        with self._lock:
            self._stop_locked()

    def _stop_locked(self) -> None:
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=4)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        self._proc = None
        self._action = None

    def restart(self) -> None:
        action = self._action
        if action:
            self.start(action)

    def status(self) -> dict | None:
        if not self.running:
            return None
        spec = ACTIONS.get(self._action, {})
        return {"action": self._action, "label": spec.get("label", self._action),
                "pid": self._proc.pid, "uptime": round(time.time() - self._started, 1)}

    def logs(self) -> list[str]:
        return list(self._log)


# --- peer registry (brain tracks sensors that check in) ----------------------
class PeerRegistry:
    def __init__(self):
        self._peers: dict[str, dict] = {}
        self._lock = threading.Lock()

    def register(self, info: dict) -> None:
        host = info.get("host") or "?"
        with self._lock:
            self._peers[host] = {**info, "last_seen": time.time()}

    def active(self, ttl: float = 15.0) -> list[dict]:
        now = time.time()
        with self._lock:
            out = []
            for host, p in list(self._peers.items()):
                age = now - p["last_seen"]
                if age > ttl * 4:
                    del self._peers[host]
                    continue
                out.append({**p, "age": round(age, 1), "online": age <= ttl})
            return out


# --- the node bundles it all together ----------------------------------------
class Node:
    def __init__(self, config_path=None):
        # Match load_config's resolution when no explicit path is given (e.g. the
        # systemd service runs `webui` with no --config).
        if config_path is None:
            for cand in (PROJECT_ROOT / "config.yaml", PROJECT_ROOT / "config.example.yaml"):
                if cand.exists():
                    config_path = cand
                    break
            else:
                config_path = PROJECT_ROOT / "config.yaml"
        self.config_path = Path(config_path)
        self.pm = ProcessManager(self.config_path)
        self.peers = PeerRegistry()
        self.ip = get_ip()
        self.hostname = socket.gethostname()

    def cfg(self):
        return load_config(self.config_path)

    def calibration_status(self) -> dict:
        cfg = self.cfg()
        path = PROJECT_ROOT / cfg.calibration.path
        if not path.exists():
            return {"present": False, "has_table": False, "reproj_error": None}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return {"present": True, "has_table": data.get("H_cam2table") is not None,
                    "reproj_error": data.get("reproj_error_px")}
        except (ValueError, OSError):
            return {"present": False, "has_table": False, "reproj_error": None}

    def status(self) -> dict:
        cfg = self.cfg()
        role = node_role(cfg)
        calib = self.calibration_status()
        needs_cal = role in ("standalone", "brain") and not (calib["present"] and calib["has_table"])
        necessary = ["standalone"] if role == "standalone" else ["brain", "sensor"]
        return {
            "role": role, "mode": cfg.mode, "hostname": self.hostname, "ip": self.ip,
            "port": cfg.webui.port, "calibration": calib,
            "running": self.pm.status(), "necessary_nodes": necessary,
            "needs_calibration": needs_cal,
            "peers": self.peers.active() if role == "brain" else [],
            "actions": [k for k, s in ACTIONS.items() if role in s["roles"]],
        }


# --- HTTP handler ------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    node: Node = None            # set by run_server
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):   # quiet
        pass

    # -- response helpers --
    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,PUT,OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self._cors()
        self.end_headers()
        self.wfile.write(body)

    def _bytes(self, data: bytes, ctype: str, code=200):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self._cors()
        self.end_headers()
        self.wfile.write(data)

    def _read_json(self) -> dict:
        n = int(self.headers.get("Content-Length", 0) or 0)
        if not n:
            return {}
        try:
            return json.loads(self.rfile.read(n) or b"{}")
        except ValueError:
            return {}

    # -- verbs --
    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_GET(self):
        p = self.path.split("?")[0]
        if p in ("/", "/index.html"):
            return self._static("index.html")
        if p.startswith("/static/"):
            return self._static(p[len("/static/"):])
        if p == "/api/status":
            return self._json(self.node.status())
        if p == "/api/config":
            cfg = self.node.cfg()
            return self._json({"sections": build_config_sections(cfg)})
        if p == "/api/actions":
            return self._json({"actions": ACTIONS})
        if p == "/api/logs":
            return self._json({"lines": self.node.pm.logs()})
        if p == "/api/snapshot.jpg":
            return self._snapshot()
        return self._json({"error": "not found"}, 404)

    def do_POST(self):
        p = self.path.split("?")[0]
        body = self._read_json()
        if p == "/api/action/start":
            action = body.get("action")
            if action not in ACTIONS:
                return self._json({"error": "unknown action"}, 400)
            args = []
            spec = ACTIONS[action]
            if spec.get("arg") and body.get("value"):
                args = [spec["arg"], str(body["value"])]
            self.node.pm.start(action, args)
            return self._json({"ok": True, "status": self.node.status()})
        if p == "/api/action/stop":
            self.node.pm.stop()
            return self._json({"ok": True})
        if p == "/api/action/restart":
            self.node.pm.restart()
            return self._json({"ok": True})
        if p == "/api/register":
            self.node.peers.register(body)
            return self._json({"ok": True})
        if p == "/api/calibration/table":
            return self._calibration_table(body)
        return self._json({"error": "not found"}, 404)

    def do_PUT(self):
        if self.path.split("?")[0] == "/api/config":
            updates = self._read_json().get("updates", {})
            cfg = self.node.cfg()
            data = apply_config_updates(cfg, updates)
            (PROJECT_ROOT / "config.yaml").write_text(
                yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
            return self._json({"ok": True})
        return self._json({"error": "not found"}, 404)

    # -- endpoints --
    def _static(self, rel: str):
        path = (STATIC_DIR / rel).resolve()
        if not str(path).startswith(str(STATIC_DIR.resolve())) or not path.exists():
            return self._json({"error": "not found"}, 404)
        ctype = {"html": "text/html", "css": "text/css", "js": "application/javascript"}.get(
            path.suffix[1:], "application/octet-stream")
        self._bytes(path.read_bytes(), ctype + "; charset=utf-8")

    def _snapshot(self):
        if self.node.pm.running:
            return self._json({"error": "stop the running app first to grab a snapshot"}, 409)
        from ..capture import open_source
        from ..net.protocol import encode_jpeg
        try:
            cfg = self.node.cfg()
            src = open_source(cfg)
            frame = None
            for _ in range(15):
                frame = src.read()
                if frame is not None:
                    break
            src.close()
        except Exception as e:                                   # noqa: BLE001
            return self._json({"error": f"camera unavailable: {e}"}, 503)
        if frame is None:
            return self._json({"error": "no frame captured"}, 503)
        self._bytes(encode_jpeg(frame.rgb, 85), "image/jpeg")

    def _calibration_table(self, body):
        """Finish calibration: 4 tapped corners (TL,TR,BR,BL) -> camera->table."""
        from ..calibration import aruco, load_calibration, save_calibration
        pts = body.get("points")
        if not pts or len(pts) != 4:
            return self._json({"error": "need 4 corner points (TL,TR,BR,BL)"}, 400)
        cfg = self.node.cfg()
        calib_path = PROJECT_ROOT / cfg.calibration.path
        if not calib_path.exists():
            return self._json({"error": "run auto calibration (camera->projector) first"}, 409)
        calib = load_calibration(str(calib_path))
        corners = np.array(pts, dtype=np.float64)
        H_ct = aruco.solve_cam2table(corners, cfg.calibration.table_length_mm,
                                     cfg.calibration.table_width_mm)
        calib.H_cam2table = H_ct
        calib.table_size_mm = (cfg.calibration.table_length_mm, cfg.calibration.table_width_mm)
        save_calibration(calib, str(calib_path))
        return self._json({"ok": True, "calibration": self.node.calibration_status()})


# --- registration heartbeat (sensor -> brain) --------------------------------
def _register_loop(node: Node, stop: threading.Event):
    import urllib.request
    while not stop.is_set():
        cfg = node.cfg()
        if node_role(cfg) == "sensor":
            payload = json.dumps({
                "role": "sensor", "host": node.ip, "hostname": node.hostname,
                "port": cfg.webui.port,
                "calibration": node.calibration_status(),
                "running": node.pm.status(),
            }).encode()
            url = f"http://{cfg.network.brain_host}:{cfg.webui.peer_port}/api/register"
            try:
                req = urllib.request.Request(url, data=payload,
                                             headers={"Content-Type": "application/json"})
                urllib.request.urlopen(req, timeout=3).read()
            except OSError:
                pass
        stop.wait(5.0)


def run_server(config_path, block: bool = True):
    """Start the control panel. Returns (httpd, node, stop_event)."""
    node = Node(Path(config_path))
    Handler.node = node
    cfg = node.cfg()
    httpd = ThreadingHTTPServer((cfg.webui.host, cfg.webui.port), Handler)

    stop = threading.Event()
    threading.Thread(target=_register_loop, args=(node, stop), daemon=True).start()

    # Auto-start streaming on the sensor node.
    if node_role(cfg) == "sensor" and cfg.webui.auto_start_sensor:
        try:
            node.pm.start("sensor_node")
        except Exception:                                        # noqa: BLE001
            pass

    if not block:
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        return httpd, node, stop
    try:
        httpd.serve_forever()
    finally:
        stop.set()
        node.pm.stop()
    return httpd, node, stop
