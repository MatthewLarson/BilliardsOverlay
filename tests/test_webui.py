"""Phase 6 (control panel) tests: config schema, HTTP API, table calibration.

The HTTP smoke test starts the real server on an ephemeral port and hits it with
urllib -- no browser, no hardware. Camera/subprocess endpoints aren't exercised
(they need hardware); their pure helpers are tested directly.
"""
import json
import urllib.request

import numpy as np

from pool_guide.config import load_config
from pool_guide.webui.server import (
    ACTIONS,
    apply_config_updates,
    build_config_sections,
    node_role,
    run_server,
)


def test_config_sections_cover_all_dataclasses():
    cfg = load_config()
    sections = build_config_sections(cfg)
    names = {s["name"] for s in sections}
    for expect in ("capture", "display", "network", "physics", "drills", "webui"):
        assert expect in names
    # 'mode' is a top-level enum with choices
    general = next(s for s in sections if s["name"] == "general")
    mode = next(f for f in general["fields"] if f["key"] == "mode")
    assert mode["type"] == "str" and "standalone" in mode["choices"]


def test_apply_config_updates_coerces_types():
    cfg = load_config()
    data = apply_config_updates(cfg, {
        "mode": "distributed",
        "capture.width": "1280",          # string -> int
        "physics.friction_decel": "750.5",  # string -> float
        "vision.use_background_subtraction": True,
    })
    assert data["mode"] == "distributed"
    assert data["capture"]["width"] == 1280 and isinstance(data["capture"]["width"], int)
    assert data["physics"]["friction_decel"] == 750.5
    assert data["vision"]["use_background_subtraction"] is True


def test_node_handles_none_config():
    # The systemd service runs `webui` with no --config; Node(None) must resolve
    # the default config path instead of crashing on Path(None).
    from pool_guide.webui.server import Node
    n = Node(None)
    assert n.status()["role"] in ("standalone", "brain", "sensor")


def test_node_role():
    cfg = load_config()
    cfg.mode = "standalone"
    assert node_role(cfg) == "standalone"
    cfg.mode = "distributed"
    cfg.network.role = "brain"
    assert node_role(cfg) == "brain"


def test_action_catalog_roles_valid():
    valid = {"standalone", "brain", "sensor"}
    for key, spec in ACTIONS.items():
        assert spec["roles"] and set(spec["roles"]) <= valid
        assert "module" in spec and "label" in spec


def _free_config(tmp_path):
    # write a config.yaml with an ephemeral webui port (0 lets the OS pick)
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        "mode: standalone\nwebui:\n  host: 127.0.0.1\n  port: 0\n", encoding="utf-8")
    return cfg_path


def test_http_status_and_config_endpoints(tmp_path):
    cfg_path = _free_config(tmp_path)
    httpd, node, stop = run_server(str(cfg_path), block=False)
    try:
        port = httpd.server_address[1]
        base = f"http://127.0.0.1:{port}"

        status = json.loads(urllib.request.urlopen(base + "/api/status", timeout=5).read())
        assert status["role"] == "standalone"
        assert "needs_calibration" in status
        assert "aim_assist" in status["actions"]

        cfg = json.loads(urllib.request.urlopen(base + "/api/config", timeout=5).read())
        assert any(s["name"] == "physics" for s in cfg["sections"])

        # index.html serves
        html = urllib.request.urlopen(base + "/", timeout=5).read().decode()
        assert "Pool Guide" in html
    finally:
        stop.set()
        httpd.shutdown()


def test_table_calibration_endpoint(tmp_path, monkeypatch):
    # Seed a calibration.json with only camera->projector, then POST corners.
    import pool_guide.webui.server as srv
    from pool_guide.calibration import aruco, save_calibration

    monkeypatch.setattr(srv, "PROJECT_ROOT", tmp_path)
    calib = aruco.make_calibration(np.eye(3), 0.5, (640, 480), (1280, 720), (2540, 1270))
    save_calibration(calib, str(tmp_path / "calibration.json"))

    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("mode: standalone\nwebui:\n  host: 127.0.0.1\n  port: 0\n", encoding="utf-8")
    httpd, node, stop = run_server(str(cfg_path), block=False)
    try:
        port = httpd.server_address[1]
        # corners TL,TR,BR,BL in camera px
        pts = [[64, 48], [576, 48], [576, 432], [64, 432]]
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/calibration/table",
            data=json.dumps({"points": pts}).encode(),
            headers={"Content-Type": "application/json"})
        out = json.loads(urllib.request.urlopen(req, timeout=5).read())
        assert out["ok"] and out["calibration"]["has_table"]
    finally:
        stop.set()
        httpd.shutdown()
