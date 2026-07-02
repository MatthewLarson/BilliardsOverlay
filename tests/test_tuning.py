"""Camera-tuning score + config controls round-trip (hardware-free)."""
from pool_guide.config import Config, load_config, write_config
from pool_guide.capture import open_source
from pool_guide.vision.tuning import score_separation


def _synthetic_frame():
    cfg = Config()                      # standalone + synthetic by default
    src = open_source(cfg)
    for _ in range(4):
        f = src.read()
    return f.rgb, cfg


def test_score_separation_finds_balls_on_synthetic_felt():
    rgb, cfg = _synthetic_frame()
    r = score_separation(rgb, cfg.vision)
    assert r["balls"] >= 3            # synthetic table has 4 balls
    assert r["score"] > -1e5          # a real (finite) score, not the "no felt" sentinel
    assert 0 <= r["felt_v"] <= 255


def test_capture_controls_roundtrip(tmp_path):
    cfg = load_config()
    cfg.capture.controls = {"gain": 40, "brightness": 8, "auto_exposure": 1}
    path = tmp_path / "config.yaml"
    write_config(cfg, str(path))
    reloaded = load_config(str(path))
    assert reloaded.capture.controls == {"gain": 40, "brightness": 8, "auto_exposure": 1}


def test_controls_hidden_from_webui_form():
    # capture.controls is a dict -> must not appear as an editable form field
    from pool_guide.webui.server import build_config_sections
    cfg = load_config()
    cap = next(s for s in build_config_sections(cfg) if s["name"] == "capture")
    assert "capture.controls" not in {f["key"] for f in cap["fields"]}
