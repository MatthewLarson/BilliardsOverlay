"""Entrypoint for the web control panel.

Run on each node:
    python -m pool_guide.apps.webui

Then open http://<this-node-ip>:8080 from your phone or laptop (same LAN).
The page adapts to the node's role (standalone / brain / sensor) from config.yaml.
"""
from __future__ import annotations

import argparse

from ..config import load_config
from ..webui.server import get_ip, node_role, run_server


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Pool Guide web control panel")
    ap.add_argument("--config", default=None, help="path to config.yaml")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    role = node_role(cfg)
    url = f"http://{get_ip()}:{cfg.webui.port}"
    print(f"Pool Guide control panel [{role}] -> {url}")
    print("Open that address on any device on the same network. Ctrl-C to stop.")
    try:
        run_server(args.config, block=True)
    except KeyboardInterrupt:
        print("\nStopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
