"""Provision the Raspberry Pi sensor node over SSH.

Reads credentials from .env (git-ignored), connects with paramiko, and:
  * installs system deps (git, python venv, webcam + opencv runtime libs),
  * clones/updates this repo on the Pi,
  * creates a venv and installs the package + catt (for Chromecast),
  * writes config.yaml from deploy/sensor_config.yaml (brain host + Chromecast),
  * installs and starts a systemd service that runs the web control panel
    (which auto-starts webcam streaming + casting).

Usage:
    .venv/Scripts/python scripts/provision_pi.py           # full provision
    .venv/Scripts/python scripts/provision_pi.py --check   # connectivity only

Requires: pip install paramiko   (already installed in the brain venv).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import paramiko

# Remote output can contain non-cp1252 characters (piwheels progress, box-drawing).
# Force UTF-8 so printing them on a Windows console never crashes provisioning.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

ROOT = Path(__file__).resolve().parents[1]
REPO_URL = "https://github.com/MatthewLarson/BilliardsOverlay.git"
SERVICE = "pool-webui"


def load_env(path: Path) -> dict:
    env = {}
    if not path.exists():
        sys.exit(f"Missing {path}. Copy .env.example to .env and fill it in.")
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip()
    return env


class Pi:
    def __init__(self, env: dict):
        self.env = env
        self.user = env.get("PI_USER", "pi")
        self.password = env.get("PI_PASSWORD", "")
        self.app_dir = env.get("PI_APP_DIR", f"/home/{self.user}/pool_guide").replace(
            "PI_USER", self.user)
        self.client = paramiko.SSHClient()
        self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self.client.connect(
            env.get("PI_HOST"), port=int(env.get("PI_PORT", "22")),
            username=self.user, password=self.password, timeout=15,
            allow_agent=False, look_for_keys=False)

    def run(self, cmd: str, sudo: bool = False, check: bool = True) -> int:
        if sudo:
            full = f"sudo -S -p '' bash -lc {shq(cmd)}"
            stdin, stdout, stderr = self.client.exec_command(full, get_pty=True)
            stdin.write(self.password + "\n")
            stdin.flush()
        else:
            # merge stderr so progress/errors stream live and encode-safely
            stdin, stdout, stderr = self.client.exec_command(f"bash -lc {shq(cmd + ' 2>&1')}")
        for line in iter(stdout.readline, ""):
            sys.stdout.write("  " + line.rstrip() + "\n")
            sys.stdout.flush()
        rc = stdout.channel.recv_exit_status()
        err = stderr.read().decode(errors="replace").strip()
        if rc != 0 and check:
            print(err)
            raise RuntimeError(f"[rc={rc}] {cmd}")
        return rc

    def put(self, text: str, remote_path: str) -> None:
        sftp = self.client.open_sftp()
        with sftp.open(remote_path, "w") as f:
            f.write(text)
        sftp.close()

    def close(self):
        self.client.close()


def shq(s: str) -> str:
    return "'" + s.replace("'", "'\\''") + "'"


def sensor_config(env: dict) -> str:
    text = (ROOT / "deploy" / "sensor_config.yaml").read_text(encoding="utf-8")
    brain = env.get("BRAIN_HOST", "192.168.4.162")
    cast = env.get("CHROMECAST", "")
    text = text.replace('brain_host: "192.168.4.162"', f'brain_host: "{brain}"')
    text = text.replace('cast_target: ""', f'cast_target: "{cast}"')
    return text


def service_unit(user: str, app_dir: str) -> str:
    return f"""[Unit]
Description=Pool Guide control panel + sensor node
After=network-online.target
Wants=network-online.target

[Service]
User={user}
WorkingDirectory={app_dir}
ExecStart={app_dir}/.venv/bin/python -m pool_guide.apps.webui
Restart=on-failure
RestartSec=3

[Install]
WantedBy=multi-user.target
"""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Provision the Pi sensor node")
    ap.add_argument("--check", action="store_true", help="connectivity check only")
    ap.add_argument("--env", default=str(ROOT / ".env"))
    args = ap.parse_args(argv)

    env = load_env(Path(args.env))
    if not env.get("PI_PASSWORD"):
        sys.exit("PI_PASSWORD is empty in .env -- fill it in first.")

    print(f"Connecting to {env.get('PI_USER')}@{env.get('PI_HOST')} ...")
    pi = Pi(env)
    try:
        pi.run("echo Connected as $(whoami) on $(hostname), arch $(uname -m); python3 --version")
        if args.check:
            print("Connectivity OK.")
            return 0

        print("\n[1/6] System packages ...")
        pi.run("apt-get -qq update", sudo=True)
        pi.run("DEBIAN_FRONTEND=noninteractive apt-get -qq -y install "
               "git python3-venv python3-pip libgl1 libglib2.0-0 v4l-utils", sudo=True)

        print("\n[2/6] Clone / update repo ...")
        pi.run(f"if [ -d {pi.app_dir}/.git ]; then cd {pi.app_dir} && git pull; "
               f"else git clone {REPO_URL} {pi.app_dir}; fi")

        print("\n[3/6] Python venv + package + catt (this can take a few minutes) ...")
        pi.run(f"cd {pi.app_dir} && python3 -m venv .venv && "
               ".venv/bin/pip install -q -U pip wheel && "
               ".venv/bin/pip install -q -e . && .venv/bin/pip install -q catt")

        print("\n[4/6] Write sensor config.yaml ...")
        pi.put(sensor_config(env), f"{pi.app_dir}/config.yaml")

        print("\n[5/6] Verify imports + webcam ...")
        pi.run(f"cd {pi.app_dir} && .venv/bin/python -c "
               "\"import pool_guide, cv2, zmq, yaml; print('imports OK')\"")
        pi.run("v4l2-ctl --list-devices || echo 'no webcam listed (check USB)'", check=False)

        print("\n[6/6] Install + start service ...")
        pi.put(service_unit(pi.user, pi.app_dir), "/tmp/pool-webui.service")
        pi.run(f"cp /tmp/pool-webui.service /etc/systemd/system/{SERVICE}.service", sudo=True)
        pi.run("systemctl daemon-reload", sudo=True)
        pi.run(f"systemctl enable --now {SERVICE}", sudo=True)
        pi.run(f"systemctl --no-pager --lines=8 status {SERVICE} || true", check=False)

        print(f"\nDone. Sensor control panel: http://{env.get('PI_HOST')}:8080")
        print(f"Brain control panel:  http://{env.get('BRAIN_HOST')}:8080")
    finally:
        pi.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
