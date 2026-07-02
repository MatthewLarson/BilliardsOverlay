# Pool Guide — Full Setup Guide

This walks you from bare hardware to a working ceiling-projected pool assistant,
in the order that actually works: **mount → wire → install → calibrate → bring up
each feature → tune**. Do the phases in order; each one depends on the previous
one being solid (especially calibration).

If you just want the command cheat-sheet, jump to [Command reference](#command-reference).

---

## 1. Choose your topology first

Everything downstream depends on this one choice. It's a single line in config
(`mode:`), but it changes the wiring.

| | **Standalone** | **Distributed** |
|---|---|---|
| Who does the work | one machine captures + computes + projects | Pi captures + projects; a PC "brain" computes |
| Best for | a laptop with the Kinect + projector attached, or a Pi‑only trial | the real rig: Pi at the table, your PC does the heavy lifting |
| Kinect plugs into | that one machine | the Pi |
| Projector plugs into | that one machine | the Pi |
| Network needed | no | yes (Pi and PC on the same LAN) |

**Recommendation:** build and learn in **standalone on a laptop** (fastest
iteration), then move to **distributed** for the permanent ceiling install. A
Pi‑only standalone rig works for calibration and light use but will struggle to
run capture + vision + physics in real time.

---

## 2. Bill of materials

**You already have:** 8‑ft pool table, projector, Raspberry Pi, Xbox 360 Kinect
(a.k.a. Kinect **v1** — the libfreenect‑compatible one).

**You'll also need:**

- **Powered USB hub** — *required* for the Kinect. The Kinect's camera/audio
  draw more than a Pi (or many laptop ports) can supply; its wall adapter only
  feeds the tilt motor. Without a powered hub the Kinect times out.
- **Kinect → USB + power adapter** — the 360 Kinect uses a proprietary connector;
  you need the official "Kinect AC adapter / USB" cable if yours came off a
  console.
- **Ceiling mounts** for the projector and the Kinect (a rigid bracket or a
  shared board). **Rigidity matters more than anything** — any sag or vibration
  breaks calibration.
- **Cables to reach the ceiling:** long HDMI (projector), long active USB
  extension if the brain isn't at the table (Kinect USB is short and USB has a
  ~5 m passive limit — use an active/powered extension or keep the Pi on the
  ceiling in distributed mode).
- **The brain machine** (distributed) or the all‑in‑one machine (standalone): a
  laptop or mini‑PC. For the optional high‑fidelity `pooltool` physics backend it
  must run **Python 3.10–3.12** (see §5).

> **Raspberry Pi 5 (4 GB) notes.** A Pi 5 works well — it's much faster than the
> Pi 3/4 in older Kinect guides, and 4 GB is ample (this is CPU‑bound, not memory‑
> bound). Best role: the **sensor node** in distributed mode. It's also fine
> **standalone for Phases 0–2** (calibration, ball tracking, aim line). But
> per‑frame **Phase 3** and the **Phase 4** search are sluggish on the Pi alone —
> run distributed with a PC brain for smooth physics, or coarsen the engine on
> the Pi (set `physics.dt: 0.003` and lower `physics.max_time`, ~4× faster). Add
> the **active cooler** (sustained CV throttles a bare board); the powered USB hub
> is still required (the Kinect v1 is USB 2.0). Pi OS Bookworm ships Python 3.11
> and piwheels provides a prebuilt `opencv‑contrib‑python`, so install is quick.
> `pooltool` still isn't practical on ARM — use the default `simple` engine.
- Optional but recommended: a **higher‑resolution overhead USB camera**. See the
  resolution note in §7 — the Kinect's 640×480 RGB is marginal for ball
  detection over an 8‑ft table.

---

## 3. Mount the hardware

Both the **projector** and the **Kinect** go on the ceiling, centered over the
table, **pointing straight down**, rigidly fixed.

### Positioning

- **Center both over the middle of the playing surface.** Keep the Kinect as
  close to the projector lens as you can so they see the same area.
- **Projector:** it must cover the whole playing surface (~2.24 m × 1.12 m for an
  8‑ft table). Use your projector's throw ratio to check: `image width = throw
  ratio × distance`. Most ceilings need a short‑throw projector to fill an 8‑ft
  table from ~2.4–2.7 m. Aim/zoom so the image slightly overfills the cloth.
- **Kinect height:** the v1's RGB field of view is roughly 62° × 48°, so at a
  ~2.4–2.7 m ceiling it comfortably covers the long rail with margin. Higher =
  fewer pixels per ball (worse detection); lower = risk of not seeing the whole
  table. Exact framing doesn't need to be perfect — calibration handles the rest
  — as long as the **entire cloth plus a little margin is in view**.
- **Keystone:** get the projector as square to the table as possible mechanically.
  Avoid heavy digital keystone correction — it distorts the pixel grid and makes
  calibration less accurate.

### Safety & practicality

- Secure everything to a joist or a proper mount — a falling projector over a
  slate table is a bad day.
- Leave slack and strain‑relief on cables; route them so nobody snags them with a
  cue.
- You'll want to reach the Kinect/projector to nudge aim during first setup —
  don't seal it all up until calibration passes.

---

## 4. Wiring

### Standalone (one machine)

```
Kinect ──▶ Powered USB hub ──▶ USB on the machine
Projector ◀── HDMI ── the machine   (set as a second display / extended desktop)
```

### Distributed (Pi at the table, PC as brain)

```
Kinect ──▶ Powered USB hub ──▶ Raspberry Pi (USB)
Projector ◀── HDMI ── Raspberry Pi
Raspberry Pi ◀───── LAN / Wi‑Fi ─────▶ Brain PC
```

In distributed mode the Pi runs a thin **sensor node** (grab frames → send to PC;
receive overlay → show on projector); the PC runs the actual app.

---

## 5. Install the software

### 5a. Brain / all‑in‑one machine

Use **Python 3.10–3.12** if you want the optional `pooltool` backend; otherwise
3.13/3.14 is fine for the default engine.

```bash
git clone <your repo>  # or copy the c:\_pool_guide folder
cd pool_guide
python -m venv .venv
# Windows:  .venv\Scripts\activate
# Linux/mac: source .venv/bin/activate
pip install -e .
cp config.example.yaml config.yaml
```

That installs numpy, OpenCV (contrib build, for ArUco), PyYAML, and pyzmq, and
creates three console commands: `pool-calibrate`, `pool-verify`,
`pool-sensor-node` (all other apps run via `python -m pool_guide.apps.<name>`).

> **Windows + Kinect:** the Kinect driver (`freenect`) does **not** run on
> Windows. On Windows use `capture.source: webcam` or `synthetic` to develop; do
> the real Kinect capture on the Pi/Linux.

### 5b. Kinect driver (the machine the Kinect is plugged into — Linux/Pi)

`freenect` is **not** pip‑installable; build libfreenect with its Python wrapper:

```bash
sudo apt update
sudo apt install freenect libfreenect-dev python3-freenect  # names vary by distro
# If python3-freenect isn't packaged, build libfreenect from source and its
# wrappers/python, then `pip install` that wrapper into the venv.
```

Verify the Kinect is seen (with the powered hub connected):

```bash
freenect-glview      # should show live RGB + depth; Ctrl-C to quit
```

If `freenect-glview` works but Python can't import `freenect`, the Python wrapper
isn't in your venv — build/install `wrappers/python` from the libfreenect source.

### 5c. Optional: pooltool physics backend (brain, Python ≤3.12 only)

```bash
pip install pooltool-billiards
# then set physics.engine: pooltool in config.yaml
```

Skip this unless you want research‑grade physics — the default `simple` engine
runs everywhere and is fine to start.

---

## 6. Configure

Edit `config.yaml`. The important fields per machine:

### Standalone (laptop with Kinect + projector)

```yaml
mode: standalone
capture:
  source: kinect_v1        # or 'webcam' to test without a Kinect
  depth: false
display:
  sink: projector          # fullscreen on the projector display
  monitor: 2               # which display index is the projector
  width: 1280              # projector native resolution
  height: 720
calibration:
  table_length_mm: 2540    # measure your cloth! 8ft ~ 2540 x 1270
  table_width_mm: 1270
```

### Distributed

On the **Pi** (`config.yaml` on the Pi):

```yaml
mode: distributed
capture: { source: kinect_v1 }
display: { sink: projector, monitor: 1, width: 1280, height: 720 }
network:
  role: sensor
  brain_host: "192.168.1.50"   # your PC's LAN IP
```

On the **PC** (`config.yaml` on the PC):

```yaml
mode: distributed
network:
  role: brain
  brain_host: ""               # ignored for the brain
display: { width: 1280, height: 720 }   # must match the projector resolution
calibration: { table_length_mm: 2540, table_width_mm: 1270 }
```

> **Measure your cloth** and set `table_length_mm`/`table_width_mm` to the real
> playing surface (cushion nose to cushion nose). This is what converts camera
> pixels into millimetres for the physics.

**Projector as a second display:** set your OS to *extend* (not mirror) the
desktop onto the projector, and set `display.monitor` to the projector's index.
If the fullscreen window lands on the wrong screen, drag it over once, or adjust
`monitor`.

---

## 7. Bring-up sequence (do these in order)

For distributed mode, first start the **sensor node on the Pi** and leave it
running; then run each app **on the PC**:

```bash
# On the Pi:
python -m pool_guide.apps.sensor_node
```

### Phase 0 — Calibrate (the make‑or‑break step)

```bash
python -m pool_guide.apps.calibrate
# distributed brain: same command; it streams the pattern to the Pi's projector
```

It projects a grid of ArUco markers, watches the camera, and solves the
camera→projector mapping. Then it shows the camera image and asks you to **click
the four table corners in order: TL, TR, BR, BL** (Enter to accept, `r` to reset,
Esc to skip). This second step builds the camera→table‑millimetre mapping the
physics needs — **don't skip it** (only use `--skip-table` for a quick
projector‑only check).

- Aim for **mean reprojection error < 3 px**. Over ~8 px it warns you — see
  tuning below.
- Result is saved to `calibration.json`.

**Verify it:**

```bash
python -m pool_guide.apps.verify_calibration
```

Green reticles are projected onto detected markers; move a printed ArUco marker
around and its reticle should stick to it. If a table homography was saved, an
orange rectangle should hug the real rails. **If projected lines don't land on the
cloth, fix this before going further** — nothing downstream works without it.

### (Optional) Capture an empty‑table background

Improves detection under tricky lighting. Clear all balls, then:

```bash
python -m pool_guide.apps.capture_background
# then set vision.use_background_subtraction: true in config.yaml
```

### Phase 1 — Ball detection

```bash
python -m pool_guide.apps.track_balls --debug
```

A ring should be projected onto every ball; the `--debug` window shows the camera
view + the felt mask. Rack some balls and confirm they're all found and the
pockets/rails aren't false‑positives. Tune the `vision:` block if needed (§8).

> **Resolution reality check:** over an 8‑ft table the Kinect's 640×480 makes a
> ball only ~6 px across, which is marginal. If detection is flaky no matter how
> you tune it, a higher‑resolution overhead camera is the single biggest upgrade
> — set `capture.source: webcam` and point `capture.webcam_index` at it.

### Phase 2 — Cue tracking + aim line

```bash
python -m pool_guide.apps.aim_assist --debug
```

Hold your cue over the table; a projected aim line with cushion bounces and a
ghost ball should track it. Cue detection is the noisiest part — expect to tune
`cue_*` params and watch `--debug` to see what it's locking onto.

### Phase 3 — Shot prediction with strength + english

```bash
python -m pool_guide.apps.shot_predictor
```

Projects the true predicted paths of every ball. Adjust while aiming:
`[` `]` strength, `a`/`d` side english, `w`/`s` follow/draw, `c` centre, `q` quit.
Requires the full calibration (camera→table). Tune `physics:` to your cloth (§8).

### Phase 4 — Best‑shot recommendation

```bash
python -m pool_guide.apps.best_shot
```

Projects the highest‑scoring shot (target, pocket, aim, cue path). It recomputes
when the table changes or you press **SPACE**. `--include-8` allows recommending
the 8‑ball with balls remaining.

### Phase 5 — Practice drills

```bash
python -m pool_guide.apps.drills --list     # see drills + your stats
python -m pool_guide.apps.drills            # start a suggested drill
```

Place the balls on the projected markers, shoot, and it scores you automatically.
Keys: `n` next (suggested), `[`/`]` prev/next drill, `SPACE` force‑ready, `r`
reset, `q` quit. Stats are saved to `drill_progress.json`.

---

## 8. Tuning

### Calibration accuracy (high reprojection error / lines off the cloth)

- Make the mount **rigid** — the #1 cause of drift.
- **Dim the room lights** and lower **projector brightness** if glare washes out
  the ArUco markers (fewer markers detected → worse fit).
- Re‑run `calibrate`; increase `--samples` for a steadier average.
- Re‑click the table corners carefully (cushion nose, not the rail top).

### Ball detection (`vision:` block)

Run `track_balls --debug` and adjust:

- `min_ball_radius_px` / `max_ball_radius_px` — only used when there's no table
  calibration; otherwise the expected size is derived from the table scale.
- `felt_hue_tolerance`, `min_saturation`, `min_value` — widen/narrow the felt
  mask; raise `min_value` if pocket shadows are detected as balls.
- `table_erode_px` — increase to pull the play area further off the rails.
- `use_background_subtraction: true` (after capturing a background) for robustness
  under uneven lighting.

### Cue detection (`vision:` `cue_*`)

- `cue_min_length_frac` — raise if short lines (chalk, cables) get picked as the
  cue; lower if the cue isn't detected.
- `cue_max_ball_dist_px` — how close the cue line must pass to the cue ball to
  count as aiming at it.
- `cue_canny_lo/hi` — edge sensitivity for your lighting.

### Physics (`physics:` block) — match your real table

Defaults are generic. To make predicted distances match reality:

- `friction_decel` — **the main dial.** Increase if predicted balls roll too far;
  decrease if they stop too short. (Hit a ball a known distance and compare.)
- `restitution_cushion` — lower if real cushions kill more speed than predicted.
- `max_speed_mmps` — the cue‑ball speed at strength = 100%. Set so full power
  looks like your hardest realistic break.
- `restitution_ball`, `follow_draw_gain`, `side_throw_deg` — finer feel.

> The `simple` engine isn't spin‑accurate (no swerve/masse/throw‑from‑cut). For
> research‑grade physics, install and select the `pooltool` backend (§5c).

---

## 9. Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `freenect` import error | Python wrapper not in the venv; build libfreenect `wrappers/python`. On Windows this is expected — use `webcam`/`synthetic`. |
| Kinect times out / no frames | Not on a **powered USB hub**; or pulling RGB+depth at once — set `capture.depth: false`. |
| Calibration error > 8 px | Wobbly mount, glare on markers, or too few markers detected. Dim lights, lower projector brightness, re‑run. |
| Projected lines don't land on balls | Calibration bad or mount moved since calibrating — re‑run `calibrate` + `verify_calibration`. |
| "no camera→table homography" error (Phase 3/4/5) | You skipped the table corners — re‑run `calibrate` **without** `--skip-table` and click TL/TR/BR/BL. |
| Balls not detected | Resolution too low (see §7), or felt mask off — tune `vision:` with `track_balls --debug`. |
| Pockets detected as balls | Raise `vision.min_value`; increase `table_erode_px`. |
| Cue not detected / jumps around | Tune `cue_*`; reduce clutter/cables in view; watch `aim_assist --debug`. |
| Predicted distances wrong | Tune `physics.friction_decel` and restitution to your cloth (§8). |
| Fullscreen on the wrong screen | Set `display.monitor` to the projector index, or drag the window over once. |
| `pip install pooltool-billiards` fails | It needs Python ≤3.12 + panda3d. Use the default `simple` engine instead. |
| Distributed: PC shows nothing | Check `network.brain_host` on the Pi = PC's IP, firewall allows ports 5555/5556, sensor node is running. |

---

## 10. Optional: auto‑start the Pi sensor node

To have the Pi start streaming on boot (distributed mode), create a systemd
service (adjust paths/user):

```ini
# /etc/systemd/system/pool-sensor.service
[Unit]
Description=Pool Guide sensor node
After=network-online.target
Wants=network-online.target

[Service]
User=pi
WorkingDirectory=/home/pi/pool_guide
ExecStart=/home/pi/pool_guide/.venv/bin/python -m pool_guide.apps.sensor_node
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable --now pool-sensor.service
journalctl -u pool-sensor -f    # watch its logs
```

Note the projector display: a headless‑booted Pi may need the display forced on
(e.g. `hdmi_force_hotplug=1` in `/boot/config.txt`) so the projector receives a
signal.

---

## 11. Performance tips

- Prefer **distributed** for live play — keep vision + physics on the PC.
- Keep `capture.depth: false` until you actually need depth (Phase 3+ cue
  elevation, not yet used) — the second stream strains the Kinect/Pi.
- Best‑shot search cost grows with ball count; it runs on demand (table change /
  SPACE), not every frame. On weaker hardware, reduce the speed samples the search
  tries (in `apps/best_shot.py`) or cap targets.
- Good, **even** lighting on the cloth (no bright spots, no deep shadows) helps
  detection more than any single parameter.

---

## Command reference

| Command | What it does |
|---|---|
| `python -m pool_guide.apps.sensor_node` | (Pi, distributed) capture → stream; receive overlay → project |
| `python -m pool_guide.apps.calibrate` | Phase 0: solve + save calibration (click TL/TR/BR/BL) |
| `python -m pool_guide.apps.verify_calibration` | Phase 0: live accuracy check |
| `python -m pool_guide.apps.capture_background` | capture empty‑table reference for bg subtraction |
| `python -m pool_guide.apps.track_balls [--debug]` | Phase 1: detect balls, project a dot on each |
| `python -m pool_guide.apps.aim_assist [--debug]` | Phase 2: cue tracking + projected aim line |
| `python -m pool_guide.apps.shot_predictor [--debug]` | Phase 3: physics prediction + strength/english |
| `python -m pool_guide.apps.best_shot [--debug] [--include-8]` | Phase 4: best‑shot recommendation |
| `python -m pool_guide.apps.drills [--drill ID] [--list] [--debug]` | Phase 5: practice drills |
| `pytest -q` | run the hardware‑free test suite |

Console scripts (after `pip install -e .`): `pool-calibrate`, `pool-verify`,
`pool-sensor-node`.

### Keybindings

- **Calibrate (corner picking):** click TL, TR, BR, BL · Enter accept · `r` reset · Esc skip
- **Shot predictor:** `[` `]` strength · `a` `d` side english · `w` `s` follow/draw · `c` centre · `q`/Esc quit
- **Best shot:** SPACE recompute · `q`/Esc quit
- **Drills:** `n` next (suggested) · `[` `]` prev/next drill · SPACE force‑ready · `r` reset · `q`/Esc quit
- **Most apps:** `q` or Esc to quit

---

## Quick start (TL;DR)

1. Mount projector + Kinect on the ceiling, straight down, rigid; extend the
   desktop onto the projector.
2. Kinect → **powered USB hub** → the machine (Pi in distributed mode).
3. `pip install -e .`, `cp config.example.yaml config.yaml`, set `mode`,
   `display` resolution, and real `table_*_mm`.
4. (distributed) run `sensor_node` on the Pi.
5. `calibrate` → click the four corners → `verify_calibration` until lines land on
   the cloth.
6. Walk up the phases: `track_balls` → `aim_assist` → `shot_predictor` →
   `best_shot` → `drills`, tuning as you go.
