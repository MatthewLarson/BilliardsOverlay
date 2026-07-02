# Pool Guide

Ceiling-mounted **projector + camera (Xbox 360 Kinect)** augmented-reality
assistant for an 8‑ft pool table. It projects predicted ball trajectories, an
aim line, adjustable strength/contact controls, a "best shot" recommendation,
and guided practice drills onto the felt.

All planned phases (**0-5**) are implemented and tested: calibration, ball
detection, cue tracking + aim line, projected shot prediction with a real physics
engine plus strength/english controls, best-shot recommendation, and projected
practice drills with automatic scoring and coaching. See the roadmap at the
bottom for what each phase delivers and what's deliberately deferred.

---

## Two ways to run it (set in `config.yaml`)

| `mode` | Where the work happens | Use when |
|--------|------------------------|----------|
| `standalone` | One machine captures, computes, and projects. | A **Pi-only** rig, or a **laptop** with the Kinect + projector plugged in. |
| `distributed` | A **sensor node** on the Pi captures the Kinect and drives the projector; a **brain** (your PC) does the vision + physics and streams the overlay back. | You want the Pi at the table but need your PC's horsepower. Recommended for the real rig. |

The switch is one line in config; the same app code runs either way because
capture and display are pluggable behind a network layer.

> **Why not Pi-only for everything?** A Pi can pull Kinect frames *or* run the
> vision+physics pipeline, but doing both in real time is a stretch. Standalone
> Pi-only works for calibration and light use; `distributed` is the smoother
> path for live play. Both are supported — pick per your hardware.

---

## Install

> **Setting up the real rig?** See **[SETUP.md](SETUP.md)** for the full
> hardware → mounting → wiring → calibration → tuning guide. The steps below are
> just the software install.

Use **Python 3.10–3.12** (the vision/physics wheels are most reliable there).

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate      Linux/Pi:  source .venv/bin/activate
pip install -e .
cp config.example.yaml config.yaml      # then edit it
```

The **Kinect driver** (`freenect`) is **not** pip-installable and does not run on
Windows. On the Pi/Linux sensor machine, build `libfreenect` with its Python
wrapper (`sudo apt install freenect libfreenect-dev`, then build
`wrappers/python`). Use a **powered USB hub** — the Pi can't feed the Kinect's
camera on its own. For development on Windows with no Kinect, use
`capture.source: webcam` or `synthetic`.

---

## Web control panel (easiest way to drive it)

Run the **mobile-friendly control panel** on each node and manage everything from
your phone — no terminal needed:

```bash
python -m pool_guide.apps.webui        # then open http://<node-ip>:8080
```

The page adapts to the node's role (standalone / brain / sensor) from
`config.yaml`. From it you can edit any config option, start/stop/restart the
games and drills, and follow a **guided calibration wizard** — including tapping
the four table corners on a snapshot straight from your phone. On the brain, the
dashboard lists any sensor nodes that have checked in; if a node has never been
calibrated it walks you through it. Full walkthrough in
[SETUP.md](SETUP.md#6b-web-control-panel).

The per-app commands below still work — they're what the panel launches for you.

---

## Phase 0: calibrate the projector to the table

Mount the Kinect and projector on the ceiling, both pointing straight down and
rigidly fixed (any wobble ruins calibration).

**Standalone (laptop/Pi with camera + projector attached):**

```bash
# config.yaml:  mode: standalone,  capture.source: kinect_v1 (or webcam),
#               display.sink: projector,  display.{width,height}: projector resolution
python -m pool_guide.apps.calibrate
```

It projects a grid of ArUco markers, detects them in the camera over ~40 frames,
solves the **camera→projector homography**, and reports the mean reprojection
error (aim for **< 3 px**; > 8 px triggers a warning). It then asks you to click
the four **table corners** to also solve the **camera→table (mm)** mapping the
physics engine will need. Result is saved to `calibration.json`.

**Verify it:**

```bash
python -m pool_guide.apps.verify_calibration
```

Live green reticles are projected onto each detected marker. Move a printed
ArUco marker around the table — its reticle should stick to it. If a table
homography was saved, the projected orange rectangle should hug the real rails.

**Distributed:** on the Pi run the sensor node; on the PC run the same apps with
`mode: distributed`, `network.role: brain`:

```bash
# On the Pi (config: mode distributed, network.role sensor, brain_host = PC ip):
python -m pool_guide.apps.sensor_node
# On the PC (config: mode distributed, network.role brain):
python -m pool_guide.apps.calibrate --skip-table
```

---

## Phase 1: detect the balls, project a dot on each

With calibration saved, this is the visual proof that detection and calibration
agree — a projected ring lands on every real ball:

```bash
python -m pool_guide.apps.track_balls           # projects rings onto the table
python -m pool_guide.apps.track_balls --debug   # + a camera-view window for tuning
```

Detection is **adaptive**: it finds the felt colour automatically (green, blue,
or red cloth), masks it out, and treats round, ball-sized blobs inside the table
as balls. Pockets and rails are excluded. Each ball gets a coarse label
(`cue` / `8` / `solid` / `stripe`) and a stable id across frames. Reading the
printed ball *number* is a later refinement — Phase 1 targets reliable positions.

Optional **background subtraction** (more robust under tricky lighting): clear
the table, capture an empty reference, then enable it:

```bash
python -m pool_guide.apps.capture_background     # writes background.jpg
# then set vision.use_background_subtraction: true in config.yaml
```

If detection is noisy on your real table, tune the `vision:` block in
`config.yaml` (see comments there) or run with `--debug` to watch the mask.

---

## Phase 2: cue tracking + projected aim line

```bash
python -m pool_guide.apps.aim_assist            # projects the aim line onto the table
python -m pool_guide.apps.aim_assist --debug    # + camera-view window
```

Detects the cue stick (edge + line detection, collinear-segment merge), figures
out which way it aims (the line that passes closest to the cue ball), then casts
the shot ray and projects:

- the **cue-ball path**, reflecting off cushions up to `aim_max_bounces`,
- a **ghost-ball ring** at the first ball it would strike, and
- that ball's **predicted departure direction**.

This preview is **pure geometry** — straight lines and mirror bounces, no spin,
speed, or friction. For real, strength-and-english-aware trajectories, use the
Phase 3 predictor below. Cue *elevation* via the Kinect depth stream (for jump/
masse shots) is deferred to Phase 3+; the top-down aim angle it needs comes from
RGB alone.

---

## Phase 3: projected shot prediction + strength/english controls

Requires a **full** calibration (including the four table corners, i.e. the
camera→table homography — so pixels become real millimetres for the physics).

```bash
python -m pool_guide.apps.shot_predictor            # projects real predicted paths
python -m pool_guide.apps.shot_predictor --debug    # + camera-view window
```

Reads the balls + cue, then runs a physics engine and projects the **true
predicted paths of every ball** — cushions, ball-ball collisions, follow/draw,
side-spin throw, and potting. Adjust while aiming:

```
[ ]  strength      a d  side english      w s  follow / draw      c  centre
```

The projected **strength meter** and **cue-ball contact widget** show the current
values (they're also drawn on the felt so you set them at the table).

**Physics backends** (`physics.engine` in config):

- `simple` *(default)* — a built-in numpy simulator. Runs anywhere including
  Python 3.14 and the Pi. Models friction, collisions, cushions, english, and
  potting. Fast and deterministic, but **not** spin-accurate (no swerve/masse,
  no throw-from-cut). Tune `physics.friction_decel`/restitution to your cloth.
- `pooltool` — optional research-grade backend
  ([pooltool](https://github.com/ekiefl/pooltool)). It pins `panda3d`, which has
  **no wheel for Python 3.14** and is heavy on ARM, so it isn't installed by
  default. To use it, run the brain on **Python 3.10–3.12**,
  `pip install pooltool-billiards`, and set `physics.engine: pooltool`.

> **Camera resolution note:** over an 8-ft table, the Kinect v1's 640×480 RGB
> makes each ball only ~6 px across, which is marginal for detection. If balls
> track poorly on the real rig, a higher-resolution overhead camera will help
> more than any tuning.

---

## Phase 4: best-shot recommendation

```bash
python -m pool_guide.apps.best_shot            # projects the recommended shot
python -m pool_guide.apps.best_shot --debug    # + camera-view window
```

No cue stick needed — this is advice. It searches every **target ball × pocket ×
strength**, simulates each with the physics engine, and projects the highest
scoring shot: the target ball and pocket highlighted, the ghost-ball aim, and the
predicted cue + object paths. The strength meter is preset to the recommended
power. Scoring rewards potting, penalises scratches, and prefers a central
cue-ball leave for the next shot; obstructed lines are pruned automatically.

The search runs only when the table changes (or you press **SPACE**), using a
coarse/fast engine so the display stays responsive. `--include-8` allows
recommending the 8-ball even when other balls remain. Requires a full
calibration (camera→table) like the shot predictor.

---

## Phase 5: practice drills + coaching

```bash
python -m pool_guide.apps.drills                 # start with a suggested drill
python -m pool_guide.apps.drills --drill cut_shot
python -m pool_guide.apps.drills --list          # drills + your stats
python -m pool_guide.apps.drills --debug         # + camera-view window
```

The projector draws **where to place the balls**, the **target pocket(s)**, and
any **cue-ball position zone**. Set up the balls on the markers, take your shot,
and the vision system scores the attempt automatically — potted? scratched? cue
ball left in the zone? — then shows SUCCESS/MISS and the reason.

A drill runs as a state machine: `SETUP` (place the balls) → `READY` (take your
shot) → `SHOOTING` (balls moving) → `RESULT` (verdict), then re-racks. Scoring is
**count-based** (object balls before vs after, cue present or not), which is
robust to the tracker swapping ids during fast motion.

Built-in drills: straight-in pot, stop shot (position), cut shot, speed-control
lane, wagon wheel. Results are tracked in `drill_progress.json` with make-rate
and streaks; **`n`** jumps to the drill you should work on next (unplayed drills
first, then your lowest make-rate).

Keys: `n` next (suggested) · `[ ]` prev/next drill · `SPACE` force-ready · `r`
reset · `q` quit.

---

## Project layout

```
src/pool_guide/
  config.py                 typed config loader (config.yaml)
  capture/                  frame sources behind one interface
    base.py                   Frame + FrameSource contract
    kinect.py  webcam.py  synthetic.py  network_client.py
  display/                  overlay sinks (projector / window / network)
  net/protocol.py           ZeroMQ frame + overlay streaming (distributed mode)
  calibration/
    aruco.py                  build pattern, detect, solve homographies
    model.py  store.py        Calibration type + JSON persistence
  vision/                   Phases 1-2: perception
    table.py                  find the play area (calibrated rect or felt colour)
    balls.py                  detect + classify balls (cue/8/solid/stripe)
    tracking.py               stable ids across frames
    cue.py                    detect the cue stick + its aim direction
    aim.py                    ghost-ball + cushion-bounce aim geometry
  physics/                  Phase 3: shot simulation
    engine.py                 Shot / Trajectory / PhysicsEngine interface
    simple.py                 built-in numpy billiards simulator (default)
    pooltool_engine.py        optional high-fidelity pooltool backend
  ui/
    controls.py               strength + english state and projected widgets
  recommend.py              Phase 4: best-shot search + scoring
  drills/                   Phase 5: practice drills + coaching
    model.py                  Drill / BallSpec / TargetZone (normalised layouts)
    library.py                built-in drills
    session.py                setup->ready->shooting->result scoring state machine
    progress.py               JSON stats store + next-drill suggestion
  apps/
    calibrate.py              Phase 0: solve + save calibration
    verify_calibration.py     Phase 0: live accuracy check
    sensor_node.py            Pi agent for distributed mode
    capture_background.py     Phase 1: empty-table reference for bg subtraction
    track_balls.py            Phase 1: detect balls, project a dot on each
    aim_assist.py             Phase 2: cue tracking + projected aim line
    shot_predictor.py         Phase 3: physics prediction + strength/english UI
    best_shot.py              Phase 4: projected best-shot recommendation
    drills.py                 Phase 5: projected practice drills + scoring
tests/                      hardware-free calibration + vision + physics + drills tests
```

Run the tests: `pytest -q`

---

## Roadmap

- **Phase 0 — Rig & calibration** ✅
- **Phase 1 — Ball detection & static overlay** ✅ — adaptive felt masking,
  circle/blob detection, coarse classification, projected dot per ball.
- **Phase 2 — Cue tracking & aim line** ✅ — cue detection in RGB; projected
  ghost-ball aim line with cushion bounces (geometry-only preview).
- **Phase 3 — Projected controls & physics** ✅ — projected strength meter and
  cue-ball contact/english selector; numpy physics engine (pooltool optional)
  projecting the full predicted multi-ball trajectory with potting.
- **Phase 4 — Best-shot recommendation** ✅ — search target×pocket×strength with
  the engine, score by pot + scratch + cue leave, project the best shot.
- **Phase 5 — Drills & coaching** ✅ — projected drill layouts, automatic scoring
  from the vision system, progress tracking, and next-drill suggestions.

### Beyond the roadmap (not built)

The core is complete; these are natural next steps: reading ball **numbers**
(solids vs stripes) for rules-aware play; **camera-touch** input so the projected
controls are adjustable by hand instead of the keyboard; Kinect-**depth** cue
elevation for jump/masse; tuning the physics to your cloth; and richer strategy
(safeties, banks, combos). See the caveats under each phase above.

## Prior art / references

- [pooltool](https://github.com/ekiefl/pooltool) — the physics engine (Apache-2.0)
- [Cassapa](https://github.com/aporto/cassapa) — OpenCV AR aiming assistant
- [interactive-pool (GOSAI)](https://github.com/GOSAI-DVIC/interactive-pool) — ArUco projector calibration + background-subtraction detection
- [OpenPool](http://www.openpool.cc/) — Kinect-overhead + projector effects
- PoolLiveAid / POOL-AID — academic writeups of trajectory projection
