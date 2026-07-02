"use strict";
const $ = (s, r = document) => r.querySelector(s);
const view = $("#view");
let STATE = null, TAB = "home", cornerPts = [];

/* ---------- API ---------- */
async function api(path, opts) {
  const r = await fetch(path, opts);
  const ct = r.headers.get("content-type") || "";
  const data = ct.includes("json") ? await r.json() : await r.text();
  if (!r.ok) throw new Error((data && data.error) || r.statusText);
  return data;
}
const getJSON = (p) => api(p);
const post = (p, body) => api(p, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body || {}) });
const put = (p, body) => api(p, { method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body || {}) });

function toast(msg, isErr) {
  const t = $("#toast");
  t.textContent = msg; t.classList.toggle("err", !!isErr); t.classList.add("show");
  clearTimeout(toast._t); toast._t = setTimeout(() => t.classList.remove("show"), 2600);
}
const esc = (s) => String(s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

/* ---------- status polling ---------- */
async function refresh() {
  try {
    STATE = await getJSON("/api/status");
    paintChip();
    if (["home", "play", "train", "calibrate"].includes(TAB)) render();
    markCalTab();
  } catch (e) {
    $("#statusDot").className = "dot err";
    $("#brandSub").textContent = "offline";
  }
}
function paintChip() {
  const roleName = { standalone: "Standalone", brain: "Brain", sensor: "Sensor" }[STATE.role] || STATE.role;
  $("#roleLabel").textContent = roleName;
  $("#brandSub").textContent = `${STATE.hostname} · ${STATE.ip}`;
  const dot = $("#statusDot");
  dot.className = "dot " + (STATE.needs_calibration ? "warn" : "ok");
}
function markCalTab() {
  const tab = document.querySelector('.tab[data-tab="calibrate"]');
  let dot = tab.querySelector(".badge-dot");
  if (STATE && STATE.needs_calibration) { if (!dot) { dot = document.createElement("span"); dot.className = "badge-dot"; tab.appendChild(dot); } }
  else if (dot) dot.remove();
}

/* ---------- tabs ---------- */
document.querySelectorAll(".tab").forEach(b => b.addEventListener("click", () => {
  document.querySelectorAll(".tab").forEach(t => t.classList.remove("active"));
  b.classList.add("active"); TAB = b.dataset.tab; render();
}));

function render() {
  if (!STATE) return;
  ({ home: renderHome, play: renderPlay, train: renderTrain, calibrate: renderCalibrate, setup: renderSetup }[TAB] || renderHome)();
}

/* ---------- HOME ---------- */
function renderHome() {
  const c = STATE.calibration;
  const running = STATE.running;
  const calBadge = c.present && c.has_table
    ? `<span class="badge ok">ready${c.reproj_error != null ? " · " + c.reproj_error.toFixed(1) + "px" : ""}</span>`
    : `<span class="badge warn">needs calibration</span>`;
  let html = `
    <div class="card">
      <div class="row"><h3>This node</h3><span class="badge ok">${esc(STATE.role)}</span></div>
      <div class="kv"><span class="k">Host</span><span class="v">${esc(STATE.hostname)}</span></div>
      <div class="kv"><span class="k">Address</span><span class="v">${esc(STATE.ip)}:${STATE.port}</span></div>
      <div class="kv"><span class="k">Mode</span><span class="v">${esc(STATE.mode)}</span></div>
      <div class="kv"><span class="k">Calibration</span><span class="v">${calBadge}</span></div>
    </div>`;

  html += `<div class="card">
      <div class="row"><h3>Running now</h3>${running ? `<span class="badge ok">${esc(running.label)}</span>` : `<span class="badge">idle</span>`}</div>
      ${running ? `<div class="kv"><span class="k">Uptime</span><span class="v">${running.uptime}s · pid ${running.pid}</span></div>` : `<p class="desc">Nothing is running. Start something from Play, Train, or Calibrate.</p>`}
      <div class="btn-row">
        <button class="btn secondary small" ${running ? "" : "disabled"} onclick="doRestart()">Restart</button>
        <button class="btn danger small" ${running ? "" : "disabled"} onclick="doStop()">Stop</button>
        <button class="btn ghost small" onclick="showLogs()">Logs</button>
      </div>
    </div>`;

  if (STATE.role === "brain") {
    const peers = STATE.peers || [];
    html += `<div class="card"><h3>Sensor nodes</h3>`;
    if (!peers.length) html += `<p class="desc">No sensor has checked in yet. Start the control panel on the Pi (it registers automatically).</p>`;
    else peers.forEach(p => {
      html += `<div class="peer"><span class="dot ${p.online ? "ok" : "err"}"></span>
        <div style="flex:1"><b>${esc(p.hostname || p.host)}</b><div class="desc">${esc(p.host)}:${p.port || ""} · ${p.online ? "online" : "offline " + p.age + "s"}</div></div>
        <a class="btn ghost small" href="http://${esc(p.host)}:${p.port || 8080}" target="_blank" rel="noopener">Open</a></div>`;
    });
    html += `</div>`;
  }

  if (STATE.needs_calibration) {
    html += `<div class="card" style="border-color:var(--gold)">
      <div class="row"><h3>⚠️ Calibration needed</h3></div>
      <p class="desc">This node isn't calibrated yet. Let's set it up — it takes a couple of minutes.</p>
      <div class="btn-row"><button class="btn" onclick="gotoTab('calibrate')">Start calibration</button></div>
    </div>`;
  }
  view.innerHTML = html;
}

window.gotoTab = (t) => { document.querySelector(`.tab[data-tab="${t}"]`).click(); };
window.doStop = async () => { await post("/api/action/stop"); toast("Stopped"); refresh(); };
window.doRestart = async () => { await post("/api/action/restart"); toast("Restarted"); refresh(); };
window.showLogs = async () => {
  try { const d = await getJSON("/api/logs"); view.insertAdjacentHTML("afterbegin",
    `<div class="card"><div class="row"><h3>Recent output</h3><button class="btn ghost small" onclick="this.closest('.card').remove()">Close</button></div><pre class="log">${esc((d.lines || []).join("\n") || "(no output)")}</pre></div>`);
  } catch (e) { toast(e.message, true); }
};

/* ---------- PLAY / TRAIN (action cards) ---------- */
const EMOJI = { aim_assist: "🎯", shot_predictor: "🧮", best_shot: "⭐", drills: "🏋️",
  calibrate: "🎚️", verify_calibration: "✅", capture_background: "🖼️", sensor_node: "📡" };

function actionCard(key, spec, extra) {
  const running = STATE.running && STATE.running.action === key;
  return `<div class="card action-card">
    <div class="top"><div class="emoji">${EMOJI[key] || "▶️"}</div>
      <div style="flex:1"><h3>${esc(spec.label)} ${running ? '<span class="badge ok">running</span>' : ""}</h3>
        <p class="desc">${esc(spec.desc || "")}</p></div></div>
    ${extra || ""}
    <div class="btn-row">
      ${running
        ? `<button class="btn danger" onclick="doStop()">Stop</button>`
        : `<button class="btn" onclick='startAction(${JSON.stringify(key)})'>Start</button>`}
    </div></div>`;
}

async function actionsCatalog() {
  if (!actionsCatalog._c) actionsCatalog._c = (await getJSON("/api/actions")).actions;
  return actionsCatalog._c;
}

async function renderGroup(group, title, intro) {
  const cat = await actionsCatalog();
  const keys = STATE.actions.filter(k => cat[k] && cat[k].group === group);
  let html = `<h2 class="section">${title}</h2>`;
  if (intro) html += `<p class="desc" style="margin:0 4px 12px">${intro}</p>`;
  if (!keys.length) html += `<div class="card"><p class="desc">Nothing here for a ${esc(STATE.role)} node.</p></div>`;
  keys.forEach(k => {
    const extra = k === "drills" ? drillPicker() : "";
    html += actionCard(k, cat[k], extra);
  });
  view.innerHTML = html;
}
function drillPicker() {
  const drills = ["", "straight_pot", "stop_shot", "cut_shot", "speed_control", "wagon_wheel"];
  return `<label class="field"><span class="lbl">Drill (blank = suggested)</span>
    <select class="inp" id="drillSel">${drills.map(d => `<option value="${d}">${d || "★ suggested"}</option>`).join("")}</select></label>`;
}
window.startAction = async (key) => {
  const body = { action: key };
  if (key === "drills") { const s = $("#drillSel"); if (s && s.value) body.value = s.value; }
  try { await post("/api/action/start", body); toast(`Started ${key}`); refresh(); }
  catch (e) { toast(e.message, true); }
};
const renderPlay = () => renderGroup("play", "Play a game", "Pick a mode. Starting one stops whatever is running.");
const renderTrain = () => renderGroup("train", "Practice");

/* ---------- CALIBRATE (wizard) ---------- */
function renderCalibrate() {
  if (STATE.role === "sensor") {
    view.innerHTML = `<div class="card"><h3>Sensor node</h3><p class="desc">Calibration runs from the <b>brain</b>. This Pi just streams the camera. Open the brain's control panel to calibrate.</p></div>`;
    return;
  }
  const c = STATE.calibration;
  const nodesOk = STATE.mode === "standalone" || (STATE.peers || []).some(p => p.online);
  const s1 = nodesOk, s2 = c.present, s3 = c.has_table;
  const stepCls = (done, active) => done ? "done" : (active ? "active" : "");
  const running = STATE.running;

  let html = `<div class="card">
    <h3>Calibration</h3>
    <p class="desc">${s3 ? "You're calibrated. Re-run any step below if the rig moved." : "Follow the steps to line the projector up with your table."}</p>`;

  // Step 1 — nodes
  html += `<div class="step ${stepCls(s1, !s1)}"><div class="num">${s1 ? "✓" : "1"}</div><div class="body">
    <h4>Detect nodes</h4>
    <p>${STATE.mode === "standalone" ? "Standalone node ready." : (s1 ? "Sensor node online." : "Waiting for the sensor node to check in… start the control panel on the Pi.")}</p>
  </div></div>`;

  // Step 2 — auto calibration
  const busy2 = running && running.action === "calibrate";
  html += `<div class="step ${stepCls(s2, s1 && !s2)}"><div class="num">${s2 ? "✓" : "2"}</div><div class="body">
    <h4>Camera → projector</h4>
    <p>Projects a marker grid and solves the mapping automatically. Watch the table.${c.reproj_error != null ? ` <b>Error: ${c.reproj_error.toFixed(1)}px</b>` : ""}</p>
    <button class="btn ${s2 ? "secondary" : ""} small" ${!s1 || busy2 ? "disabled" : ""} onclick='startAction("calibrate")'>${busy2 ? "Running…" : (s2 ? "Re-run auto calibration" : "Run auto calibration")}</button>
  </div></div>`;

  // Step 3 — corners
  html += `<div class="step ${stepCls(s3, s2 && !s3)}"><div class="num">${s3 ? "✓" : "3"}</div><div class="body">
    <h4>Mark table corners</h4>
    <p>Tap the four inside cushion corners — <b>top-left, top-right, bottom-right, bottom-left</b> — on a snapshot.</p>
    <button class="btn ${s3 ? "secondary" : ""} small" ${!s2 ? "disabled" : ""} onclick="openCornerPicker()">${s3 ? "Redo corners" : "Mark corners"}</button>
    <div id="pickerHost"></div>
  </div></div>`;

  // Step 4 — verify
  const busy4 = running && running.action === "verify_calibration";
  html += `<div class="step ${stepCls(false, s3)}"><div class="num">4</div><div class="body">
    <h4>Verify</h4>
    <p>Project reticles onto the markers to check alignment. Optional but recommended.</p>
    <button class="btn secondary small" ${!s3 || busy4 ? "disabled" : ""} onclick='startAction("verify_calibration")'>${busy4 ? "Running…" : "Verify"}</button>
  </div></div>`;

  html += `</div>`;
  view.innerHTML = html;
}

window.openCornerPicker = async () => {
  const host = $("#pickerHost");
  host.innerHTML = `<p class="desc">Loading snapshot…</p>`;
  cornerPts = [];
  try {
    // ensure camera is free
    if (STATE.running) { await post("/api/action/stop"); }
    const url = "/api/snapshot.jpg?ts=" + Date.now();
    const resp = await fetch(url);
    if (!resp.ok) { const j = await resp.json().catch(() => ({})); throw new Error(j.error || "snapshot failed"); }
    const blob = await resp.blob();
    const img = URL.createObjectURL(blob);
    host.innerHTML = `
      <div class="picker" id="picker"><img id="snap" src="${img}" alt="table" /></div>
      <p class="desc" id="pickHint">Tap corner 1 of 4: <b>top-left</b></p>
      <div class="btn-row">
        <button class="btn secondary small" onclick="resetCorners()">Reset</button>
        <button class="btn small" id="submitCorners" disabled onclick="submitCorners()">Save corners</button>
      </div>`;
    $("#picker").addEventListener("click", onPickTap);
  } catch (e) {
    host.innerHTML = `<p class="desc" style="color:var(--danger)">${esc(e.message)}</p>
      <p class="desc">${STATE.mode !== "standalone" ? "In distributed mode the sensor must be streaming so the brain can grab a frame." : "Make sure the camera is connected and no app is using it."}</p>`;
  }
};
const CORNER_NAMES = ["top-left", "top-right", "bottom-right", "bottom-left"];
function onPickTap(ev) {
  if (cornerPts.length >= 4) return;
  const img = $("#snap"), rect = img.getBoundingClientRect();
  const x = (ev.clientX - rect.left) / rect.width * img.naturalWidth;
  const y = (ev.clientY - rect.top) / rect.height * img.naturalHeight;
  cornerPts.push([x, y]);
  const m = document.createElement("div");
  m.className = "marker"; m.textContent = cornerPts.length;
  m.style.left = ((ev.clientX - rect.left) / rect.width * 100) + "%";
  m.style.top = ((ev.clientY - rect.top) / rect.height * 100) + "%";
  $("#picker").appendChild(m);
  const hint = $("#pickHint");
  if (cornerPts.length < 4) hint.innerHTML = `Tap corner ${cornerPts.length + 1} of 4: <b>${CORNER_NAMES[cornerPts.length]}</b>`;
  else { hint.textContent = "All four set — save to finish."; $("#submitCorners").disabled = false; }
}
window.resetCorners = () => { cornerPts = []; openCornerPicker(); };
window.submitCorners = async () => {
  try { await post("/api/calibration/table", { points: cornerPts }); toast("Calibration complete 🎉"); refresh(); }
  catch (e) { toast(e.message, true); }
};

/* ---------- SETUP (config form) ---------- */
async function renderSetup() {
  let cfg;
  try { cfg = await getJSON("/api/config"); } catch (e) { view.innerHTML = `<div class="card"><p class="desc">${esc(e.message)}</p></div>`; return; }
  let html = `<h2 class="section">Configuration</h2>
    <p class="desc" style="margin:0 4px 12px">Changes save to config.yaml. Restart the running app (or this panel) to apply.</p>`;
  cfg.sections.forEach((sec, i) => {
    const open = ["general", "capture", "display", "network"].includes(sec.name);
    html += `<details class="group" ${open ? "open" : ""}><summary>${esc(sec.name)}</summary><div class="body">`;
    sec.fields.forEach(f => html += fieldInput(f));
    html += `</div></details>`;
  });
  html += `<button class="btn" onclick="saveConfig()">Save configuration</button>
    <div style="height:12px"></div>
    <button class="btn secondary" onclick="doRestart()">Restart running app</button>`;
  view.innerHTML = html;
}
function fieldInput(f) {
  const id = "cfg::" + f.key;
  if (f.type === "bool") {
    return `<div class="toggle" style="margin-bottom:12px"><label class="switch"><input type="checkbox" data-key="${f.key}" data-type="bool" id="${id}" ${f.value ? "checked" : ""}><span class="track"></span></label><span class="lbl" style="margin:0;text-transform:none">${esc(f.label)}</span></div>`;
  }
  if (f.choices) {
    return `<label class="field"><span class="lbl">${esc(f.label)}</span>
      <select class="inp" data-key="${f.key}" data-type="${f.type}" id="${id}">
        ${f.choices.map(c => `<option ${c === f.value ? "selected" : ""}>${esc(c)}</option>`).join("")}</select></label>`;
  }
  const numeric = f.type === "int" || f.type === "float";
  return `<label class="field"><span class="lbl">${esc(f.label)}</span>
    <input class="inp" data-key="${f.key}" data-type="${f.type}" id="${id}" type="${numeric ? "number" : "text"}" ${f.type === "float" ? 'step="any"' : ""} value="${esc(f.value)}"></label>`;
}
window.saveConfig = async () => {
  const updates = {};
  document.querySelectorAll("[data-key]").forEach(el => {
    const t = el.dataset.type;
    updates[el.dataset.key] = t === "bool" ? el.checked : el.value;
  });
  try { await put("/api/config", { updates }); toast("Saved ✓"); }
  catch (e) { toast(e.message, true); }
};

/* ---------- boot ---------- */
refresh();
setInterval(refresh, 3500);
