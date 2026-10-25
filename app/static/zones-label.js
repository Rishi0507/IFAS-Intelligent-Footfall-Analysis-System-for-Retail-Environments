// Zone occupancy labelling: one saved frame at a time, every zone outlined,
// click to mark it taken, free or unsure. Writes zone_labels_<annotator>.json
// in the run folder, one file per annotator so agreement can be measured, and
// the dashboard's Validity card scores the system against it.
'use strict';

const $ = (id) => document.getElementById(id);
const NAME_OK = /^[A-Za-z0-9_-]{1,32}$/;
const CYCLE = [undefined, true, false, null];        // unlabelled, taken, free, unsure

const state = {
  run: null, zones: [], moments: [], index: 0, labels: {}, annotator: '',
  hover: null, saveTimer: null,
};

function store(key, value) {
  try { value == null ? localStorage.removeItem(key) : localStorage.setItem(key, value); } catch { /* blocked */ }
}
function recall(key) { try { return localStorage.getItem(key); } catch { return null; } }
async function getJSON(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || res.statusText);
  return res.json();
}
const key = (m) => m.t.toFixed(2);
const clock = (s) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, '0')}`;
const cssVar = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();

/* ----------------------------------------------------------------- saving */
function setSaveState(text, kind) { const el = $('save-state'); el.textContent = text; el.dataset.state = kind || ''; }
function scheduleSave() {
  if (!NAME_OK.test(state.annotator)) { setSaveState('Type your name to save', 'error'); return; }
  clearTimeout(state.saveTimer);
  setSaveState('Saving…');
  state.saveTimer = setTimeout(save, 300);
}
async function save() {
  const frames = {};
  for (const [t, row] of Object.entries(state.labels)) {
    const clean = Object.fromEntries(Object.entries(row).filter(([, v]) => v !== undefined));
    if (Object.keys(clean).length) frames[t] = clean;
  }
  try {
    const res = await fetch(`/api/runs/${encodeURIComponent(state.run)}/zone-labels`, {
      method: 'POST', headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ annotator: state.annotator, frames, zones_version: state.zones.map((z) => z.id).join(',') }),
    });
    if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || res.statusText);
    setSaveState('Saved', 'saved');
  } catch (err) {
    setSaveState(`Not saved: ${err.message}`, 'error');
  }
}

/* ---------------------------------------------------------------- drawing */
const canvas = $('frame-canvas');
const ctx = canvas.getContext('2d');
function layout() {
  const img = $('frame-img');
  const r = img.getBoundingClientRect();
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  canvas.width = Math.round(r.width * dpr);
  canvas.height = Math.round(r.height * dpr);
  canvas.style.width = `${r.width}px`;
  canvas.style.height = `${r.height}px`;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  draw();
}
function centroid(poly) {
  let a = 0, cx = 0, cy = 0;
  poly.forEach(([x1, y1], i) => {
    const [x2, y2] = poly[(i + 1) % poly.length];
    const c = x1 * y2 - x2 * y1;
    a += c; cx += (x1 + x2) * c; cy += (y1 + y2) * c;
  });
  return Math.abs(a) < 1e-9 ? poly[0] : [cx / (3 * a), cy / (3 * a)];
}
function inside([x, y], poly) {
  let hit = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const [xi, yi] = poly[i], [xj, yj] = poly[j];
    if ((yi > y) !== (yj > y) && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) hit = !hit;
  }
  return hit;
}
function draw() {
  const W = canvas.clientWidth, H = canvas.clientHeight;
  ctx.clearRect(0, 0, W, H);
  const m = state.moments[state.index];
  if (!m) return;
  const row = state.labels[key(m)] || {};
  const colors = { true: cssVar('--zone-taken'), false: cssVar('--zone-free'), null: cssVar('--muted') };
  state.zones.forEach((z) => {
    const v = row[z.id];
    ctx.beginPath();
    z.polygon.forEach(([x, y], i) => (i ? ctx.lineTo(x * W, y * H) : ctx.moveTo(x * W, y * H)));
    ctx.closePath();
    const c = v === undefined ? '#ffffff' : colors[String(v)];
    if (v !== undefined) { ctx.fillStyle = c; ctx.globalAlpha = v === null ? 0.15 : 0.28; ctx.fill(); ctx.globalAlpha = 1; }
    ctx.setLineDash(v === undefined || v === null ? [5, 4] : []);
    ctx.lineWidth = state.hover === z.id ? 3 : 1.8;
    ctx.strokeStyle = c;
    ctx.stroke();
    ctx.setLineDash([]);
    const [cx, cy] = centroid(z.polygon);
    const text = `${z.id}${v === true ? ' taken' : v === false ? ' free' : v === null ? ' ?' : ''}`;
    ctx.font = '600 11.5px Inter, system-ui, sans-serif';
    const w = ctx.measureText(text).width + 10;
    ctx.fillStyle = v === undefined ? 'rgba(0,0,0,.6)' : c;
    ctx.beginPath(); ctx.roundRect(cx * W - w / 2, cy * H - 9, w, 18, 9); ctx.fill();
    ctx.fillStyle = '#fff'; ctx.textBaseline = 'middle';
    ctx.fillText(text, cx * W - w / 2 + 5, cy * H + 0.5);
  });
}

/* ------------------------------------------------------------------ edits */
function setZone(zid, value) {
  const m = state.moments[state.index];
  const row = state.labels[key(m)] || (state.labels[key(m)] = {});
  row[zid] = value;
  renderMoment();
  scheduleSave();
}
function cycle(zid) {
  const m = state.moments[state.index];
  const cur = (state.labels[key(m)] || {})[zid];
  const i = CYCLE.findIndex((v) => v === cur);
  setZone(zid, CYCLE[(i + 1) % CYCLE.length]);
}
const point = (e) => {
  const r = canvas.getBoundingClientRect();
  return [(e.clientX - r.left) / r.width, (e.clientY - r.top) / r.height];
};
const zoneAt = (p) => [...state.zones].reverse().find((z) => inside(p, z.polygon));
canvas.addEventListener('click', (e) => { const z = zoneAt(point(e)); if (z) cycle(z.id); });
canvas.addEventListener('pointermove', (e) => {
  const z = zoneAt(point(e));
  const id = z ? z.id : null;
  canvas.style.cursor = z ? 'pointer' : '';
  if (id !== state.hover) { state.hover = id; draw(); }
});
canvas.addEventListener('pointerleave', () => { state.hover = null; draw(); });

function restFree() {
  const m = state.moments[state.index];
  const row = state.labels[key(m)] || (state.labels[key(m)] = {});
  state.zones.forEach((z) => { if (row[z.id] === undefined) row[z.id] = false; });
  renderMoment();
  scheduleSave();
}

/* --------------------------------------------------------------- rendering */
function momentState(m) {
  const row = state.labels[key(m)] || {};
  const n = state.zones.filter((z) => row[z.id] !== undefined).length;
  return n === 0 ? 'empty' : n === state.zones.length ? 'done' : 'partial';
}
function renderMoment() {
  const m = state.moments[state.index];
  $('moment-title').textContent = `Moment ${state.index + 1} of ${state.moments.length}`;
  $('moment-meta').textContent = `${clock(m.t)} into the recording`;
  const img = $('frame-img');
  const src = `/api/runs/${encodeURIComponent(state.run)}/${m.file}`;
  if (img.getAttribute('src') !== src) img.setAttribute('src', src);
  const row = state.labels[key(m)] || {};
  const n = state.zones.filter((z) => row[z.id] !== undefined).length;
  $('moment-count').textContent = `${n} of ${state.zones.length} zones labelled`;
  $('prev').disabled = state.index === 0;
  $('next').disabled = state.index === state.moments.length - 1;
  draw();
  renderRoster();
}
function renderRoster() {
  const done = state.moments.filter((m) => momentState(m) === 'done').length;
  $('progress').textContent = `${done} of ${state.moments.length} done`;
  $('progress-bar').style.width = `${(done / (state.moments.length || 1)) * 100}%`;
  $('roster').innerHTML = state.moments.map((m, i) =>
    `<button type="button" data-i="${i}" data-state="${momentState(m)}"${i === state.index ? ' aria-current="true"' : ''}>${clock(m.t)}</button>`).join('');
  [...$('roster').children].forEach((b) => { b.onclick = () => go(+b.dataset.i); });
  let taken = 0, free = 0, unsure = 0;
  Object.values(state.labels).forEach((row) => Object.values(row).forEach((v) => {
    if (v === true) taken++; else if (v === false) free++; else if (v === null) unsure++;
  }));
  $('tally').innerHTML = `<dt>Taken</dt><dd>${taken}</dd><dt>Free</dt><dd>${free}</dd><dt>Unsure</dt><dd>${unsure}</dd>`;
}
function go(i) {
  state.index = Math.max(0, Math.min(state.moments.length - 1, i));
  renderMoment();
}

/* ---------------------------------------------------------------- loading */
async function loadRun(id) {
  state.run = id;
  store('ifas-zone-label-run', id);
  const [doc, z] = await Promise.all([
    getJSON(`/api/runs/${encodeURIComponent(id)}/analytics.json`),
    getJSON(`/api/runs/${encodeURIComponent(id)}/zones`),
  ]);
  state.zones = z.zones?.zones || [];
  state.moments = doc.meta.frames || [];
  state.index = 0;
  state.labels = {};
  const ready = state.zones.length && state.moments.length;
  $('work').hidden = !ready;
  $('empty').hidden = !!ready;
  if (!ready) {
    $('empty-text').textContent = !state.zones.length
      ? 'This recording has no zones yet. Draw them (or find seats) in the dashboard’s Zones tab first.'
      : 'This recording has no saved frames. Process it again: the pipeline now saves a clean frame every 30 seconds.';
    return;
  }
  await loadLabels();
  const img = $('frame-img');
  img.onload = layout;
  renderMoment();
}
async function loadLabels() {
  if (!NAME_OK.test(state.annotator) || !state.run) return;
  try {
    const d = await getJSON(`/api/runs/${encodeURIComponent(state.run)}/zone-labels/${encodeURIComponent(state.annotator)}`);
    state.labels = d.frames || {};
  } catch { state.labels = {}; }
}

async function init() {
  state.annotator = recall('ifas-annotator') || '';
  $('annotator').value = state.annotator;
  $('annotator').addEventListener('change', async (e) => {
    state.annotator = e.target.value.trim();
    if (!NAME_OK.test(state.annotator)) { setSaveState('Names use letters, digits, _ and -', 'error'); return; }
    store('ifas-annotator', state.annotator);
    await loadLabels();
    if (state.moments.length) renderMoment();
    setSaveState('');
  });
  let runs = [];
  try { runs = await getJSON('/api/runs'); } catch { runs = []; }
  const usable = runs.filter((r) => r.has_zones && !r.demo);
  if (!usable.length) { $('empty').hidden = false; return; }
  $('run').innerHTML = usable.map((r) => `<option value="${r.id}">${r.source.replace(/[<>&"]/g, '')}</option>`).join('');
  const want = new URLSearchParams(location.search).get('run') || recall('ifas-zone-label-run');
  const pick = usable.find((r) => r.id === want) || usable[0];
  $('run').value = pick.id;
  $('run').onchange = (e) => loadRun(e.target.value);
  $('prev').onclick = () => go(state.index - 1);
  $('next').onclick = () => go(state.index + 1);
  $('rest-free').onclick = restFree;
  document.addEventListener('keydown', (e) => {
    if (e.target instanceof Element && e.target.closest('input, select, textarea')) return;
    if (e.key === 'ArrowRight') go(state.index + 1);
    else if (e.key === 'ArrowLeft') go(state.index - 1);
    else if (e.key === 'r' || e.key === 'R') restFree();
    else if (state.hover && 'tfuTFU'.includes(e.key)) setZone(state.hover, { t: true, f: false, u: null }[e.key.toLowerCase()]);
  });
  new ResizeObserver(layout).observe($('frame'));
  await loadRun(pick.id);
}

init();
