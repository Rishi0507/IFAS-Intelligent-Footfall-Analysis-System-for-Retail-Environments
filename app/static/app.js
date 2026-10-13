/* Dashboard controller: load a run, filter it, render every view.

   All aggregation happens here on the client, over the *filtered* track list,
   so a filter chip changes every chart consistently - the analytics.json is
   the raw record, not a set of pre-baked chart payloads.

   The page is six tabbed views. One 3D stage (stage3d.js) is shared between
   the overview hero and the Floor tab: the element is moved, not rebuilt, so
   there is only ever one WebGL scene for it. */

import * as C from './charts.js';
import { hasWebGL, cssVar } from './three-kit.js';
import { createFloor } from './stage3d.js';
import { createSkyline } from './skyline3d.js';

const { fmt } = C;

const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];

const VIEWS = ['overview', 'floor', 'people', 'trends', 'video', 'log'];
const GENDERS = ['female', 'male'];
const AGES = ['Age16-30', 'Age31-45', 'Age46-60', 'AgeAbove61'];
const TIERS = ['Passing', 'Glancing', 'Browsing', 'Engaged', 'Dwelling'];
const AGE_SHORT = { 'Age16-30': '16 to 30', 'Age31-45': '31 to 45', 'Age46-60': '46 to 60', 'AgeAbove61': '61 and over' };
const AGE_TICK = { 'Age16-30': '16–30', 'Age31-45': '31–45', 'Age46-60': '46–60', 'AgeAbove61': '61+' };
const GENDER_PLURAL = { female: 'Women', male: 'Men' };
const COLOR = {
  female: 'var(--female)', male: 'var(--male)', none: 'var(--muted)', one: 'var(--neutral)',
  age: { 'Age16-30': 'var(--age-1)', 'Age31-45': 'var(--age-2)', 'Age46-60': 'var(--age-3)', 'AgeAbove61': 'var(--age-4)' },
  tier: Object.fromEntries(TIERS.map((t, i) => [t, `var(--tier-${i + 1})`])),
};
const cap = (s) => (s ? s[0].toUpperCase() + s.slice(1) : 'Unknown');
const plural = (n, one, many = `${one}s`) => `${fmt.int(n)} ${n === 1 ? one : many}`;
const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => `&#${c.charCodeAt(0)};`);

const ICON = {
  clock: '<svg viewBox="0 0 18 18"><circle cx="9" cy="9" r="6.6"/><path d="M9 5.2V9l2.6 1.7"/></svg>',
  people: '<svg viewBox="0 0 18 18"><circle cx="9" cy="5.6" r="2.8"/><path d="M3.4 15.6c.5-3.1 2.7-4.9 5.6-4.9s5.1 1.8 5.6 4.9"/></svg>',
  stay: '<svg viewBox="0 0 18 18"><path d="M5 2.4h8M5 15.6h8M5.8 2.4c0 3.4 3.2 4 3.2 6.6s-3.2 3.2-3.2 6.6M12.2 2.4c0 3.4-3.2 4-3.2 6.6s3.2 3.2 3.2 6.6"/></svg>',
  spot: '<svg viewBox="0 0 18 18"><path d="M9 16.2s5-4.5 5-8.4a5 5 0 0 0-10 0c0 3.9 5 8.4 5 8.4z"/><circle cx="9" cy="7.7" r="1.8"/></svg>',
};

const state = {
  doc: null,
  runId: null,
  runs: [],
  filters: { gender: new Set(), age: new Set(), tier: new Set(), minConf: 0 },
  video: null,
  playhead: 0,          // seconds into the recording; starts at the end (everything shown)
  view: 'overview',
  colorBy: 'gender',
  skyMetric: 'count',
};

let floor = null;       // 3D floor stage, or null when WebGL is unavailable
let skyline = null;     // created the first time the People tab opens

/* ------------------------------------------------------------- loading */
async function loadRuns(select) {
  try {
    const res = await fetch('/api/runs');
    if (!res.ok) throw new Error();
    state.runs = await res.json();
  } catch {
    state.runs = [];
  }
  renderRunList();
  if (!state.runs.length) return loadRun('demo');
  const pick = state.runs.find((r) => r.id === select) || state.runs[0];
  return loadRun(pick.id);
}

function renderRunList() {
  const host = $('#run-list');
  if (!state.runs.length) {
    host.innerHTML = '<p class="empty">Start the server with <code>python -m app.server</code> to list your recordings.</p>';
    return;
  }
  host.innerHTML = state.runs.map((r) => `
    <button role="option" data-run="${esc(r.id)}" aria-selected="${r.id === state.runId}">
      <span class="thumb"${r.has_poster ? ` style="background-image:url('/api/runs/${encodeURIComponent(r.id)}/poster.jpg')"` : ''}></span>
      <span class="r-text">
        <span class="r-name">${esc(r.source)}</span><br>
        <span class="r-meta">${plural(r.visitors, 'person', 'people')} in ${fmt.dur(r.duration_sec)}</span>
      </span>
      ${r.demo ? '<span class="badge">Sample</span>' : ''}
    </button>`).join('');
  $$('button', host).forEach((b) => {
    b.onclick = () => {
      $('#run-pop').hidePopover?.();
      loadRun(b.dataset.run);
    };
  });
}

let loadTicket = 0;
async function loadRun(id) {
  const ticket = ++loadTicket;
  // With the server: the API. Served as plain files (python -m http.server from
  // the repo root): the run directory, relative or absolute.
  const urls = [
    `/api/runs/${encodeURIComponent(id)}/analytics.json`,
    `../../outputs/runs/${id}/analytics.json`,
    `/outputs/runs/${id}/analytics.json`,
  ];
  let doc = null;
  for (const u of urls) {
    try {
      const res = await fetch(u);
      if (res.ok) { doc = await res.json(); break; }
    } catch { /* try the next location */ }
  }
  if (ticket !== loadTicket) return;        // a newer choice was made while this one loaded
  if (!doc) {
    $('#headline').textContent = 'This recording could not be loaded';
    $('#ov-sub').textContent = '';
    $('#summary').innerHTML = `<li>${ICON.spot}<span>Start the server with <code>python -m app.server</code>, or make the sample run with <code>python scripts/generate_demo_run.py</code>.</span></li>`;
    return;
  }
  floor?.pause();
  state.doc = doc;
  state.runId = id;
  state.playhead = doc.meta.duration_sec || 0;

  const run = state.runs.find((r) => r.id === id);
  $('#run-name').textContent = doc.meta.source || id;
  $('#run-meta').textContent = `${plural(doc.kpis.visitors, 'person', 'people')} in ${fmt.dur(doc.meta.duration_sec)}`;
  $$('#run-list button').forEach((b) => b.setAttribute('aria-selected', String(b.dataset.run === id)));
  $('#run-badge').hidden = !doc.meta.demo;
  $('#run-badge-text').textContent = 'Sample data, no models were run';
  const hasFrame = !!(run?.has_poster || run?.has_video);
  $('#sw-frame').hidden = !(floor && hasFrame);
  if (!hasFrame) { $('#chk-frame').checked = false; floor?.setLayers({ frame: false }); }

  setupVideo(id, doc);
  buildScrubBars(doc);
  renderAll();
  renderTrends();
  floor?.intro();
}

/* ------------------------------------------------------------ filtering */
function filtered() {
  const f = state.filters;
  return state.doc.tracks.filter((t) =>
    (!f.gender.size || f.gender.has(t.gender)) &&
    (!f.age.size || f.age.has(t.age)) &&
    (!f.tier.size || f.tier.has(t.tier)) &&
    (t.gender_conf >= f.minConf));
}
const filtering = () => {
  const f = state.filters;
  return !!(f.gender.size || f.age.size || f.tier.size || f.minConf);
};
const clearFilters = () => {
  state.filters = { gender: new Set(), age: new Set(), tier: new Set(), minConf: 0 };
  $('#f-conf').value = '0';
  syncChips();
};

function buildChips(host, values, key, colorOf, labelOf = (v) => v) {
  host.innerHTML = '';
  values.forEach((v) => {
    const b = document.createElement('button');
    b.className = 'chip';
    b.setAttribute('aria-pressed', 'false');
    b.dataset.value = v;
    b.dataset.key = key;
    b.innerHTML = (colorOf ? `<i class="swatch" style="background:${colorOf(v)}"></i>` : '') + labelOf(v);
    b.addEventListener('click', () => {
      const set = state.filters[key];
      set.has(v) ? set.delete(v) : set.add(v);
      b.setAttribute('aria-pressed', String(set.has(v)));
      renderAll();
    });
    host.appendChild(b);
  });
}

function syncChips() {
  $$('.filters .chip').forEach((b) => {
    b.setAttribute('aria-pressed', String(state.filters[b.dataset.key].has(b.dataset.value)));
  });
}

/* ------------------------------------------------------- aggregation */
const mean = (a) => (a.length ? a.reduce((s, v) => s + v, 0) / a.length : 0);
const median = (a) => {
  if (!a.length) return 0;
  const s = [...a].sort((x, y) => x - y);
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
};
const countBy = (rows, key, values) =>
  Object.fromEntries(values.map((v) => [v, rows.filter((r) => r[key] === v).length]));

function timeline(rows, doc) {
  const bin = doc.timeline.bin_sec;
  const n = doc.timeline.bins.length;
  const bins = [...Array(n)].map((_, i) => ({
    t: i * bin,
    entries: 0,
    gender: Object.fromEntries(GENDERS.map((g) => [g, 0])),
    age: Object.fromEntries(AGES.map((a) => [a, 0])),
  }));
  rows.forEach((r) => {
    const i = Math.min(n - 1, Math.floor(r.first_sec / bin));
    bins[i].entries++;
    if (r.gender) bins[i].gender[r.gender]++;
    if (r.age) bins[i].age[r.age]++;
  });
  return bins;
}

function occupancy(rows, doc, steps = 120) {
  const dur = doc.meta.duration_sec || 1;
  const step = dur / steps;
  const x = [], y = [];
  for (let i = 0; i <= steps; i++) {
    const t = i * step;
    x.push(t);
    y.push(rows.filter((r) => r.first_sec <= t && r.last_sec >= t).length);
  }
  return { x, y };
}

function histogram(values, nBins, lo, hi) {
  const counts = Array(nBins).fill(0);
  const w = (hi - lo) / nBins || 1;
  values.forEach((v) => { counts[Math.min(nBins - 1, Math.max(0, Math.floor((v - lo) / w)))]++; });
  return { counts, edges: [...Array(nBins + 1)].map((_, i) => lo + i * w) };
}

/* Seconds of presence per cell. Each step of a trail counts for as long as it
   lasted - trails are subsampled to 60 points, so counting points would make a
   long stay weigh the same as a short one. The 3D floor uses the same rule. */
function heatFrom(rows, doc) {
  const { cols, rows: nRows } = doc.heatmap;
  const grid = [...Array(nRows)].map(() => Array(cols).fill(0));
  const { width: W, height: H } = doc.meta;
  rows.forEach((r) => r.path.forEach(([t, x, y], i) => {
    const c = Math.min(cols - 1, Math.max(0, Math.floor((x / W) * cols)));
    const rr = Math.min(nRows - 1, Math.max(0, Math.floor((y / H) * nRows)));
    const next = r.path[i + 1];
    grid[rr][c] += r.path.length === 1 ? (r.dwell_sec || 0) : (next ? Math.max(0, next[0] - t) : 0);
  }));
  const peak = Math.max(...grid.flat(), 0) || 1;
  return { cols, rows: nRows, grid: grid.map((row) => row.map((v) => v / peak)) };
}

/* --------------------------------------------------------- table views */
const tableData = new Map();
function setTable(id, columns, rows) { tableData.set(id, { columns, rows }); }

function wireTableToggles() {
  $$('.toggle-table').forEach((btn) => {
    btn.onclick = () => {
      const id = btn.dataset.table;
      const chart = document.getElementById(id);
      let tbl = document.getElementById(`${id}-table`);
      if (!tbl) {
        tbl = document.createElement('div');
        tbl.id = `${id}-table`;
        tbl.className = 'table-wrap';
        chart.after(tbl);
      }
      const show = chart.style.display !== 'none';
      if (show) {
        const d = tableData.get(id);
        tbl.innerHTML = `<table><thead><tr>${d.columns.map((c, i) => `<th${i ? ' class="num"' : ''}>${c}</th>`).join('')}</tr></thead>
          <tbody>${d.rows.map((r) => `<tr>${r.map((c, i) => `<td${i ? ' class="num"' : ''}>${c}</td>`).join('')}</tr>`).join('')}</tbody></table>`;
      }
      chart.style.display = show ? 'none' : '';
      tbl.hidden = !show;
      btn.textContent = show ? 'Chart' : 'Table';
    };
  });
}

/* ------------------------------------------------------------- render */
function renderAll() {
  const doc = state.doc;
  if (!doc) return;
  const rows = filtered();
  const all = doc.tracks;
  $('#f-count').textContent = rows.length === all.length
    ? plural(all.length, 'visitor')
    : `${fmt.int(rows.length)} of ${plural(all.length, 'visitor')}`;
  $('#f-reset').hidden = !filtering();

  renderHero(doc, rows);
  renderInsights(doc);
  renderOverview(doc, rows);
  renderDemographics(doc, rows);
  renderEngagement(doc, rows);
  renderFlow(doc, rows);
  renderModel(doc, rows);
  renderLog(rows);
  renderStage(doc, rows);
  renderSkyline(rows);
  renderPlayhead();
  paintAmbient(doc, rows);
  wireTableToggles();
}

const simple = () => document.body.dataset.mode === 'simple';

function spokenDuration(sec) {
  if (sec < 90) return plural(Math.round(sec), 'second');
  if (sec < 3600) {
    const m = Math.floor(sec / 60), s = Math.round(sec - m * 60);
    return s && m < 10 ? `${plural(m, 'minute')} ${plural(s, 'second')}` : plural(Math.round(sec / 60), 'minute');
  }
  const h = Math.floor(sec / 3600), m = Math.round((sec - h * 3600) / 60);
  return m ? `${plural(h, 'hour')} ${plural(m, 'minute')}` : plural(h, 'hour');
}

/* The hero: the same numbers, said in words a shop owner would use. No model
   vocabulary - no confidence, tracks, occupancy or mA. */
function renderHero(doc, rows) {
  const m = doc.meta;
  const span = spokenDuration(m.duration_sec || 0);
  $('#headline').textContent = filtering()
    ? `${plural(rows.length, 'person matches', 'people match')} your filters, out of ${fmt.int(doc.tracks.length)}`
    : rows.length
      ? `${plural(rows.length, 'person', 'people')} came past in ${span}`
      : `Nobody came past in ${span}`;

  const when = m.captured_at ? new Date(m.captured_at) : null;
  const meta = [esc(m.source)];
  if (when && !Number.isNaN(+when)) {
    meta.push(`Recorded ${when.toLocaleDateString(undefined, { day: 'numeric', month: 'short', year: 'numeric' })} at ${when.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' })}`);
  }
  if (!simple()) meta.push(`${m.width}×${m.height} at ${fmt.num(m.fps, 0)} fps`);
  // A long file name is cut to two lines; the full name stays on hover.
  $('#ov-sub').innerHTML = meta.map((s, i) => `<span${i ? '' : ` title="${s}"`}>${s}</span>`).join('');

  const host = $('#summary');
  if (!rows.length) {
    host.innerHTML = `<li>${ICON.people}<span>${filtering() ? 'No one fits these filters. Clear them to see everyone.' : 'No people were found in this footage.'}</span></li>`;
    return;
  }
  const bins = timeline(rows, doc);
  const busiest = bins.reduce((b, c) => (c.entries > b.entries ? c : b), bins[0]);
  const g = countBy(rows, 'gender', GENDERS);
  const a = countBy(rows, 'age', AGES);
  const hasAge = rows.some((r) => r.age);
  const topAge = AGES.reduce((x, k) => (a[k] > a[x] ? k : x), AGES[0]);
  const med = median(rows.map((r) => r.dwell_sec));
  const zones = ['Left', 'Center', 'Right'];
  const inZone = (z) => rows.filter((r) => r.zone === z).length;
  const topZone = zones.reduce((x, z) => (inZone(z) > inZone(x) ? z : x), zones[0]);
  const lingered = rows.filter((r) => r.dwell_sec >= 30).length;
  const women = g.female >= g.male;

  const lines = [
    [ICON.clock, `Busiest at <b>${fmt.clock(busiest.t)}</b>, when ${plural(busiest.entries, 'person', 'people')} arrived.`],
    [ICON.people, `Mostly <b>${women ? 'women' : 'men'}</b> (${fmt.pct((Math.max(g.female, g.male) / rows.length) * 100, 0)})`
      + (hasAge
        ? `, and the biggest age group was <b>${AGE_SHORT[topAge]}</b> (${fmt.pct((a[topAge] / rows.length) * 100, 0)}).`
        : '. Ages were not measured in this run.')],
    [ICON.stay, `Half of them stayed longer than <b>${fmt.dur(med)}</b>`
      + (lingered ? `, and ${fmt.int(lingered)} stayed more than 30 seconds.` : '.')],
    // Only name a busiest part when one actually leads.
    [ICON.spot, zones.filter((z) => inZone(z) === inZone(topZone)).length > 1
      ? 'People were spread evenly across the view.'
      : `The <b>${topZone.toLowerCase()}</b> of the view was the busiest part (${fmt.pct((inZone(topZone) / rows.length) * 100, 0)} of people).`],
  ];
  // Same for gender: a small or half-hidden person can go either way.
  const shaky = rows.filter((r) => r.gender && r.gender_conf < 0.7).length;
  if (shaky) {
    lines[1][1] += ` The system was unsure about the gender of ${shaky === rows.length ? 'all of them' : plural(shaky, 'person', 'people')}.`;
  }
  // The age model says how sure it was. When it was not, say so next to the claim.
  const aged = rows.filter((r) => r.age);
  if (aged.length && mean(aged.map((r) => r.age_conf)) < 0.5) {
    lines[1][1] += ' The age model was unsure here, so treat the age as a rough guess.';
  }
  host.innerHTML = lines.map(([ico, text]) => `<li>${ico}<span>${text}</span></li>`).join('');
}

function renderOverview(doc, rows) {
  const m = doc.meta;
  const plain = simple();
  $('#occ-title').textContent = plain ? 'How busy it was' : 'Occupancy';
  $('#occ-sub').textContent = plain
    ? 'How many people were in view at the same time.'
    : 'People in frame at the same moment. Click the chart to jump there in the recording.';
  $('#peak-label').textContent = plain ? 'Most at once' : 'Peak occupancy';
  $('#dwell-label').textContent = plain ? 'Average visit' : 'Average dwell';

  const bins = timeline(rows, doc);
  const occ = occupancy(rows, doc);
  const dwells = rows.map((r) => r.dwell_sec);
  const g = countBy(rows, 'gender', GENDERS);

  $('#k-visitors').textContent = fmt.int(rows.length);
  $('#k-visitors-note').textContent = rows.length === doc.tracks.length
    ? `in ${fmt.dur(m.duration_sec)} of footage`
    : `of ${fmt.int(doc.tracks.length)} before filters`;
  C.sparkline($('#k-spark'), bins.map((b) => b.entries), COLOR.one);

  const peak = Math.max(0, ...occ.y);
  $('#k-peak').textContent = fmt.int(peak);
  $('#k-peak-note').textContent = `${fmt.num(mean(occ.y), 1)} on average`;
  C.sparkline($('#k-peak-spark'), occ.y, COLOR.one);

  $('#k-dwell').innerHTML = `${fmt.num(mean(dwells), 1)}<small>s</small>`;
  $('#k-dwell-note').textContent = `Median ${fmt.num(median(dwells), 1)}s, longest ${fmt.dur(Math.max(0, ...dwells))}`;
  const dh = histogram(dwells, 16, 0, Math.max(10, ...dwells));
  C.sparkline($('#k-dwell-spark'), dh.counts, COLOR.one);

  const busiest = bins.reduce((b, c) => (c.entries > b.entries ? c : b), bins[0] || { entries: 0, t: 0 });
  $('#k-rate').innerHTML = `${fmt.int(busiest.entries)}<small>${busiest.entries === 1 ? 'arrival' : 'arrivals'}</small>`;
  $('#k-rate-note').textContent =
    `${fmt.clock(busiest.t)} to ${fmt.clock(busiest.t + doc.timeline.bin_sec)}, ${fmt.num(rows.length / ((m.duration_sec || 1) / 60), 1)} a minute overall`;
  C.sparkline($('#k-rate-spark'), bins.map((b) => b.entries), COLOR.one);

  C.splitBar($('#k-gender-split'), [
    { label: 'Women', value: g.female, color: COLOR.female },
    { label: 'Men', value: g.male, color: COLOR.male },
  ]);
  $('#k-gender-note').textContent = `${plural(g.female, 'woman', 'women')} and ${plural(g.male, 'man', 'men')}`
    + (!plain && g.female && g.male ? `, ${fmt.num(g.female / g.male, 2)} women for each man` : '');

  const hasAge = rows.some((r) => r.age);
  let best = { n: -1 };
  GENDERS.forEach((gg) => (hasAge ? AGES : [null]).forEach((aa) => {
    const n = rows.filter((r) => r.gender === gg && (!aa || r.age === aa)).length;
    if (n > best.n) best = { n, g: gg, a: aa };
  }));
  $('#k-segment').textContent = best.n > 0
    ? `${GENDER_PLURAL[best.g]}${best.a ? ` aged ${AGE_SHORT[best.a]}` : ''}` : 'None';
  $('#k-segment-note').textContent = best.n > 0
    ? `${plural(best.n, 'person', 'people')}, ${fmt.pct((best.n / rows.length) * 100)} of everyone`
      + (hasAge ? '' : '. Ages were not measured.')
    : 'No visitors match';

  const gc = mean(rows.map((r) => r.gender_conf));
  const ac = mean(rows.map((r) => r.age_conf));
  $('#k-conf').textContent = fmt.fixed(gc);
  $('#k-conf-meter').style.width = `${gc * 100}%`;
  $('#k-conf-note').textContent = `Gender ${fmt.fixed(gc)} and age ${fmt.fixed(ac)}, averaged over visitors`;

  // Arrivals over time, stacked by gender.
  C.groupedColumns($('#c-footfall'), {
    categories: bins.map((b) => fmt.clock(b.t)),
    series: GENDERS.map((gg) => ({ name: GENDER_PLURAL[gg], color: COLOR[gg], values: bins.map((b) => b.gender[gg]) })),
    stacked: true,
    height: 250,
  });
  C.legend($('#l-footfall'), GENDERS.map((gg) => ({ label: GENDER_PLURAL[gg], color: COLOR[gg] })));
  setTable('c-footfall', ['From', 'Women', 'Men', 'Total'],
    bins.map((b) => [fmt.clock(b.t), b.gender.female, b.gender.male, b.entries]));

  C.lines($('#c-occupancy'), {
    x: occ.x,
    series: [{ name: 'In view', color: COLOR.one, values: occ.y }],
    area: true,
    height: 210,
    // Read at draw time, so the marker follows a replay without a re-mount.
    marker: () => (state.playhead < (m.duration_sec || 0) - 0.01 ? state.playhead : null),
    onSeek: (t) => seek(t),
  });
  setTable('c-occupancy', ['Time', 'In view'],
    occ.x.filter((_, i) => i % 8 === 0).map((t, i) => [fmt.clock(t), occ.y[i * 8]]));
}

function renderDemographics(doc, rows) {
  const a = countBy(rows, 'age', AGES);
  const plain = simple();

  // The same panels, described for whoever is reading them.
  const copy = plain ? {
    demoTitle: 'Who came in', demoSub: 'The mix of people the camera saw.',
    ageTitle: 'Age groups', ageSub: 'Roughly how old your visitors were.',
    agTitle: 'Age by gender', agSub: 'Which groups visit you most.',
  } : {
    demoTitle: 'Demographics', demoSub: 'What the age and gender models found, set against each other.',
    ageTitle: 'Age mix', ageSub: 'The four PETA age buckets. Click a bar to filter the whole dashboard.',
    agTitle: 'Age by gender', agSub: 'Head count for each age bucket, women beside men.',
  };
  $('#demo-title').textContent = copy.demoTitle;
  $('#demo-sub').textContent = copy.demoSub;
  $('#agemix-title').textContent = copy.ageTitle;
  $('#agemix-sub').textContent = copy.ageSub;
  $('#agegender-title').textContent = copy.agTitle;
  $('#agegender-sub').textContent = copy.agSub;
  $('#video-sub').textContent = plain
    ? 'Your footage, with a box around each person the system found.'
    : 'The annotated output. Box colours match the charts.';

  const noAge = rows.length && !rows.some((r) => r.age);
  const notice = plain
    ? '<p class="empty">Ages were not measured for this video, so there is no age breakdown yet.</p>'
    : '<p class="empty">No age model ran for this video, so there is no age breakdown. '
      + 'Point <code>--age-bundle</code> at a trained bundle to fill this in.</p>';
  if (noAge) {
    $('#c-age').innerHTML = notice;
    $('#c-agegender').innerHTML = notice;
    $('#l-agegender').innerHTML = '';
    $('#t-segments').innerHTML = '';
    return;
  }
  C.columns($('#c-age'), {
    data: AGES.map((k) => ({
      label: AGE_TICK[k], value: a[k], color: COLOR.age[k], key: k,
      hint: 'Visitors',
    })),
    height: 190,
    onClick: plain ? null : (d) => {
      const set = state.filters.age;
      set.has(d.key) ? set.delete(d.key) : set.add(d.key);
      syncChips(); renderAll();
    },
  });
  setTable('c-age', ['Age', 'Visitors', 'Share'],
    AGES.map((k) => [AGE_TICK[k], a[k], fmt.pct((a[k] / (rows.length || 1)) * 100)]));

  C.groupedColumns($('#c-agegender'), {
    categories: AGES.map((k) => AGE_TICK[k]),
    series: GENDERS.map((g) => ({
      name: GENDER_PLURAL[g], color: COLOR[g],
      values: AGES.map((k) => rows.filter((r) => r.age === k && r.gender === g).length),
    })),
    height: 190,
  });
  C.legend($('#l-agegender'), GENDERS.map((g) => ({ label: GENDER_PLURAL[g], color: COLOR[g] })));
  setTable('c-agegender', ['Age', 'Women', 'Men'],
    AGES.map((k) => [AGE_TICK[k],
      rows.filter((r) => r.age === k && r.gender === 'female').length,
      rows.filter((r) => r.age === k && r.gender === 'male').length]));

  const segRows = [];
  GENDERS.forEach((g) => AGES.forEach((k) => {
    const seg = rows.filter((r) => r.gender === g && r.age === k);
    if (!seg.length) return;
    const d = seg.map((r) => r.dwell_sec);
    segRows.push({
      g, k, n: seg.length,
      share: (seg.length / rows.length) * 100,
      avg: mean(d), med: median(d), max: Math.max(...d),
      gc: mean(seg.map((r) => r.gender_conf)),
      ac: mean(seg.map((r) => r.age_conf)),
      dwellShare: seg.reduce((s, r) => s + r.dwell_sec, 0),
    });
  }));
  segRows.sort((x, y) => y.n - x.n);
  const totalDwell = segRows.reduce((s, r) => s + r.dwellShare, 0) || 1;
  $('#t-segments').innerHTML =
    `<thead><tr><th>Group</th><th class="num">Visitors</th><th class="num">Share</th>
      <th class="num">Average stay</th><th class="num">Median</th><th class="num">Longest</th>
      <th class="num">Share of all time</th><th class="num">Gender conf.</th><th class="num">Age conf.</th></tr></thead>
     <tbody>${segRows.map((r) => `<tr>
       <td><span class="pill"><i class="swatch" style="background:${COLOR[r.g]}"></i>${GENDER_PLURAL[r.g]}, ${AGE_TICK[r.k]}</span></td>
       <td class="num">${r.n}</td>
       <td class="num"><div class="bar-cell"><i style="width:${Math.max(2, r.share * 1.4)}px;background:${COLOR[r.g]}"></i>${fmt.pct(r.share)}</div></td>
       <td class="num">${fmt.num(r.avg, 1)}s</td><td class="num">${fmt.num(r.med, 1)}s</td>
       <td class="num">${fmt.dur(r.max)}</td>
       <td class="num">${fmt.pct((r.dwellShare / totalDwell) * 100)}</td>
       <td class="num">${fmt.fixed(r.gc)}</td><td class="num">${fmt.fixed(r.ac)}</td></tr>`).join('')
     || '<tr><td colspan="9" class="empty">No visitors match.</td></tr>'}</tbody>`;
}

function renderEngagement(doc, rows) {
  const dwells = rows.map((r) => r.dwell_sec);
  const hi = Math.max(10, ...dwells);
  const h = histogram(dwells, 12, 0, hi);
  C.columns($('#c-dwellhist'), {
    data: h.counts.map((c, i) => ({
      label: i % 3 === 0 ? `${Math.round(h.edges[i])}s` : '',
      value: c, color: COLOR.one, hint: 'Visitors',
    })),
    height: 200,
  });
  setTable('c-dwellhist', ['Range', 'Visitors'],
    h.counts.map((c, i) => [`${Math.round(h.edges[i])} to ${Math.round(h.edges[i + 1])}s`, c]));

  const t = countBy(rows, 'tier', TIERS);
  C.hbars($('#c-tiers'), {
    data: TIERS.map((k) => ({
      label: k, value: t[k], color: COLOR.tier[k],
      rows: [{ label: 'Visitors', value: fmt.int(t[k]), color: COLOR.tier[k] },
        { label: 'Share', value: fmt.pct((t[k] / (rows.length || 1)) * 100) }],
    })),
    labelW: 78, rowH: 30,
  });
  setTable('c-tiers', ['Tier', 'Visitors', 'Share'],
    TIERS.map((k) => [k, t[k], fmt.pct((t[k] / (rows.length || 1)) * 100)]));

  C.columns($('#c-dwellage'), {
    data: AGES.map((k) => ({
      label: AGE_TICK[k],
      value: mean(rows.filter((r) => r.age === k).map((r) => r.dwell_sec)),
      color: COLOR.age[k], hint: 'Average stay',
    })),
    height: 200, format: (v) => fmt.num(v, 0), unit: 's',
  });
  setTable('c-dwellage', ['Age', 'Average stay', 'Visitors'],
    AGES.map((k) => {
      const seg = rows.filter((r) => r.age === k);
      return [AGE_TICK[k], `${fmt.num(mean(seg.map((r) => r.dwell_sec)), 1)}s`, seg.length];
    }));

  const segs = [];
  GENDERS.forEach((g) => AGES.forEach((k) => {
    const seg = rows.filter((r) => r.gender === g && r.age === k);
    if (seg.length) {
      segs.push({
        label: `${GENDER_PLURAL[g]}, ${AGE_TICK[k]}`, value: mean(seg.map((r) => r.dwell_sec)), color: COLOR[g],
        rows: [{ label: 'Average stay', value: `${fmt.num(mean(seg.map((r) => r.dwell_sec)), 1)}s`, color: COLOR[g] },
          { label: 'Visitors', value: seg.length }],
      });
    }
  }));
  segs.sort((x, y) => y.value - x.value);
  if (segs.length) {
    C.hbars($('#c-dwellseg'), { data: segs, labelW: 116, rowH: 30, format: (v) => fmt.num(v, 1), unit: 's' });
  } else {
    $('#c-dwellseg').innerHTML = '<p class="empty">No visitors match.</p>';
  }

  C.stackedRows($('#c-tiergender'), {
    categories: GENDERS.map((g) => GENDER_PLURAL[g]),
    rowH: 48,
    series: TIERS.map((k) => ({
      name: k, color: COLOR.tier[k],
      values: GENDERS.map((g) => rows.filter((r) => r.gender === g && r.tier === k).length),
    })),
  });
  C.legend($('#l-tiergender'), TIERS.map((k) => ({ label: k, color: COLOR.tier[k] })));
}

function renderFlow(doc, rows) {
  renderFlatMap(doc, rows);

  const zones = ['Left', 'Center', 'Right'];
  $('#t-zones').innerHTML =
    `<thead><tr><th>Zone</th><th class="num">Visitors</th><th class="num">Share</th>
      <th class="num">Average stay</th><th class="num">Women / men</th></tr></thead>
     <tbody>${zones.map((z) => {
      const seg = rows.filter((r) => r.zone === z);
      const f = seg.filter((r) => r.gender === 'female').length;
      return `<tr><td>${z}</td><td class="num">${seg.length}</td>
        <td class="num">${fmt.pct((seg.length / (rows.length || 1)) * 100)}</td>
        <td class="num">${fmt.num(mean(seg.map((r) => r.dwell_sec)), 1)}s</td>
        <td class="num">${f} / ${seg.length - f}</td></tr>`;
    }).join('')}</tbody>`;

  C.stackedRows($('#c-zoneage'), {
    categories: zones,
    rowH: 42, labelW: 62,
    series: AGES.map((k) => ({
      name: AGE_TICK[k], color: COLOR.age[k],
      values: zones.map((z) => rows.filter((r) => r.zone === z && r.age === k).length),
    })),
  });
  C.legend($('#l-zoneage'), AGES.map((k) => ({ label: AGE_TICK[k], color: COLOR.age[k] })));

  const sides = ['left', 'right', 'top', 'bottom', 'interior'];
  const sideName = { interior: 'Already inside' };
  C.hbars($('#c-entry'), {
    data: sides.map((s) => ({ label: sideName[s] || cap(s), value: rows.filter((r) => r.entry === s).length, hint: 'Visitors' })),
    labelW: 104, rowH: 30,
  });
  const flows = ['entered', 'exited', 'passed through', 'in view'];
  const flowName = { 'in view': 'Stayed in view' };
  C.hbars($('#c-flow'), {
    data: flows.map((s) => ({ label: flowName[s] || cap(s), value: rows.filter((r) => r.flow === s).length, hint: 'Visitors' })),
    labelW: 112, rowH: 30,
  });
}

/* The flat alternative to the 3D floor, and the fallback without WebGL. */
function renderFlatMap(doc, rows) {
  const box = $('#c-heat');
  if (box.hidden) return;
  let inner = $('#c-heat-inner');
  if (!inner) {
    inner = document.createElement('div');
    inner.id = 'c-heat-inner';
    box.appendChild(inner);
    new ResizeObserver(() => state.doc && renderFlatMap(state.doc, filtered())).observe(box);
  }
  // In the Floor tab the legend sits above the scrubber; keep the map clear of it.
  const full = $('#stage').dataset.mode === 'full';
  box.style.bottom = full ? `${78 + $('#stage-legend').offsetHeight + 14}px` : '';
  const aspect = doc.meta.width / doc.meta.height;
  inner.style.width = `${Math.max(0, Math.min(box.clientWidth, box.clientHeight * aspect))}px`;
  C.heatgrid(inner, {
    ...heatFrom(rows, doc),
    aspect,
    heat: $('#chk-heat').checked,
    // Coloured like the 3D paths, by whatever Gender / Age / Stay is selected.
    paths: $('#chk-paths').checked
      ? rows.slice(0, 160).map((r) => ({
        points: r.path.map(([, x, y]) => [x / doc.meta.width, y / doc.meta.height]),
        color: colorOf(r),
      }))
      : null,
  });
}

function renderModel(doc, rows) {
  const gh = histogram(rows.map((r) => r.gender_conf), 10, 0, 1);
  const ah = histogram(rows.map((r) => r.age_conf), 10, 0, 1);
  const bars = (h) => h.counts.map((c, i) => ({
    label: i % 2 === 0 ? fmt.num(h.edges[i], 1) : '', value: c, color: COLOR.one, hint: 'Visitors',
  }));
  C.columns($('#c-gconf'), { data: bars(gh), height: 190 });
  C.columns($('#c-aconf'), { data: bars(ah), height: 190 });

  const thresh = doc.config.low_conf_threshold ?? 0.6;
  const low = rows.filter((r) => r.gender_conf < thresh).length;
  $('#m-lowconf').textContent = fmt.pct((low / (rows.length || 1)) * 100);
  $('#m-lowconf-note').textContent =
    `${low} of ${plural(rows.length, 'visitor')} under ${fmt.num(thresh, 2)} gender confidence. `
    + `${plural(doc.kpis.tracks_dropped, 'short track')} discarded`
    + (doc.kpis.tracks_rejoined ? `, and ${plural(doc.kpis.tracks_rejoined, 'interrupted track')} joined back up.` : '.')
    + (doc.kpis.hunches_dropped != null
      ? ` ${plural(doc.kpis.hunches_dropped, 'faint detection')} checked and ruled out.` : '');

  const M = doc.model || {};
  const none = 'Not used';
  $('#m-models').innerHTML = [
    ['Gender', M.gender?.family ?? none],
    ['Checkpoint', M.gender?.source ?? none],
    ['Age', M.age?.family ?? none],
    ['Reported mA', M.age?.reported_avg_mA ? fmt.num(M.age.reported_avg_mA, 3) : none],
    ['Crops classified', fmt.int(doc.meta.crops_classified || 0)],
  ].map(([k, v]) => `<dt>${k}</dt><dd>${esc(v)}</dd>`).join('');

  const c = doc.config;
  $('#m-config').innerHTML = [
    ['Detector', c.detector],
    ['Detection confidence', c.track_conf && c.track_conf < c.det_conf
      ? `${c.det_conf} to count a person, ${c.track_conf} to pick one back up, ${c.weak_conf ?? c.track_conf} to keep following one`
      : c.det_conf],
    ['Second detector', c.verifier || 'Not used'],
    ['Frame stride', `every ${c.frame_stride} frames`],
    ['Bin size', `${c.bin_sec}s`],
    ['Device', doc.meta.device],
    ['Throughput', `${fmt.num(doc.meta.processing_fps || 0, 1)} fps`],
    ['Generated', (doc.generated_at || '').replace('T', ' ').slice(0, 19)],
  ].map(([k, v]) => `<dt>${k}</dt><dd>${esc(v)}</dd>`).join('');
}

/* ---------------------------------------------------------- insights */
function renderInsights(doc) {
  const host = $('#insight-list');
  const items = doc.insights || [];
  host.innerHTML = items.length
    ? items.map((i) => `<li><span class="k">${esc(i.label || '')}</span><span class="v">${esc(i.text)}</span></li>`).join('')
    : '<li class="v">This run was made before insights existed. Process the video again to get them.</li>';
}

/* Period analysis (daily / weekly / monthly) across every processed run.
   Needs the server, because it aggregates all runs - and needs more than one
   day, which the UI states rather than drawing a line through one point. */
const MONTH_NAMES = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
const METRICS = [
  { id: 'per_hour', label: 'Visitors per hour', fmt: (v) => fmt.num(v, 0), unit: '' },
  { id: 'arrivals', label: 'Total visitors', fmt: (v) => fmt.num(v, 0), unit: '' },
  { id: 'avg_dwell_sec', label: 'Average visit', fmt: (v) => fmt.num(v, 0), unit: 's' },
  { id: 'engaged_pct', label: 'Stayed 30s or more', fmt: (v) => fmt.num(v, 0), unit: '%' },
  { id: 'female_pct', label: 'Share of women', fmt: (v) => fmt.num(v, 0), unit: '%' },
];
const PERIOD_KEYS = { day: 'by_day', weekday: 'by_weekday', month: 'by_month' };
const periodState = { period: 'weekday', metric: 'per_hour', data: null };

const periodKey = (period, k) => (period === 'month' ? MONTH_NAMES[k - 1]
  : period === 'day' ? String(k).slice(5) : k);

async function renderTrends() {
  const empty = (el, msg) => { el.innerHTML = `<p class="empty">${msg}</p>`; };
  let t;
  try {
    const withDemo = $('#chk-demo-trends')?.checked ? 'true' : 'false';
    const res = await fetch(`/api/trends?include_demo=${withDemo}`);
    if (!res.ok) throw new Error();
    t = await res.json();
  } catch {
    $('#trend-coverage').textContent = 'Comparing recordings needs the server running.';
    ['#c-trendline', '#c-period', '#c-hour', '#c-mix'].forEach((id) => empty($(id),
      'Start the dashboard with <code>python -m app.server</code> to compare recordings.'));
    return;
  }
  periodState.data = t;
  const c = t.coverage;
  const src = t.sources;

  // Say plainly what these numbers are made of.
  const parts = [];
  if (c.runs) {
    parts.push(plural(c.runs, 'recording'));
    parts.push(`${fmt.num(c.observed_hours, 1)} hours of footage`);
    parts.push(plural(c.days, 'day'));
  }
  $('#trend-coverage').innerHTML = c.runs === 0
    ? `No recordings yet. Process a few videos and this fills in.${src.sample_runs ? ` ${plural(src.sample_runs, 'sample')} left out.` : ''}`
    : `Built from ${parts.join(', ')}.`
      + (src.samples_included
        ? ` <span class="warn">Includes ${plural(src.sample_runs, 'generated sample')}.</span>`
        : (src.sample_runs ? ` ${plural(src.sample_runs, 'sample')} left out.` : ''))
      + (c.mixed_cameras ? ' <b>Several cameras are mixed together.</b>' : '');

  const head = $('#trend-headline');
  if (t.trend && c.enough_for_trend) {
    const d = t.trend.change_pct;
    const dir = d > 0 ? 'up' : 'down';
    head.innerHTML =
      `<div class="headline"><span class="big">${fmt.num(t.trend.latest_per_hour, 0)}</span>`
      + `<span class="unit">visitors per hour on ${t.trend.latest_date}</span>`
      + (d == null ? '' : `<span class="delta ${dir}">${d > 0 ? 'Up' : 'Down'} ${fmt.pct(Math.abs(d), 0)}</span>`) + '</div>'
      + `<p class="sub">Against ${fmt.num(t.trend.previous_avg_per_hour, 0)} per hour averaged over the earlier days.</p>`;
    C.lines($('#c-trendline'), {
      x: t.by_day.map((_, i) => i),
      series: [{ name: 'Visitors per hour', color: COLOR.one, values: t.by_day.map((d2) => d2.per_hour) }],
      area: true, height: 150,
      xFormat: (i) => (t.by_day[i]?.key || '').slice(5),
      format: (v) => fmt.num(v, 0),
    });
  } else {
    const need = Math.max(1, c.min_days_for_trend - c.usable_days);
    head.innerHTML = `<p class="empty">Record ${plural(need, 'more day', 'more days')} to compare footfall over time.</p>`;
    $('#c-trendline').innerHTML = '';
  }

  renderPeriod();

  if (t.by_hour.length) {
    C.columns($('#c-hour'), {
      data: t.by_hour.map((h) => ({
        label: `${h.key}:00`, value: h.per_hour, color: COLOR.one,
        hint: `Visitors per hour (${fmt.num(h.observed_hours, 1)}h watched)`,
      })),
      height: 200, format: (v) => fmt.num(v, 0),
    });
  } else {
    empty($('#c-hour'), 'No dated recordings yet.');
  }

  const mix = t[PERIOD_KEYS[periodState.period]];
  if (mix.length) {
    C.stackedRows($('#c-mix'), {
      categories: mix.map((r) => periodKey(periodState.period, r.key)),
      rowH: Math.max(32, Math.min(46, 240 / mix.length)),
      labelW: 62,
      series: [
        { name: 'Women', color: COLOR.female, values: mix.map((r) => r.gender.female) },
        { name: 'Men', color: COLOR.male, values: mix.map((r) => r.gender.male) },
      ],
    });
    C.legend($('#l-mix'), GENDERS.map((g) => ({ label: GENDER_PLURAL[g], color: COLOR[g] })));
  } else {
    empty($('#c-mix'), 'No dated recordings yet.');
    $('#l-mix').innerHTML = '';
  }

  $('#t-runs').innerHTML =
    `<thead><tr><th>Recording</th><th>Source</th><th>When</th><th>Shop and camera</th>
      <th class="num">Visitors</th><th class="num">Per hour</th><th class="num">Average visit</th>
      <th class="num">Stayed 30s+</th><th class="num">Women</th></tr></thead>
     <tbody>${t.runs.map((r) => `<tr>
       <td>${esc(r.source)}</td>
       <td>${r.demo ? '<span class="badge">Sample</span>' : 'Footage'}</td>
       <td>${esc((r.captured_at || 'Unknown').replace('T', ' '))}</td>
       <td>${esc(r.site)}, ${esc(r.camera)}</td>
       <td class="num">${r.visitors}</td>
       <td class="num">${r.comparable ? fmt.num(r.per_hour, 0) : 'Too short'}</td>
       <td class="num">${fmt.num(r.avg_dwell_sec, 1)}s</td>
       <td class="num">${fmt.pct(r.engaged_pct, 0)}</td>
       <td class="num">${fmt.pct(r.female_pct, 0)}</td></tr>`).join('')
     || '<tr><td colspan="9" class="empty">No recordings yet.</td></tr>'}</tbody>`;
}

function renderPeriod() {
  const t = periodState.data;
  if (!t) return;
  const rows = t[PERIOD_KEYS[periodState.period]];
  const metric = METRICS.find((m) => m.id === periodState.metric);

  $('#seg-metric').innerHTML = METRICS.map((m) =>
    `<button class="chip" data-metric="${m.id}" aria-pressed="${m.id === periodState.metric}">${m.label}</button>`).join('');
  $$('#seg-metric .chip').forEach((b) => {
    b.onclick = () => { periodState.metric = b.dataset.metric; renderPeriod(); };
  });
  $$('#seg-period button').forEach((b) => {
    b.setAttribute('aria-pressed', String(b.dataset.period === periodState.period));
    b.onclick = () => { periodState.period = b.dataset.period; renderTrends(); };
  });
  $('#period-sub').textContent = {
    day: 'One bar for each day you recorded. Rates are per hour of footage watched, so a short clip cannot outweigh a long one.',
    weekday: 'Every recording grouped by day of the week, Mondays with Mondays.',
    month: 'Every recording grouped by calendar month.',
  }[periodState.period];

  if (!rows.length) {
    $('#c-period').innerHTML = '<p class="empty">No dated recordings yet.</p>';
    $('#t-period').innerHTML = '';
    return;
  }
  C.columns($('#c-period'), {
    data: rows.map((r) => ({
      label: periodKey(periodState.period, r.key),
      value: r[metric.id], color: COLOR.one,
      hint: metric.label,
    })),
    height: 220, format: metric.fmt, unit: metric.unit,
  });

  const head = { day: 'Date', weekday: 'Day', month: 'Month' }[periodState.period];
  $('#t-period').innerHTML =
    `<thead><tr><th>${head}</th>
      <th class="num">Hours watched</th><th class="num">Visitors</th><th class="num">Per hour</th>
      <th class="num">Average visit</th><th class="num">Stayed 30s+</th><th class="num">Women</th>
      <th class="num">Biggest age group</th></tr></thead>
     <tbody>${rows.map((r) => {
      const ages = Object.entries(r.age);
      const top = ages.reduce((a, b) => (b[1] > a[1] ? b : a), ages[0]);
      return `<tr><td>${periodKey(periodState.period, r.key)}</td>
        <td class="num">${fmt.num(r.observed_hours, 1)}</td>
        <td class="num">${r.arrivals}</td>
        <td class="num">${fmt.num(r.per_hour, 0)}</td>
        <td class="num">${fmt.num(r.avg_dwell_sec, 0)}s</td>
        <td class="num">${fmt.pct(r.engaged_pct, 0)}</td>
        <td class="num">${fmt.pct(r.female_pct, 0)}</td>
        <td class="num">${top && top[1] ? `${AGE_TICK[top[0]]} (${top[1]})` : 'None'}</td></tr>`;
    }).join('')}</tbody>`;
}

/* ------------------------------------------------------------ log/CSV */
let sortKey = 'first_sec', sortDir = 1;
function renderLog(rows) {
  $('#log-count').textContent = plural(rows.length, 'tracked visitor');
  const cols = [
    ['id', 'No.'], ['first_sec', 'First seen'], ['dwell_sec', 'Stayed'], ['tier', 'Tier'],
    ['gender', 'Gender'], ['gender_conf', 'Conf.'], ['age', 'Age'], ['age_conf', 'Conf.'],
    ['zone', 'Zone'], ['flow', 'Path'], ['frames', 'Frames'], ['det_conf', 'Detection'],
    ['avg_size_pct', 'Frame area'],
  ];
  const sorted = [...rows].sort((a, b) => {
    const x = a[sortKey], y = b[sortKey];
    return (typeof x === 'string' ? String(x).localeCompare(String(y)) : x - y) * sortDir;
  });
  $('#t-tracks').innerHTML =
    `<thead><tr>${cols.map(([k, l]) =>
      `<th data-k="${k}"${sortKey === k ? ` aria-sort="${sortDir > 0 ? 'ascending' : 'descending'}"` : ''}>${l}${sortKey === k ? (sortDir > 0 ? ' ↑' : ' ↓') : ''}</th>`).join('')}</tr></thead>
     <tbody>${sorted.slice(0, 500).map((r) => `<tr>
       <td>${r.id}</td><td>${fmt.clock(r.first_sec)}</td><td class="num">${fmt.num(r.dwell_sec, 1)}s</td>
       <td>${r.tier}</td>
       <td><span class="pill"><i class="swatch" style="background:${COLOR[r.gender] || COLOR.none}"></i>${cap(r.gender)}</span></td>
       <td class="num">${fmt.fixed(r.gender_conf)}</td>
       <td><span class="pill"><i class="swatch" style="background:${COLOR.age[r.age] || COLOR.none}"></i>${AGE_TICK[r.age] || 'Unknown'}</span></td>
       <td class="num">${fmt.fixed(r.age_conf)}</td>
       <td>${r.zone}</td><td>${cap(r.flow)}</td><td class="num">${r.frames}</td>
       <td class="num">${fmt.fixed(r.det_conf)}</td><td class="num">${fmt.pct(r.avg_size_pct)}</td></tr>`).join('')
     || '<tr><td colspan="13" class="empty">No visitors match.</td></tr>'}</tbody>`;
  $$('#t-tracks th').forEach((th) => {
    th.onclick = () => {
      const k = th.dataset.k;
      sortDir = sortKey === k ? -sortDir : 1;
      sortKey = k;
      renderLog(filtered());
    };
  });
}

function exportCsv() {
  const rows = filtered();
  const cols = ['id', 'gender', 'gender_conf', 'age', 'age_conf', 'first_sec', 'last_sec',
    'dwell_sec', 'tier', 'zone', 'entry', 'exit', 'flow', 'frames', 'classified_frames',
    'det_conf', 'avg_size_pct'];
  const csv = [cols.join(','), ...rows.map((r) => cols.map((c) => r[c]).join(','))].join('\n');
  const url = URL.createObjectURL(new Blob([csv], { type: 'text/csv' }));
  const a = document.createElement('a');
  a.href = url;
  a.download = `${state.runId}-visitors.csv`;
  a.click();
  URL.revokeObjectURL(url);
}

/* --------------------------------------------------------------- video */
function setupVideo(id, doc) {
  const wrap = $('#video-wrap');
  const has = doc.meta.has_video;
  wrap.innerHTML = '';
  state.video = null;
  if (!has) {
    wrap.innerHTML = `<div class="video-empty"><div>
        <p><b>This run has no annotated video.</b></p>
        <p>Run the pipeline again without <code>--no-video</code>, or use Process video. The other tabs work without it.</p>
      </div></div>`;
  } else {
    const v = document.createElement('video');
    v.controls = true;
    v.preload = 'metadata';
    v.src = `/api/runs/${encodeURIComponent(id)}/video.mp4`;
    v.poster = `/api/runs/${encodeURIComponent(id)}/poster.jpg`;
    wrap.appendChild(v);
    state.video = v;
    if (doc.meta.video_codec && doc.meta.video_codec !== 'avc1') {
      wrap.insertAdjacentHTML('beforeend',
        `<div class="badge"><i class="dot"></i>Saved as ${esc(doc.meta.video_codec)}. Install ffmpeg and process again for a file browsers can play.</div>`);
    }
    let last = -1;
    v.addEventListener('timeupdate', () => {
      if (Math.abs(v.currentTime - last) < 0.2) return;
      last = v.currentTime;
      setPlayhead(v.currentTime, { fromVideo: true });
    });
  }
  C.legend($('#l-video'), [
    { label: 'Women', color: COLOR.female },
    { label: 'Men', color: COLOR.male },
  ]);
  $('#l-video').insertAdjacentHTML('beforeend',
    '<span class="item">Each box shows the visitor number, gender with confidence, and age group.</span>');
}

/* One clock for the whole page: the video, the 3D floor, the occupancy chart
   and the "at this moment" panel all read state.playhead. */
function setPlayhead(t, { fromVideo = false, fromStage = false } = {}) {
  const dur = state.doc?.meta.duration_sec || 0;
  state.playhead = Math.max(0, Math.min(dur, t));
  if (!fromStage) floor?.setTime(state.playhead);
  if (!fromVideo && state.video && state.video.paused && Number.isFinite(state.video.duration)) {
    state.video.currentTime = Math.min(state.playhead, state.video.duration);
  }
  renderClock();
  // The panel and the chart marker follow at ten frames a second, not sixty.
  if (!momentTimer) {
    momentTimer = setTimeout(() => { momentTimer = 0; renderMoment(); }, 100);
  }
}
let momentTimer = 0;
function seek(t) {
  floor?.pause();
  setPlayhead(t);
}

function renderPlayhead() {
  renderClock();
  renderMoment();
}

/* What was on the floor at the playhead. */
function renderMoment() {
  const doc = state.doc;
  if (!doc) return;
  const t = state.playhead, dur = doc.meta.duration_sec || 0;
  const rows = filtered();
  const now = rows.filter((r) => r.first_sec <= t && r.last_sec >= t);
  $('#video-time').textContent = `${fmt.clock(t)} of ${fmt.clock(dur)}`
    + (state.video ? '' : '. Drag the time bar on the Floor tab to move through the recording.');
  $('#v-count').textContent = now.length;
  $('#v-entries').textContent = rows.filter((r) => r.first_sec <= t).length;
  C.splitBar($('#v-split'), [
    { label: 'Women', value: now.filter((r) => r.gender === 'female').length, color: COLOR.female },
    { label: 'Men', value: now.filter((r) => r.gender === 'male').length, color: COLOR.male },
  ], (v) => `${fmt.num(v, 0)}%`);
  C.hbars($('#v-ages'), {
    data: AGES.map((k) => ({ label: AGE_TICK[k], value: now.filter((r) => r.age === k).length, color: COLOR.age[k], hint: 'In view' })),
    labelW: 52, rowH: 26,
  });
  if (state.view === 'overview') $('#c-occupancy').__render?.();
}

/* Stage clock and scrubber: cheap enough to follow every frame of a replay. */
function renderClock() {
  const doc = state.doc;
  if (!doc) return;
  const t = state.playhead, dur = doc.meta.duration_sec || 0;
  const frac = dur ? t / dur : 1;
  const scrub = $('#stage-scrub');
  if (document.activeElement !== scrub || !scrub.matches(':active')) scrub.value = Math.round(frac * 1000);
  scrub.setAttribute('aria-valuetext', `${fmt.clock(t)} of ${fmt.clock(dur)}`);
  $('#stage-clock').textContent = `${fmt.clock(t)} / ${fmt.clock(dur)}`;
  const bars = $$('#scrub-bars i');
  bars.forEach((b, i) => b.classList.toggle('past', (i + 0.5) / bars.length <= frac));
}

/* ------------------------------------------------------------ 3D stage */
const colorOf = (r) => ({
  gender: COLOR[r.gender] || COLOR.none,
  age: COLOR.age[r.age] || COLOR.none,
  tier: COLOR.tier[r.tier] || COLOR.none,
}[state.colorBy]);

const describe = (r) => [
  { label: 'Gender', value: cap(r.gender), color: COLOR[r.gender] },
  { label: 'Age', value: AGE_SHORT[r.age] || 'Not measured' },
  { label: 'Stayed', value: fmt.dur(r.dwell_sec) },
  { label: 'Engagement', value: r.tier, color: COLOR.tier[r.tier] },
];

/* The scrubber's track is a histogram of arrivals, so the busy stretches are
   visible before anything is played. */
function buildScrubBars(doc) {
  const dur = doc.meta.duration_sec || 1;
  const counts = Array(56).fill(0);
  doc.tracks.forEach((r) => { counts[Math.min(counts.length - 1, Math.floor((r.first_sec / dur) * counts.length))]++; });
  const max = Math.max(1, ...counts);
  $('#scrub-bars').innerHTML = counts.map((n) => `<i style="height:${Math.round((n / max) * 100)}%"></i>`).join('');
}

function renderStage(doc, rows) {
  const run = state.runs.find((r) => r.id === state.runId);
  floor?.setData({
    doc, rows, colorOf,
    posterUrl: run?.has_poster ? `/api/runs/${encodeURIComponent(state.runId)}/poster.jpg` : null,
    videoUrl: run?.has_video ? `/api/runs/${encodeURIComponent(state.runId)}/video.mp4` : null,
  });
  floor?.setTime(state.playhead);

  const keys = {
    gender: GENDERS.map((g) => [GENDER_PLURAL[g], COLOR[g]]),
    age: AGES.filter((k) => rows.some((r) => r.age === k)).map((k) => [AGE_TICK[k], COLOR.age[k]]),
    tier: TIERS.map((k) => [k, COLOR.tier[k]]),
  }[state.colorBy];
  const ramp = [1, 2, 3, 4, 5, 6].map((i) => `<i style="background:var(--seq-${i})"></i>`).join('');
  $('#stage-legend').innerHTML =
    keys.map(([label, c]) => `<span class="item"><i class="key" style="background:${c}"></i>${label}</span>`).join('')
    + ($('#chk-heat').checked ? `<span class="item">Less time <span class="ramp">${ramp}</span> More</span>` : '');
}

/* The stage lives in the overview hero and moves into the Floor tab. */
function placeStage(view) {
  const el = $('#stage');
  const full = view === 'floor';
  const slot = full ? $('#slot-floor') : $('#slot-overview');
  if (el.parentElement !== slot) slot.appendChild(el);
  el.dataset.mode = full ? 'full' : 'hero';
  floor?.setMode(el.dataset.mode);
  // The hero is always the 3D picture; Top and Flat belong to the Floor tab.
  if (!full && floor && el.dataset.layout !== '3d') setLayout('3d');
  if (state.doc) renderFlatMap(state.doc, filtered());
}

function setLayout(layout) {
  const el = $('#stage');
  if (!floor) layout = 'flat';
  el.dataset.layout = layout;
  $('#c-heat').hidden = layout !== 'flat';
  $$('#stage-tools [data-layout]').forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.layout === layout)));
  if (layout !== 'flat') floor.setLayout(layout);
  else if (state.doc) renderFlatMap(state.doc, filtered());
}

function wireStage() {
  const el = $('#stage');
  if (hasWebGL()) {
    try {
      floor = createFloor($('#stage-canvas'), {
        showTip: C.showTip, hideTip: C.hideTip, fmt, describe,
        // The toolbar and legend wrap on narrow screens, so measure them.
        insets: () => ({
          top: $('#stage-tools').offsetHeight + 30,
          bottom: 78 + $('#stage-legend').offsetHeight + 14,
        }),
        onTime: (t) => setPlayhead(t, { fromStage: true }),
        onPlayState: (on) => {
          el.classList.toggle('playing', on);
          $('#stage-play').setAttribute('aria-label', on ? 'Pause the replay' : 'Replay the recording');
        },
      });
    } catch (err) {
      console.warn('3D floor unavailable:', err);
      floor = null;
    }
  }
  if (!floor) {
    // No WebGL: the flat map carries the same data.
    $('#stage-fallback').hidden = false;
    $('#stage-hint').hidden = true;
    $$('#stage-tools [data-layout="3d"], #stage-tools [data-layout="top"]').forEach((b) => { b.hidden = true; });
    $('#stage-colorby').hidden = true;
    $('#chk-heat').closest('.switch').hidden = true;
    setLayout('flat');
  }

  $('#stage-play').onclick = () => {
    if (floor) { floor.toggle(); return; }
    setPlayhead(state.playhead >= (state.doc?.meta.duration_sec || 0) ? 0 : state.doc.meta.duration_sec);
  };
  $('#stage-scrub').addEventListener('input', (e) => {
    seek((+e.target.value / 1000) * (state.doc?.meta.duration_sec || 0));
  });
  $$('#stage-tools [data-layout]').forEach((b) => { b.onclick = () => setLayout(b.dataset.layout); });
  $$('#stage-tools [data-colorby]').forEach((b) => {
    b.onclick = () => {
      state.colorBy = b.dataset.colorby;
      $$('#stage-tools [data-colorby]').forEach((x) => x.setAttribute('aria-pressed', String(x === b)));
      if (state.doc) { renderStage(state.doc, filtered()); renderFlatMap(state.doc, filtered()); }
    };
  });
  $('#chk-heat').onchange = (e) => {
    floor?.setLayers({ heat: e.target.checked });
    if (state.doc) { renderStage(state.doc, filtered()); renderFlatMap(state.doc, filtered()); }
  };
  $('#chk-paths').onchange = (e) => {
    floor?.setLayers({ paths: e.target.checked });
    if (state.doc) renderFlatMap(state.doc, filtered());
  };
  $('#chk-frame').onchange = (e) => floor?.setLayers({ frame: e.target.checked });
  $('#open-floor').onclick = () => setView('floor');
  el.addEventListener('pointerdown', () => $('#stage-hint').classList.add('gone'), { once: true });
}

/* ------------------------------------------------------------- skyline */
function skylineData(rows) {
  const hasAge = rows.some((r) => r.age);
  const groups = hasAge ? AGES.map((k) => ({ key: k, label: AGE_TICK[k] })) : [{ key: null, label: 'All ages' }];
  const series = GENDERS.map((g) => ({ key: g, label: GENDER_PLURAL[g], color: COLOR[g] }));
  const byDwell = state.skyMetric === 'dwell';
  const cell = (g, k) => rows.filter((r) => r.gender === g && (k === null || r.age === k));
  const values = [], text = [], tips = [];
  series.forEach((s) => {
    const segs = groups.map((gr) => cell(s.key, gr.key));
    const stays = segs.map((seg) => mean(seg.map((r) => r.dwell_sec)));
    values.push(segs.map((seg, i) => (byDwell ? stays[i] : seg.length)));
    text.push(segs.map((seg, i) => (byDwell ? (seg.length ? `${fmt.num(stays[i], 0)}s` : '') : String(seg.length))));
    tips.push(segs.map((seg, i) => [
      { label: 'People', value: fmt.int(seg.length), color: s.color },
      { label: 'Share of everyone', value: fmt.pct((seg.length / (rows.length || 1)) * 100) },
      { label: 'Average stay', value: seg.length ? `${fmt.num(stays[i], 1)}s` : 'No one' },
    ]));
  });
  return { groups, series, values, text, tips };
}

function renderSkyline(rows) {
  const byDwell = state.skyMetric === 'dwell';
  $('#skyline-sub').textContent =
    `Each block is one group of visitors. Taller means ${byDwell ? 'they stayed longer on average' : 'more people'}.`
    + (simple() || !skyline ? '' : ' Click a block to filter the whole dashboard.');
  if (!skyline) {
    // Without WebGL the two flat charts beside this card carry the same split.
    if (!hasWebGL()) $('.skyline-card').hidden = true;
    return;
  }
  skyline.setData(skylineData(rows));
  C.legend($('#l-skyline'), GENDERS.map((g) => ({ label: GENDER_PLURAL[g], color: COLOR[g] })));
}

function ensureSkyline() {
  if (skyline || !hasWebGL()) return;
  try {
    skyline = createSkyline($('#c-skyline'), {
      showTip: C.showTip, hideTip: C.hideTip,
      onPick: (gender, age) => {
        if (simple()) return;                 // filters are a Detailed-view tool
        const f = state.filters;
        const same = f.gender.size === 1 && f.gender.has(gender)
          && (age === null || (f.age.size === 1 && f.age.has(age)));
        f.gender = new Set(same ? [] : [gender]);
        if (age !== null) f.age = new Set(same ? [] : [age]);
        syncChips();
        renderAll();
      },
    });
  } catch (err) {
    console.warn('3D skyline unavailable:', err);
    $('.skyline-card').hidden = true;
    return;
  }
  if (state.doc) renderSkyline(filtered());
}

/* ------------------------------------------------------------- ambient */
/* Paint this run's heat map onto the page background. Low resolution on
   purpose: CSS scales and blurs it into soft light behind the glass. */
function paintAmbient(doc, rows) {
  const cv = $('#ambient');
  const g = cv.getContext('2d');
  if (!g) return;
  const heat = heatFrom(rows, doc);
  cv.width = 96;
  cv.height = Math.round(96 / (doc.meta.width / doc.meta.height || 1.78));
  g.clearRect(0, 0, cv.width, cv.height);
  const stops = ['--seq-2', '--seq-3', '--seq-4', '--seq-5', '--seq-6'].map(cssVar);
  const cw = cv.width / heat.cols, ch = cv.height / heat.rows;
  heat.grid.forEach((row, r) => row.forEach((v, c) => {
    if (v < 0.03) return;
    const x = (c + 0.5) * cw, y = (r + 0.5) * ch, rad = cw * (1.4 + v * 2.4);
    const grad = g.createRadialGradient(x, y, 0, x, y, rad);
    grad.addColorStop(0, stops[Math.min(stops.length - 1, Math.floor(v * stops.length))]);
    grad.addColorStop(1, 'transparent');
    g.globalAlpha = 0.25 + v * 0.6;
    g.fillStyle = grad;
    g.fillRect(x - rad, y - rad, rad * 2, rad * 2);
  }));
}

/* ---------------------------------------------------------- processing */
function wireProcess() {
  const dlg = $('#dlg-process');
  const file = $('input[name="video"]', dlg);
  const drop = $('#drop');
  const showFile = () => {
    $('#drop-title').textContent = file.files[0]?.name || 'Choose a video, or drop one here';
  };
  file.addEventListener('change', showFile);
  ['dragenter', 'dragover'].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add('over'); }));
  ['dragleave', 'drop'].forEach((ev) => drop.addEventListener(ev, () => drop.classList.remove('over')));
  drop.addEventListener('drop', (e) => {
    e.preventDefault();
    if (e.dataTransfer?.files.length) { file.files = e.dataTransfer.files; showFile(); }
  });

  $('#btn-process').onclick = () => dlg.showModal();
  $('#proc-cancel').onclick = () => dlg.close();
  $('#form-process').addEventListener('submit', async (e) => {
    e.preventDefault();
    const form = new FormData(e.target);
    if (!form.get('gender_repo').trim() && !form.get('age_bundle').trim()) {
      $('#proc-status').textContent = 'Fill in at least one model. With both fields empty there is nothing to classify people with.';
      return;
    }
    $('#proc-status').textContent = 'Uploading the video';
    $('#proc-progress').hidden = false;
    $('#proc-go').disabled = true;
    try {
      const res = await fetch('/api/runs', { method: 'POST', body: form });
      if (!res.ok) throw new Error(await res.text());
      const { job } = await res.json();
      const poll = setInterval(async () => {
        const s = await (await fetch(`/api/jobs/${job}`)).json();
        const eta = s.eta_sec != null ? `, about ${C.fmt.dur(s.eta_sec)} left` : '';
        const seen = s.classified != null ? `, ${fmt.int(s.classified)} crops classified` : '';
        $('#proc-status').textContent = s.error
          ? `Processing failed. ${s.error}`
          : (s.stage === 'done' ? 'Processed.' : `${cap(s.stage)}, ${s.pct ?? 0}%${eta}${seen}`);
        $('#proc-progress').firstElementChild.style.width = `${s.pct ?? 0}%`;
        if (s.stage === 'done' || s.stage === 'failed') {
          clearInterval(poll);
          $('#proc-go').disabled = false;
          if (s.stage === 'done') { dlg.close(); loadRuns(s.run_id); }
        }
      }, 1200);
    } catch (err) {
      $('#proc-status').textContent = `The video could not be sent. ${err.message || ''} Check that the server is running with python -m app.server.`;
      $('#proc-go').disabled = false;
    }
  });
}

/* ----------------------------------------------------------- chrome */
function moveGlider(instant = false) {
  const tab = $('#tabs [aria-selected="true"]');
  const glider = $('#tab-glider');
  if (!tab || !tab.offsetWidth) return;
  glider.classList.toggle('instant', instant);
  glider.style.width = `${tab.offsetWidth}px`;
  glider.style.transform = `translateX(${tab.offsetLeft}px)`;
}

function setView(view, { remember = true } = {}) {
  if (!VIEWS.includes(view)) view = 'overview';
  if (view === 'log' && simple()) view = 'overview';
  const changed = view !== state.view;
  state.view = view;
  document.body.dataset.view = view;
  $$('.view').forEach((v) => { v.hidden = v.id !== `view-${view}`; });
  $$('#tabs [role="tab"]').forEach((b) => {
    const on = b.dataset.view === view;
    b.setAttribute('aria-selected', String(on));
    b.tabIndex = on ? 0 : -1;
  });
  moveGlider();
  placeStage(view);
  if (view === 'people') ensureSkyline();
  // Charts mounted while their tab was hidden had no width to draw into.
  $$(`#view-${view} [id^="c-"], #view-${view} .spark`).forEach((el) => el.__render?.());
  if (remember) history.replaceState(null, '', `#${view}`);
  if (changed) {
    window.scrollTo({ top: 0 });
    if (view !== 'overview' && view !== 'floor') floor?.pause();
    if (view !== 'video') state.video?.pause();
  }
}

function wireTabs() {
  const tabs = $$('#tabs [role="tab"]');
  const visible = () => tabs.filter((b) => !(b.hasAttribute('data-adv') && simple()));
  tabs.forEach((b) => { b.onclick = () => setView(b.dataset.view); });
  $('#tabs').addEventListener('keydown', (e) => {
    const list = visible();
    const i = list.indexOf(document.activeElement);
    if (i < 0) return;
    const next = { ArrowRight: i + 1, ArrowLeft: i - 1, Home: 0, End: list.length - 1 }[e.key];
    if (next === undefined) return;
    e.preventDefault();
    const b = list[(next + list.length) % list.length];
    b.focus();
    setView(b.dataset.view);
  });
  // 1 to 6 jump straight to a tab, unless the user is typing somewhere.
  document.addEventListener('keydown', (e) => {
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    if (e.target instanceof Element && e.target.closest('input, select, textarea, dialog')) return;
    const b = visible()[+e.key - 1];
    if (b) setView(b.dataset.view);
  });
  window.addEventListener('hashchange', () => setView(location.hash.slice(1), { remember: false }));
  window.addEventListener('resize', () => moveGlider(true));
  document.fonts?.ready.then(() => moveGlider(true));
}

function wireRunPicker() {
  const pop = $('#run-pop'), btn = $('#run-button');
  if (!('popover' in HTMLElement.prototype)) {
    // Older browsers: a plain toggled panel.
    pop.removeAttribute('popover');
    pop.hidden = true;
    pop.hidePopover = () => { pop.hidden = true; };
    btn.onclick = () => { pop.hidden = !pop.hidden; place(); };
  }
  function place() {
    const r = btn.getBoundingClientRect();
    const w = Math.min(380, window.innerWidth - 24);
    pop.style.left = `${Math.max(12, Math.min(r.left, window.innerWidth - w - 12))}px`;
    pop.style.top = `${r.bottom + 10}px`;
  }
  pop.addEventListener('beforetoggle', (e) => { if (e.newState === 'open') place(); });
}

function setMode(mode) {
  document.body.dataset.mode = mode;
  try { localStorage.setItem('ifas-mode', mode); } catch { /* storage blocked */ }
  $('#btn-simple').setAttribute('aria-pressed', String(mode === 'simple'));
  $('#btn-full').setAttribute('aria-pressed', String(mode !== 'simple'));
  // A hidden filter cannot be left silently applied.
  if (mode === 'simple') clearFilters();
  setView(state.view, { remember: false });
  moveGlider(true);
  if (state.doc) renderAll();
}

function wireMode() {
  $('#btn-simple').onclick = () => setMode('simple');
  $('#btn-full').onclick = () => setMode('full');
  let saved = null;
  try { saved = localStorage.getItem('ifas-mode'); } catch { /* storage blocked */ }
  setMode(saved || 'simple');
}

function wireTheme() {
  $('#btn-theme').onclick = () => {
    const dark = matchMedia('(prefers-color-scheme: dark)').matches;
    const cur = document.documentElement.dataset.theme || (dark ? 'dark' : 'light');
    const next = cur === 'dark' ? 'light' : 'dark';
    document.documentElement.dataset.theme = next;
    try { localStorage.setItem('ifas-theme', next); } catch { /* storage blocked */ }
    renderAll();   // charts and both 3D scenes read their colours from the theme
    skyline?.refreshTheme();
  };
}

function init() {
  buildChips($('#f-gender'), GENDERS, 'gender', (g) => COLOR[g], (g) => GENDER_PLURAL[g]);
  buildChips($('#f-age'), AGES, 'age', (a) => COLOR.age[a], (a) => AGE_TICK[a]);
  buildChips($('#f-tier'), TIERS, 'tier', (t) => COLOR.tier[t]);
  $('#f-conf').onchange = (e) => { state.filters.minConf = +e.target.value; renderAll(); };
  $('#f-reset').onclick = () => { clearFilters(); renderAll(); };
  $$('#seg-skyline button').forEach((b) => {
    b.onclick = () => {
      state.skyMetric = b.dataset.metric;
      $$('#seg-skyline button').forEach((x) => x.setAttribute('aria-pressed', String(x === b)));
      if (state.doc) renderSkyline(filtered());
    };
  });
  try { $('#chk-demo-trends').checked = localStorage.getItem('ifas-trend-samples') === '1'; } catch { /* storage blocked */ }
  $('#chk-demo-trends').onchange = (e) => {
    try { localStorage.setItem('ifas-trend-samples', e.target.checked ? '1' : '0'); } catch { /* storage blocked */ }
    renderTrends();
  };
  $('#btn-csv').onclick = exportCsv;
  wireStage();
  wireTabs();
  wireRunPicker();
  wireMode();
  wireTheme();
  wireProcess();
  setView(location.hash.slice(1) || 'overview', { remember: false });
  moveGlider(true);
  loadRuns();
}

init();
