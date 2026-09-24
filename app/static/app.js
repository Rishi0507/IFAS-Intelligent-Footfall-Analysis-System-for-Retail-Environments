/* Dashboard controller: load a run, filter it, render every view.

   All aggregation happens here on the client, over the *filtered* track list,
   so a filter chip changes every chart consistently - the analytics.json is
   the raw record, not a set of pre-baked chart payloads. */

import * as C from './charts.js';
const { fmt } = C;

const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];

const GENDERS = ['female', 'male'];
const AGES = ['Age16-30', 'Age31-45', 'Age46-60', 'AgeAbove61'];
const TIERS = ['Passing', 'Glancing', 'Browsing', 'Engaged', 'Dwelling'];
const AGE_SHORT = { 'Age16-30': '16–30', 'Age31-45': '31–45', 'Age46-60': '46–60', 'AgeAbove61': '61+' };
const COLOR = {
  female: 'var(--female)', male: 'var(--male)',
  age: { 'Age16-30': 'var(--age-1)', 'Age31-45': 'var(--age-2)', 'Age46-60': 'var(--age-3)', 'AgeAbove61': 'var(--age-4)' },
  tier: Object.fromEntries(TIERS.map((t, i) => [t, `var(--tier-${i + 1})`])),
};
const cap = (s) => (s ? s[0].toUpperCase() + s.slice(1) : '—');

const state = {
  doc: null,
  runId: null,
  runs: [],
  filters: { gender: new Set(), age: new Set(), tier: new Set(), minConf: 0 },
  video: null,
  playhead: 0,
};

/* ------------------------------------------------------------- loading */
async function loadRuns() {
  try {
    const res = await fetch('/api/runs');
    if (!res.ok) throw new Error();
    state.runs = await res.json();
  } catch {
    state.runs = [];
  }
  const sel = $('#run-select');
  if (!state.runs.length) {
    sel.innerHTML = '<option value="demo">demo (sample)</option>';
    return loadRun('demo');
  }
  sel.innerHTML = state.runs.map((r) =>
    `<option value="${r.id}">${r.source} · ${r.visitors} visitors</option>`).join('');
  return loadRun(state.runs[0].id);
}

async function loadRun(id) {
  // With the server: the API. Served as plain files (python -m http.server from
  // the repo root): the run directory, relative or absolute.
  const urls = [
    `/api/runs/${id}/analytics.json`,
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
  if (!doc) {
    document.querySelector('main').insertAdjacentHTML('afterbegin',
      `<div class="card"><p class="empty">Could not load <code>${id}</code>. Start the server with
       <code>python -m app.server</code>, or generate the sample run with
       <code>python scripts/generate_demo_run.py</code>.</p></div>`);
    return;
  }
  state.doc = doc;
  state.runId = id;
  const demo = !!doc.meta.demo;
  $('#run-badge').hidden = !demo;
  $('#run-badge-text').textContent = 'Sample data — no models were run';
  setupVideo(id, doc);
  renderAll();
  renderTrends();
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

function buildChips(host, values, key, colorOf, labelOf = (v) => v) {
  host.innerHTML = '';
  values.forEach((v) => {
    const b = document.createElement('button');
    b.className = 'chip';
    b.setAttribute('aria-pressed', 'false');
    b.dataset.value = v;
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
  $$('.chip').forEach((b) => {
    const key = b.parentElement.id.replace('f-', '');
    const map = { gender: 'gender', age: 'age', tier: 'tier' }[key];
    if (map) b.setAttribute('aria-pressed', String(state.filters[map].has(b.dataset.value)));
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

function heatFrom(rows, doc) {
  const { cols, rows: nRows } = doc.heatmap;
  const grid = [...Array(nRows)].map(() => Array(cols).fill(0));
  const { width: W, height: H } = doc.meta;
  rows.forEach((r) => r.path.forEach(([, x, y]) => {
    const c = Math.min(cols - 1, Math.max(0, Math.floor((x / W) * cols)));
    const rr = Math.min(nRows - 1, Math.max(0, Math.floor((y / H) * nRows)));
    grid[rr][c]++;
  }));
  const peak = Math.max(1, ...grid.flat());
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
        tbl.innerHTML = `<table><thead><tr>${d.columns.map((c) => `<th>${c}</th>`).join('')}</tr></thead>
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
  const rows = filtered();
  const all = doc.tracks;
  $('#f-count').textContent = rows.length === all.length
    ? `${all.length} visitors`
    : `${rows.length} of ${all.length} visitors`;

  renderSummary(doc, rows);
  renderInsights(doc);
  renderOverview(doc, rows);
  renderDemographics(doc, rows);
  renderEngagement(doc, rows);
  renderFlow(doc, rows);
  renderModel(doc, rows);
  renderLog(rows);
  renderPlayhead();
  wireTableToggles();
}

/* The simple view's headline: the same numbers, said in words a shop owner
   would use. No model vocabulary - no confidence, tracks, occupancy or mA. */
function renderSummary(doc, rows) {
  const host = $('#summary');
  if (!rows.length) { host.innerHTML = '<li>No people were found in this footage.</li>'; return; }
  const mins = doc.meta.duration_sec / 60;
  const bins = timeline(rows, doc);
  const busiest = bins.reduce((b, c) => (c.entries > b.entries ? c : b), bins[0]);
  const g = countBy(rows, 'gender', GENDERS);
  const a = countBy(rows, 'age', AGES);
  const hasAge = rows.some((r) => r.age);
  const topAge = AGES.reduce((x, k) => (a[k] > a[x] ? k : x), AGES[0]);
  const dwells = rows.map((r) => r.dwell_sec);
  const med = median(dwells);
  const zones = ['Left', 'Center', 'Right'];
  const topZone = zones.reduce((x, z) =>
    (rows.filter((r) => r.zone === z).length > rows.filter((r) => r.zone === x).length ? z : x), zones[0]);
  const zoneShare = rows.filter((r) => r.zone === topZone).length / rows.length * 100;
  const lingered = rows.filter((r) => r.dwell_sec >= 30).length;
  const women = g.female >= g.male;

  const lines = [
    ['👣', `<b>${fmt.int(rows.length)} people</b> came past the camera in ${fmt.dur(doc.meta.duration_sec)}`
      + (mins >= 1 ? ` — about <b>${fmt.num(rows.length / mins, 0)} a minute</b>.` : '.')],
    ['⏰', `Busiest at <b>${fmt.clock(busiest.t)}</b>, when ${fmt.int(busiest.entries)} `
      + `${busiest.entries === 1 ? 'person' : 'people'} arrived.`],
    ['🧍', `Mostly <b>${women ? 'women' : 'men'}</b> (${fmt.pct(Math.max(g.female, g.male) / rows.length * 100, 0)})`
      + (hasAge
        ? `, and most were <b>${AGE_SHORT[topAge]} years old</b> (${fmt.pct(a[topAge] / rows.length * 100, 0)}).`
        : '. Ages were not measured in this run.')],
    ['⏱️', `Half of them stayed longer than <b>${fmt.dur(med)}</b>`
      + (lingered ? `, and ${fmt.int(lingered)} stayed more than 30 seconds.` : '.')],
    ['📍', `The <b>${topZone.toLowerCase()} side</b> of the view was the busiest spot `
      + `(${fmt.pct(zoneShare, 0)} of people).`],
  ];
  host.innerHTML = lines.map(([ico, text]) =>
    `<li><span class="ico" aria-hidden="true">${ico}</span><span>${text}</span></li>`).join('');
}

function renderOverview(doc, rows) {
  const m = doc.meta;
  const simple = document.body.dataset.mode === 'simple';
  $('#occ-title').textContent = simple ? 'How busy it was' : 'Occupancy';
  $('#occ-sub').textContent = simple
    ? 'How many people were in view at the same time.'
    : 'People simultaneously in frame. Click to seek the video.';
  $('#peak-label').textContent = simple ? 'Most at once' : 'Peak occupancy';
  $('#dwell-label').textContent = simple ? 'Typical visit' : 'Average dwell';
  $('#ov-sub').textContent =
    `${m.source} · ${m.width}×${m.height} @ ${fmt.num(m.fps, 0)}fps · ${fmt.dur(m.duration_sec)} of footage`;

  const bins = timeline(rows, doc);
  const occ = occupancy(rows, doc);
  const dwells = rows.map((r) => r.dwell_sec);
  const g = countBy(rows, 'gender', GENDERS);
  const a = countBy(rows, 'age', AGES);

  $('#k-visitors').textContent = fmt.int(rows.length);
  $('#k-visitors-note').textContent = simple
    ? `different people seen in ${fmt.dur(m.duration_sec)}`
    : `unique tracked people · ${fmt.int(doc.tracks.length)} before filters`;
  C.sparkline($('#k-spark'), bins.map((b) => b.entries), 'var(--accent)');

  const peak = Math.max(0, ...occ.y);
  $('#k-peak').textContent = fmt.int(peak);
  $('#k-peak-note').textContent = simple
    ? `${fmt.num(mean(occ.y), 1)} on average`
    : `${fmt.num(mean(occ.y), 1)} average · tracked people in frame`;
  C.sparkline($('#k-peak-spark'), occ.y, 'var(--accent)');

  $('#k-dwell').innerHTML = `${fmt.num(mean(dwells), 1)}<small> s</small>`;
  $('#k-dwell-note').textContent = `median ${fmt.num(median(dwells), 1)}s · longest ${fmt.dur(Math.max(0, ...dwells))}`;
  const dh = histogram(dwells, 16, 0, Math.max(10, ...dwells));
  C.sparkline($('#k-dwell-spark'), dh.counts, 'var(--accent)');

  const busiest = bins.reduce((b, c) => (c.entries > b.entries ? c : b), bins[0] || { entries: 0, t: 0 });
  $('#k-rate').innerHTML = `${fmt.int(busiest.entries)}<small> visitors</small>`;
  $('#k-rate-note').textContent =
    `${fmt.clock(busiest.t)}–${fmt.clock(busiest.t + doc.timeline.bin_sec)} · ${fmt.num(rows.length / (m.duration_sec / 60), 1)}/min overall`;
  C.sparkline($('#k-rate-spark'), bins.map((b) => b.entries), 'var(--accent)');

  C.splitBar($('#k-gender-split'), [
    { label: 'Female', value: g.female, color: COLOR.female },
    { label: 'Male', value: g.male, color: COLOR.male },
  ]);
  $('#k-gender-note').textContent = `${fmt.int(g.female)} female · ${fmt.int(g.male)} male`
    + (g.female && g.male ? ` · ${fmt.num(g.female / g.male, 2)} women per man` : '');

  const hasAge = rows.some((r) => r.age);
  let best = { n: -1 };
  GENDERS.forEach((gg) => (hasAge ? AGES : [null]).forEach((aa) => {
    const n = rows.filter((r) => r.gender === gg && (!aa || r.age === aa)).length;
    if (n > best.n) best = { n, g: gg, a: aa };
  }));
  $('#k-segment').textContent = best.n > 0
    ? `${cap(best.g)}${best.a ? ` ${AGE_SHORT[best.a]}` : ''}` : '—';
  $('#k-segment-note').textContent = best.n > 0
    ? `${best.n} visitors · ${fmt.pct((best.n / rows.length) * 100)} of footfall`
      + (hasAge ? '' : ' · no age model in this run')
    : 'No visitors in this slice';

  const gc = mean(rows.map((r) => r.gender_conf));
  const ac = mean(rows.map((r) => r.age_conf));
  $('#k-conf').textContent = fmt.fixed(gc);
  $('#k-conf-meter').style.width = `${gc * 100}%`;
  $('#k-conf-note').textContent = `gender ${fmt.fixed(gc)} · age ${fmt.fixed(ac)} (mean per visitor)`;

  // Footfall over time - stacked by gender.
  C.groupedColumns($('#c-footfall'), {
    categories: bins.map((b) => fmt.clock(b.t)),
    series: GENDERS.map((gg) => ({ name: cap(gg), color: COLOR[gg], values: bins.map((b) => b.gender[gg]) })),
    stacked: true,
    height: 220,
  });
  C.legend($('#l-footfall'), GENDERS.map((gg) => ({ label: cap(gg), color: COLOR[gg] })));
  setTable('c-footfall', ['Interval', 'Female', 'Male', 'Total'],
    bins.map((b) => [fmt.clock(b.t), b.gender.female, b.gender.male, b.entries]));

  C.lines($('#c-occupancy'), {
    x: occ.x,
    series: [{ name: 'In frame', color: 'var(--accent)', values: occ.y }],
    area: true,
    height: 220,
    marker: state.playhead > 0 ? state.playhead : null,
    onSeek: (t) => {
      if (state.video) state.video.currentTime = t;
      else { state.playhead = t; renderPlayhead(); }
    },
  });
  setTable('c-occupancy', ['Time', 'In frame'],
    occ.x.filter((_, i) => i % 8 === 0).map((t, i) => [fmt.clock(t), occ.y[i * 8]]));
}

function renderDemographics(doc, rows) {
  const a = countBy(rows, 'age', AGES);
  const simple = document.body.dataset.mode === 'simple';

  // The same panels, described for whoever is reading them.
  const copy = simple ? {
    demoTitle: 'Who came in', demoSub: 'The mix of people the camera saw.',
    ageTitle: 'Age groups', ageSub: 'Roughly how old your visitors were.',
    agTitle: 'Age and gender together', agSub: 'Which groups visit you most.',
  } : {
    demoTitle: 'Demographics', demoSub: 'Everything the age and gender models emit, cross-tabulated.',
    ageTitle: 'Age mix', ageSub: 'Four PETA buckets. Click a bar to filter the whole dashboard.',
    agTitle: 'Age by gender', agSub: 'Where the two models disagree about who your shoppers are.',
  };
  $('#demo-title').textContent = copy.demoTitle;
  $('#demo-sub').textContent = copy.demoSub;
  $('#agemix-title').textContent = copy.ageTitle;
  $('#agemix-sub').textContent = copy.ageSub;
  $('#agegender-title').textContent = copy.agTitle;
  $('#agegender-sub').textContent = copy.agSub;
  $('#video-sub').textContent = simple
    ? 'Your footage, with a box around each person the system found.'
    : 'The annotated output, with box colours matching the charts.';

  const noAge = rows.length && !rows.some((r) => r.age);
  const notice = simple
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
      label: AGE_SHORT[k], value: a[k], color: COLOR.age[k], key: k,
      hint: 'Visitors',
    })),
    height: 210,
    onClick: (d) => {
      const set = state.filters.age;
      set.has(d.key) ? set.delete(d.key) : set.add(d.key);
      syncChips(); renderAll();
    },
  });
  setTable('c-age', ['Age bucket', 'Visitors', 'Share'],
    AGES.map((k) => [AGE_SHORT[k], a[k], fmt.pct((a[k] / (rows.length || 1)) * 100)]));

  C.groupedColumns($('#c-agegender'), {
    categories: AGES.map((k) => AGE_SHORT[k]),
    series: GENDERS.map((g) => ({
      name: cap(g), color: COLOR[g],
      values: AGES.map((k) => rows.filter((r) => r.age === k && r.gender === g).length),
    })),
    height: 210,
  });
  C.legend($('#l-agegender'), GENDERS.map((g) => ({ label: cap(g), color: COLOR[g] })));
  setTable('c-agegender', ['Age bucket', 'Female', 'Male'],
    AGES.map((k) => [AGE_SHORT[k],
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
    `<thead><tr><th>Segment</th><th class="num">Visitors</th><th class="num">Share</th>
      <th class="num">Avg dwell</th><th class="num">Median</th><th class="num">Longest</th>
      <th class="num">Dwell share</th><th class="num">Gender conf.</th><th class="num">Age conf.</th></tr></thead>
     <tbody>${segRows.map((r) => `<tr>
       <td><span class="pill"><i class="swatch" style="background:${COLOR[r.g]}"></i>${cap(r.g)} ${AGE_SHORT[r.k]}</span></td>
       <td class="num">${r.n}</td>
       <td class="num"><div class="bar-cell"><i style="width:${Math.max(2, r.share * 1.4)}px;background:${COLOR[r.g]}"></i>${fmt.pct(r.share)}</div></td>
       <td class="num">${fmt.num(r.avg, 1)}s</td><td class="num">${fmt.num(r.med, 1)}s</td>
       <td class="num">${fmt.dur(r.max)}</td>
       <td class="num">${fmt.pct((r.dwellShare / totalDwell) * 100)}</td>
       <td class="num">${fmt.fixed(r.gc)}</td><td class="num">${fmt.fixed(r.ac)}</td></tr>`).join('')
     || '<tr><td colspan="9" class="empty">No visitors in this slice.</td></tr>'}</tbody>`;
}

function renderEngagement(doc, rows) {
  const dwells = rows.map((r) => r.dwell_sec);
  const hi = Math.max(10, ...dwells);
  const h = histogram(dwells, 12, 0, hi);
  C.columns($('#c-dwellhist'), {
    data: h.counts.map((c, i) => ({
      label: i % 3 === 0 ? `${Math.round(h.edges[i])}s` : '',
      value: c, color: 'var(--accent)', hint: 'Visitors',
    })),
    height: 200,
  });
  setTable('c-dwellhist', ['Range', 'Visitors'],
    h.counts.map((c, i) => [`${Math.round(h.edges[i])}–${Math.round(h.edges[i + 1])}s`, c]));

  const t = countBy(rows, 'tier', TIERS);
  C.hbars($('#c-tiers'), {
    data: TIERS.map((k) => ({
      label: k, value: t[k], color: COLOR.tier[k],
      rows: [{ label: 'Visitors', value: fmt.int(t[k]), color: COLOR.tier[k] },
        { label: 'Share', value: fmt.pct((t[k] / (rows.length || 1)) * 100) }],
    })),
    labelW: 78, rowH: 28,
  });
  setTable('c-tiers', ['Tier', 'Visitors', 'Share'],
    TIERS.map((k) => [k, t[k], fmt.pct((t[k] / (rows.length || 1)) * 100)]));

  C.columns($('#c-dwellage'), {
    data: AGES.map((k) => ({
      label: AGE_SHORT[k],
      value: mean(rows.filter((r) => r.age === k).map((r) => r.dwell_sec)),
      color: COLOR.age[k], hint: 'Mean dwell',
    })),
    height: 200, format: (v) => fmt.num(v, 0), unit: 's',
  });
  setTable('c-dwellage', ['Age bucket', 'Mean dwell', 'Visitors'],
    AGES.map((k) => {
      const seg = rows.filter((r) => r.age === k);
      return [AGE_SHORT[k], `${fmt.num(mean(seg.map((r) => r.dwell_sec)), 1)}s`, seg.length];
    }));

  const segs = [];
  GENDERS.forEach((g) => AGES.forEach((k) => {
    const seg = rows.filter((r) => r.gender === g && r.age === k);
    if (seg.length) {
      segs.push({
        label: `${cap(g)} ${AGE_SHORT[k]}`, value: mean(seg.map((r) => r.dwell_sec)), color: COLOR[g],
        rows: [{ label: 'Mean dwell', value: `${fmt.num(mean(seg.map((r) => r.dwell_sec)), 1)}s`, color: COLOR[g] },
          { label: 'Visitors', value: seg.length }],
      });
    }
  }));
  segs.sort((x, y) => y.value - x.value);
  if (segs.length) {
    C.hbars($('#c-dwellseg'), { data: segs, labelW: 108, rowH: 28, format: (v) => fmt.num(v, 1), unit: 's' });
  } else {
    $('#c-dwellseg').innerHTML = '<p class="empty">No visitors in this slice.</p>';
  }

  C.stackedRows($('#c-tiergender'), {
    categories: GENDERS.map(cap),
    rowH: 46,
    series: TIERS.map((k) => ({
      name: k, color: COLOR.tier[k],
      values: GENDERS.map((g) => rows.filter((r) => r.gender === g && r.tier === k).length),
    })),
  });
  C.legend($('#l-tiergender'), TIERS.map((k) => ({ label: k, color: COLOR.tier[k] })));
}

function renderFlow(doc, rows) {
  const heat = heatFrom(rows, doc);
  C.heatgrid($('#c-heat'), {
    ...heat,
    aspect: doc.meta.width / doc.meta.height,
    paths: $('#chk-paths').checked
      ? rows.slice(0, 120).map((r) => r.path.map(([, x, y]) => [x / doc.meta.width, y / doc.meta.height]))
      : null,
  });

  const zones = ['Left', 'Center', 'Right'];
  $('#t-zones').innerHTML =
    `<thead><tr><th>Zone</th><th class="num">Visitors</th><th class="num">Share</th>
      <th class="num">Avg dwell</th><th class="num">F / M</th></tr></thead>
     <tbody>${zones.map((z) => {
      const seg = rows.filter((r) => r.zone === z);
      const f = seg.filter((r) => r.gender === 'female').length;
      return `<tr><td>${z}</td><td class="num">${seg.length}</td>
        <td class="num">${fmt.pct((seg.length / (rows.length || 1)) * 100)}</td>
        <td class="num">${fmt.num(mean(seg.map((r) => r.dwell_sec)), 1)}s</td>
        <td class="num">${f} / ${seg.length - f}</td></tr>`;
    }).join('')}</tbody>`;

  C.rampLegend($('#l-heat'), 'Fewer people', 'More');
  C.stackedRows($('#c-zoneage'), {
    categories: zones,
    rowH: 40, labelW: 56,
    series: AGES.map((k) => ({
      name: AGE_SHORT[k], color: COLOR.age[k],
      values: zones.map((z) => rows.filter((r) => r.zone === z && r.age === k).length),
    })),
  });
  C.legend($('#l-zoneage'), AGES.map((k) => ({ label: AGE_SHORT[k], color: COLOR.age[k] })));

  const sides = ['left', 'right', 'top', 'bottom', 'interior'];
  C.hbars($('#c-entry'), {
    data: sides.map((s) => ({ label: cap(s), value: rows.filter((r) => r.entry === s).length, hint: 'Visitors' })),
    labelW: 78, rowH: 28,
  });
  const flows = ['entered', 'exited', 'passed through', 'in view'];
  C.hbars($('#c-flow'), {
    data: flows.map((s) => ({ label: cap(s), value: rows.filter((r) => r.flow === s).length, hint: 'Visitors' })),
    labelW: 108, rowH: 28,
  });
}

function renderModel(doc, rows) {
  const gh = histogram(rows.map((r) => r.gender_conf), 10, 0, 1);
  const ah = histogram(rows.map((r) => r.age_conf), 10, 0, 1);
  const bars = (h, color) => h.counts.map((c, i) => ({
    label: i % 2 === 0 ? fmt.num(h.edges[i], 1) : '', value: c, color, hint: 'Visitors',
  }));
  C.columns($('#c-gconf'), { data: bars(gh, 'var(--male)'), height: 190 });
  C.columns($('#c-aconf'), { data: bars(ah, 'var(--age-3)'), height: 190 });

  const thresh = doc.config.low_conf_threshold ?? 0.6;
  const low = rows.filter((r) => r.gender_conf < thresh).length;
  $('#m-lowconf').textContent = fmt.pct((low / (rows.length || 1)) * 100);
  $('#m-lowconf-note').textContent =
    `${low} of ${rows.length} visitors below ${fmt.num(thresh, 2)} gender confidence · ${doc.kpis.tracks_dropped} short tracks discarded`;

  const M = doc.model || {};
  $('#m-models').innerHTML = [
    ['Gender', M.gender?.family ?? '—'],
    ['Checkpoint', M.gender?.source ?? '—'],
    ['Age', M.age?.family ?? '—'],
    ['Reported mA', M.age?.reported_avg_mA ? fmt.num(M.age.reported_avg_mA, 3) : '—'],
    ['Crops classified', fmt.int(doc.meta.crops_classified || 0)],
  ].map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join('');

  const c = doc.config;
  $('#m-config').innerHTML = [
    ['Detector', c.detector],
    ['Detection conf.', c.det_conf],
    ['Frame stride', `every ${c.frame_stride} frames`],
    ['Bin size', `${c.bin_sec}s`],
    ['Device', doc.meta.device],
    ['Throughput', `${fmt.num(doc.meta.processing_fps || 0, 1)} fps`],
    ['Generated', (doc.generated_at || '').replace('T', ' ').slice(0, 19)],
  ].map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join('');
}

/* ---------------------------------------------------------- insights */
function renderInsights(doc) {
  const host = $('#insight-list');
  const items = doc.insights || [];
  host.innerHTML = items.length
    ? items.map((i) => `<li><span class="k">${i.label || ''}</span><span class="v">${i.text}</span></li>`).join('')
    : '<li class="v">This run was produced before insights existed — re-process the video to get them.</li>';
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
  { id: 'engaged_pct', label: 'Stayed 30s+', fmt: (v) => fmt.num(v, 0), unit: '%' },
  { id: 'female_pct', label: 'Share women', fmt: (v) => fmt.num(v, 0), unit: '%' },
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
      'Start the dashboard with python -m app.server to compare recordings.'));
    return;
  }
  periodState.data = t;
  const c = t.coverage;
  const src = t.sources;

  // Say plainly what these numbers are made of.
  const parts = [];
  if (c.runs) {
    parts.push(`${c.runs} recording${c.runs === 1 ? '' : 's'}`);
    parts.push(`${fmt.num(c.observed_hours, 1)} hours of footage`);
    parts.push(`${c.days} day${c.days === 1 ? '' : 's'}`);
  }
  $('#trend-coverage').innerHTML = c.runs === 0
    ? 'No recordings yet — process a few videos and this fills in.'
    : parts.join(' · ')
      + (src.samples_included
        ? ` <b style="color:var(--warning)">· includes ${src.sample_runs} generated sample${src.sample_runs === 1 ? '' : 's'}</b>`
        : (src.sample_runs ? ` · ${src.sample_runs} sample${src.sample_runs === 1 ? '' : 's'} excluded` : ''))
      + (c.mixed_cameras ? ' · <b>mixing several cameras</b>' : '');

  const head = $('#trend-headline');
  if (t.trend && c.enough_for_trend) {
    const d = t.trend.change_pct;
    const dir = d > 0 ? 'up' : 'down';
    head.innerHTML =
      `<div class="headline"><span class="big">${fmt.num(t.trend.latest_per_hour, 0)}</span>`
      + `<span style="color:var(--muted);font-size:13px">visitors per hour on ${t.trend.latest_date}</span>`
      + (d == null ? '' : `<span class="delta ${dir}">${d > 0 ? '▲' : '▼'} ${fmt.pct(Math.abs(d), 0)}</span>`) + '</div>'
      + `<p class="sub" style="margin:0">vs ${fmt.num(t.trend.previous_avg_per_hour, 0)} per hour `
      + 'averaged over the earlier days.</p>';
    C.lines($('#c-trendline'), {
      x: t.by_day.map((_, i) => i),
      series: [{ name: 'Visitors per hour', color: 'var(--accent)', values: t.by_day.map((d2) => d2.per_hour) }],
      area: true, height: 150,
      xFormat: (i) => (t.by_day[i]?.key || '').slice(5),
      format: (v) => fmt.num(v, 0),
    });
  } else {
    const need = Math.max(1, c.min_days_for_trend - c.usable_days);
    head.innerHTML = `<p class="empty">Record ${need} more day${need === 1 ? '' : 's'} `
      + 'to compare footfall over time.</p>';
    $('#c-trendline').innerHTML = '';
  }

  renderPeriod();

  if (t.by_hour.length) {
    C.columns($('#c-hour'), {
      data: t.by_hour.map((h) => ({
        label: `${h.key}:00`, value: h.per_hour, color: 'var(--accent)',
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
        { name: 'Female', color: COLOR.female, values: mix.map((r) => r.gender.female) },
        { name: 'Male', color: COLOR.male, values: mix.map((r) => r.gender.male) },
      ],
    });
    C.legend($('#l-mix'), GENDERS.map((g) => ({ label: cap(g), color: COLOR[g] })));
  } else {
    empty($('#c-mix'), 'No dated recordings yet.');
    $('#l-mix').innerHTML = '';
  }

  $('#t-runs').innerHTML =
    `<thead><tr><th>Recording</th><th>Source</th><th>When</th><th>Camera</th>
      <th class="num">Visitors</th><th class="num">Per hour</th><th class="num">Avg visit</th>
      <th class="num">Stayed 30s+</th><th class="num">Women</th></tr></thead>
     <tbody>${t.runs.map((r) => `<tr>
       <td>${r.source}</td>
       <td>${r.demo ? '<span class="badge" style="padding:1px 7px">Sample</span>' : 'Footage'}</td>
       <td>${(r.captured_at || '—').replace('T', ' ')}</td>
       <td>${r.site} · ${r.camera}</td>
       <td class="num">${r.visitors}</td>
       <td class="num">${r.comparable ? fmt.num(r.per_hour, 0) : '—'}</td>
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
    day: 'One bar per day you recorded. Rates are per observed hour, so a short clip cannot outweigh a long one.',
    weekday: 'Every recording grouped by day of the week — Mondays with Mondays.',
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
      value: r[metric.id], color: 'var(--accent)',
      hint: metric.label,
    })),
    height: 220, format: metric.fmt, unit: metric.unit,
  });

  const head = { day: 'Date', weekday: 'Day', month: 'Month' }[periodState.period];
  $('#t-period').innerHTML =
    `<thead><tr><th>${head}</th>
      <th class="num">Hours watched</th><th class="num">Visitors</th><th class="num">Per hour</th>
      <th class="num">Avg visit</th><th class="num">Stayed 30s+</th><th class="num">Women</th>
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
        <td class="num">${top && top[1] ? `${AGE_SHORT[top[0]]} (${top[1]})` : '—'}</td></tr>`;
    }).join('')}</tbody>`;
}

/* ------------------------------------------------------------ log/CSV */
let sortKey = 'first_sec', sortDir = 1;
function renderLog(rows) {
  $('#log-count').textContent = `${rows.length} tracked visitors`;
  const cols = [
    ['id', '#'], ['first_sec', 'First seen'], ['dwell_sec', 'Dwell'], ['tier', 'Tier'],
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
      `<th data-k="${k}">${l}${sortKey === k ? (sortDir > 0 ? ' ↑' : ' ↓') : ''}</th>`).join('')}</tr></thead>
     <tbody>${sorted.slice(0, 500).map((r) => `<tr>
       <td>${r.id}</td><td>${fmt.clock(r.first_sec)}</td><td class="num">${fmt.num(r.dwell_sec, 1)}s</td>
       <td>${r.tier}</td>
       <td><span class="pill"><i class="swatch" style="background:${COLOR[r.gender] || 'var(--muted)'}"></i>${cap(r.gender)}</span></td>
       <td class="num">${fmt.fixed(r.gender_conf)}</td>
       <td><span class="pill"><i class="swatch" style="background:${COLOR.age[r.age] || 'var(--muted)'}"></i>${AGE_SHORT[r.age] || '—'}</span></td>
       <td class="num">${fmt.fixed(r.age_conf)}</td>
       <td>${r.zone}</td><td>${cap(r.flow)}</td><td class="num">${r.frames}</td>
       <td class="num">${fmt.fixed(r.det_conf)}</td><td class="num">${fmt.pct(r.avg_size_pct)}</td></tr>`).join('')
     || '<tr><td colspan="13" class="empty">No visitors in this slice.</td></tr>'}</tbody>`;
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
        <p style="margin:0 0 8px">This run has no annotated video.</p>
        <p style="margin:0">Re-run the pipeline without <code>--no-video</code>, or use <b>Process video</b>.</p>
      </div></div>`;
  } else {
    const v = document.createElement('video');
    v.controls = true;
    v.preload = 'metadata';
    v.src = `/api/runs/${id}/video.mp4`;
    v.poster = `/api/runs/${id}/poster.jpg`;
    wrap.appendChild(v);
    state.video = v;
    if (doc.meta.video_codec && doc.meta.video_codec !== 'avc1') {
      wrap.insertAdjacentHTML('beforeend',
        `<div class="badge" style="position:absolute;left:10px;top:10px"><i class="dot"></i>
         Encoded as ${doc.meta.video_codec} — install ffmpeg and re-run for browser-playable H.264</div>`);
    }
    let last = 0;
    v.addEventListener('timeupdate', () => {
      if (v.currentTime - last < 0.25 && v.currentTime > last) return;
      last = state.playhead = v.currentTime;
      renderPlayhead();
    });
  }
  C.legend($('#l-video'), [
    { label: 'Female', color: COLOR.female },
    { label: 'Male', color: COLOR.male },
    { label: 'Box label: #id · gender + confidence · age bucket', color: 'transparent' },
  ]);
}

function renderPlayhead() {
  const t = state.playhead;
  const rows = filtered();
  const now = rows.filter((r) => r.first_sec <= t && r.last_sec >= t);
  $('#video-time').textContent = state.video
    ? `${fmt.clock(t)} of ${fmt.clock(state.doc.meta.duration_sec)}`
    : 'No video in this run — scrub the occupancy chart to move the playhead';
  $('#v-count').textContent = now.length;
  $('#v-entries').textContent = rows.filter((r) => r.first_sec <= t).length;
  C.splitBar($('#v-split'), [
    { label: 'Female', value: now.filter((r) => r.gender === 'female').length, color: COLOR.female },
    { label: 'Male', value: now.filter((r) => r.gender === 'male').length, color: COLOR.male },
  ], (v) => `${fmt.num(v, 0)}%`);
  C.hbars($('#v-ages'), {
    data: AGES.map((k) => ({ label: AGE_SHORT[k], value: now.filter((r) => r.age === k).length, color: COLOR.age[k], hint: 'In frame' })),
    labelW: 52, rowH: 24,
  });
  const occChart = $('#c-occupancy');
  if (occChart && occChart.__render) occChart.__render();
}

/* ---------------------------------------------------------- processing */
function wireProcess() {
  const dlg = $('#dlg-process');
  $('#btn-process').onclick = () => dlg.showModal();
  $('#proc-cancel').onclick = () => dlg.close();
  $('#form-process').addEventListener('submit', async (e) => {
    e.preventDefault();
    const form = new FormData(e.target);
    $('#proc-status').textContent = 'Uploading…';
    $('#proc-progress').hidden = false;
    $('#proc-go').disabled = true;
    try {
      const res = await fetch('/api/runs', { method: 'POST', body: form });
      if (!res.ok) throw new Error(await res.text());
      const { job } = await res.json();
      const poll = setInterval(async () => {
        const s = await (await fetch(`/api/jobs/${job}`)).json();
        const eta = s.eta_sec != null ? ` · ~${C.fmt.dur(s.eta_sec)} left` : '';
        const seen = s.classified != null ? ` · ${s.classified} crops classified` : '';
        $('#proc-status').textContent = s.error
          || (s.stage === 'done' ? 'Done.' : `${s.stage}… ${s.pct ?? 0}%${eta}${seen}`);
        $('#proc-progress').firstElementChild.style.width = `${s.pct ?? 0}%`;
        if (s.stage === 'done' || s.stage === 'failed') {
          clearInterval(poll);
          $('#proc-go').disabled = false;
          if (s.stage === 'done') { dlg.close(); loadRuns(); }
        }
      }, 1200);
    } catch (err) {
      $('#proc-status').textContent = `Failed: ${err.message}. Is the server running?`;
      $('#proc-go').disabled = false;
    }
  });
}

/* ----------------------------------------------------------- chrome */
function setMode(mode) {
  document.body.dataset.mode = mode;
  localStorage.setItem('ifas-mode', mode);
  $('#btn-simple').setAttribute('aria-pressed', String(mode === 'simple'));
  $('#btn-full').setAttribute('aria-pressed', String(mode !== 'simple'));
  if (mode === 'simple') {
    // A hidden filter cannot be left silently applied.
    state.filters = { gender: new Set(), age: new Set(), tier: new Set(), minConf: 0 };
    $('#f-conf').value = '0';
    syncChips();
  }
  if (state.doc) renderAll();
}

function wireMode() {
  $('#btn-simple').onclick = () => setMode('simple');
  $('#btn-full').onclick = () => setMode('full');
  setMode(localStorage.getItem('ifas-mode') || 'simple');
}

function wireTheme() {
  const saved = localStorage.getItem('ifas-theme');
  if (saved) document.documentElement.dataset.theme = saved;
  $('#btn-theme').onclick = () => {
    const dark = matchMedia('(prefers-color-scheme: dark)').matches;
    const cur = document.documentElement.dataset.theme || (dark ? 'dark' : 'light');
    const next = cur === 'dark' ? 'light' : 'dark';
    document.documentElement.dataset.theme = next;
    localStorage.setItem('ifas-theme', next);
    renderAll();   // charts cache surface colours for mark rings
  };
}

function wireNav() {
  const links = $$('nav.sections a');
  const obs = new IntersectionObserver((entries) => {
    entries.forEach((en) => {
      if (!en.isIntersecting) return;
      links.forEach((l) => l.classList.toggle('active', l.getAttribute('href') === `#${en.target.id}`));
    });
  }, { rootMargin: '-120px 0px -70% 0px' });
  $$('main section').forEach((s) => obs.observe(s));
}

function init() {
  buildChips($('#f-gender'), GENDERS, 'gender', (g) => COLOR[g], cap);
  buildChips($('#f-age'), AGES, 'age', (a) => COLOR.age[a], (a) => AGE_SHORT[a]);
  buildChips($('#f-tier'), TIERS, 'tier', null);
  $('#f-conf').onchange = (e) => { state.filters.minConf = +e.target.value; renderAll(); };
  $('#f-reset').onclick = () => {
    state.filters = { gender: new Set(), age: new Set(), tier: new Set(), minConf: 0 };
    $('#f-conf').value = '0';
    syncChips();
    renderAll();
  };
  $('#chk-paths').onchange = () => renderFlow(state.doc, filtered());
  $('#chk-demo-trends').checked = localStorage.getItem('ifas-trend-samples') === '1';
  $('#chk-demo-trends').onchange = (e) => {
    localStorage.setItem('ifas-trend-samples', e.target.checked ? '1' : '0');
    renderTrends();
  };
  $('#btn-csv').onclick = exportCsv;
  $('#run-select').onchange = (e) => loadRun(e.target.value);
  wireMode();
  wireTheme();
  wireNav();
  wireProcess();
  loadRuns();
}

init();
