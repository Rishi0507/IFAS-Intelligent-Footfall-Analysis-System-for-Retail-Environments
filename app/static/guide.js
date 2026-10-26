/* The Insights panel: each claim beside its evidence from the recording.

   The words are written server-side (src/inference/guide.py), each with the
   numbers it rests on; this lays them out as one card per claim: the figure,
   one line, one thing to try, and the proof. A proof is the frame the number
   was counted on, with everyone counted drawn over it (GET .../moment), the
   people it is about, each in their clearest crop (GET .../portraits), or
   both ends of a track that was lost and begun again (GET .../handoffs).
   Longer explanations wait behind "Why and how". Runs processed before the
   guide existed get one from GET /api/runs/{id}/guide. */

const esc = (s) => String(s ?? '').replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

const CONF = {
  solid: { label: 'Solid', note: 'You can plan on this.' },
  rough: { label: 'Rough guide', note: 'Good for direction, not for exact figures.' },
  unreliable: { label: 'Unreliable here', note: 'Do not plan on this number for this recording.' },
};

const ICON = {
  busy: '<path d="M3 15V9m4 6V5m4 10v-7m4 7V3"/>',
  arrivals: '<circle cx="9" cy="9" r="6.2"/><path d="M9 5.6V9l2.4 1.6"/>',
  stopping: '<path d="M6 2.5h6M6 15.5h6M6.6 2.5c0 3.6 4.8 3.6 4.8 6.5s-4.8 2.9-4.8 6.5"/>',
  where: '<path d="M9 16s-5-4.6-5-8.4a5 5 0 0 1 10 0C14 11.4 9 16 9 16z"/><circle cx="9" cy="7.6" r="1.8"/>',
  seats: '<path d="M4.5 9.5V4.6a1.6 1.6 0 0 1 1.6-1.6h5.8a1.6 1.6 0 0 1 1.6 1.6v4.9M3 9.5h12v3H3zM4.5 12.5V15m9-2.5V15"/>',
  queues: '<circle cx="4.5" cy="6" r="1.8"/><circle cx="9" cy="6" r="1.8"/><circle cx="13.5" cy="6" r="1.8"/><path d="M2.5 14c.3-2.2 1-3.4 2-3.4s1.7 1.2 2 3.4m.5 0c.3-2.2 1-3.4 2-3.4s1.7 1.2 2 3.4m.5 0c.3-2.2 1-3.4 2-3.4s1.7 1.2 2 3.4"/>',
  who: '<circle cx="6.5" cy="6" r="2.6"/><path d="M1.8 15c.4-2.7 2.3-4.2 4.7-4.2s4.3 1.5 4.7 4.2"/><circle cx="12.8" cy="6.6" r="2"/><path d="M13 10.9c1.7.2 2.9 1.5 3.2 3.5"/>',
  trust: '<path d="M9 2.2 15 4.6v4.2c0 3.6-2.5 6-6 7-3.5-1-6-3.4-6-7V4.6z"/><path d="m6.4 9 1.8 1.8L11.8 7"/>',
  warn: '<path d="M9 2.5 16 15H2z"/><path d="M9 7.5v3.2m0 2.2v.1"/>',
  tip: '<path d="M6.8 13.5h4.4M7.4 16h3.2M9 2a4.8 4.8 0 0 0-2.8 8.7c.5.4.8 1 .8 1.6v.2h4v-.2c0-.6.3-1.2.8-1.6A4.8 4.8 0 0 0 9 2z"/>',
};
const ico = (name) => `<span class="g-ico"><svg viewBox="0 0 18 18" aria-hidden="true">${ICON[name] || ICON.trust}</svg></span>`;

const NUMBER_NAMES = {
  average_in_view: 'People in view, on average', peak_in_view: 'People in view, at most', peak_at_sec: 'Busiest moment (seconds in)',
  arrivals: 'Arrivals', already_there: 'Already there at the start', window: 'Window used',
  median_stay_sec: 'Typical stay (seconds)', passed_through: 'Passed straight through', looked_briefly: 'Looked briefly',
  stayed_30s_plus: 'Stayed 30 s or more', places: 'Places', taken_at_end: 'Taken at the end',
  mean_utilisation: 'Share of time taken', turnover_per_hour_median: 'Stays per hour (typical place)',
  female_pct: 'Women (%)', male_pct: 'Men (%)', gender_unknown_share: 'Not sure about gender', people: 'People followed',
  gender_test_balanced_accuracy: 'Gender model, held-out test', age_test_balanced_accuracy: 'Age model, held-out test',
  tracks_per_person_in_view: 'Tracks per person in view', queue: 'Queue', median_wait_sec: 'Typical wait (seconds)',
  max_wait_sec: 'Longest wait (seconds)', max_length: 'Most people waiting',
};
const SHARE = new Set(['passed_through', 'looked_briefly', 'stayed_30s_plus', 'mean_utilisation', 'gender_unknown_share',
  'gender_test_balanced_accuracy', 'age_test_balanced_accuracy']);
const AGE = { 'Age16-30': '16–30', 'Age31-45': '31–45', 'Age46-60': '46–60', AgeAbove61: '61+' };
const GENDER = { female: 'Woman', male: 'Man' };

function numbers(obj) {
  const rows = Object.entries(obj || {}).filter(([k, v]) => NUMBER_NAMES[k] && v !== null && v !== undefined && typeof v !== 'object');
  if (!rows.length) return '';
  const fmt = (k, v) => (SHARE.has(k) && typeof v === 'number' ? `${Math.round(v * 100)}%`
    : typeof v === 'number' ? (Number.isInteger(v) ? v.toLocaleString() : v.toFixed(1)) : esc(v));
  return `<dl class="pf-num">${rows.map(([k, v]) => `<dt>${esc(NUMBER_NAMES[k])}</dt><dd>${fmt(k, v)}</dd>`).join('')}</dl>`;
}

/* Times: the position in the recording (m:ss), which is what the video's
   time bar shows, and the time of day when the recording's start is known. */
function pos(t) {
  const s = Math.max(0, Math.round(t));
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`;
}
function clockAt(doc, t) {
  const start = doc?.meta?.captured_at ? new Date(doc.meta.captured_at) : null;
  if (!start || Number.isNaN(+start)) return pos(t);
  return new Date(+start + t * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false });
}
function shortDur(sec) {
  const s = Math.round(sec || 0);
  if (s < 90) return `${s}s`;
  const m = Math.floor(s / 60), r = s % 60;
  if (m < 10) return r ? `${m}m ${r}s` : `${m}m`;
  return m < 90 ? `${m}m` : `${Math.floor(m / 60)}h ${m % 60}m`;
}

/* A strip of bars for one series over the recording: one colour, no legend
   (the caption names it), every bar with a tooltip. The bar holding `mark`
   (the moment a proof shows) is labelled; without one, the peak is. */
function strip(series, mark = null) {
  const pts = series?.points || [];
  if (pts.length < 2) return '';
  const max = Math.max(...pts.map((p) => p.v), 0) || 1;
  const W = 600, H = 64, gap = 2;
  const bw = Math.max(1, (W - gap * (pts.length - 1)) / pts.length);
  const at = mark === null ? -1 : pts.findLastIndex((p) => p.t <= mark);
  const peak = at >= 0 ? pts[at] : pts.reduce((a, p) => (p.v > a.v ? p : a), pts[0]);
  const bars = pts.map((p, i) => {
    const h = Math.max(p.v > 0 ? 2 : 0, (p.v / max) * (H - 20));
    const x = i * (bw + gap);
    return `<rect x="${x.toFixed(1)}" y="${(H - h).toFixed(1)}" width="${bw.toFixed(1)}" height="${h.toFixed(1)}" rx="${Math.min(2, bw / 2).toFixed(1)}"${p === peak ? ' class="pk"' : ''}><title>${esc(p.clock)}: ${p.v} ${esc(series.unit)}</title></rect>`;
  }).join('');
  const px = pts.indexOf(peak) * (bw + gap) + bw / 2;
  const anchor = px < 60 ? 'start' : px > W - 60 ? 'end' : 'middle';
  return `<figure class="g-strip">
    <svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(series.label)}">
      ${bars}
      <text x="${px.toFixed(1)}" y="13" text-anchor="${anchor}">${at >= 0 ? 'the frame shown' : `${Math.round(peak.v)} ${esc(series.unit)}`}</text>
    </svg>
    <figcaption><span>${esc(pts[0].clock)}</span><span>${esc(series.label)}</span><span>${esc(pts[pts.length - 1].clock)}</span></figcaption>
  </figure>`;
}

/* Parts of a whole as one bar with gaps, a legend and the share in words. */
function bar(parts, label) {
  const total = parts.reduce((a, p) => a + p.value, 0);
  if (!total) return '';
  return `<div class="pf-bar-wrap" role="img" aria-label="${esc(label)}: ${parts.map((p) => `${p.label} ${p.text}`).join(', ')}">
    <div class="pf-bar">${parts.filter((p) => p.value > 0).map((p) => `<i style="flex:${p.value};background:${p.color}" title="${esc(p.label)}: ${esc(p.text)}"></i>`).join('')}</div>
    <div class="pf-legend">${parts.map((p) => `<span><i style="background:${p.color}"></i>${esc(p.label)} <b>${esc(p.text)}</b></span>`).join('')}</div>
  </div>`;
}
function stayBar(shares) {
  const colors = ['var(--tier-1)', 'var(--tier-3)', 'var(--tier-5)'];
  return bar(shares.map((s, i) => ({ label: s.label, value: s.share, text: `${Math.round(s.share * 100)}%`, color: colors[i] })), 'How long people stayed');
}
function genderBar(doc) {
  const t = doc?.tracks || [];
  const n = (g) => t.filter((x) => x.gender === g).length;
  const parts = [
    { label: 'Women', value: n('female'), color: 'var(--female)' },
    { label: 'Men', value: n('male'), color: 'var(--male)' },
    { label: 'Not sure', value: t.filter((x) => !x.gender).length, color: 'var(--muted)' },
  ].filter((p) => p.value).map((p) => ({ ...p, text: String(p.value) }));
  return bar(parts, 'Gender');
}

/* ------------------------------------------------------------- proofs */

function frameProof(p, run) {
  const src = `/api/runs/${encodeURIComponent(run)}/still?t=${p.t}&w=960`;
  return `<figure class="pf-fig pf-frame" data-t="${p.t}" data-show="${esc(p.show || 'people')}" data-focus="${esc(p.focus || '')}">
    <div class="pf-shot" role="button" tabindex="0" aria-label="Enlarge the frame">
      <img decoding="async" src="${src}" alt="${esc(p.caption)}">
      <svg class="pf-over" aria-hidden="true"></svg>
      <span class="pf-badge">${pos(p.t)} into the recording</span>
    </div>
    <figcaption><span class="pf-keys"><span class="pf-wait">Marking everyone counted</span></span>
      <button class="link pf-play" data-seek="${p.t}">Play from here</button></figcaption>
  </figure>`;
}

function heatProof(p, run, doc) {
  const h = doc?.heatmap;
  if (!h?.grid?.length) return '';
  const max = Math.max(...h.grid.flat(), 1e-9);
  const cells = h.grid.flatMap((row, r) => row.map((v, c) => (v > 0
    ? `<rect x="${c * 10}" y="${r * 10}" width="10" height="10" style="opacity:${(0.12 + 0.68 * Math.sqrt(v / max)).toFixed(2)}"><title>${Math.round((v / max) * 100)}% of the busiest spot</title></rect>`
    : ''))).join('');
  const zones = doc.distributions?.zones || {};
  const total = Object.values(zones).reduce((a, z) => a + (z.visitors || 0), 0) || 1;
  const strips = [['Left', 'Left'], ['Center', 'Middle'], ['Right', 'Right']]
    .map(([k, name], i) => `<span class="pf-third" style="left:${(i * 100) / 3}%">${name} <b>${Math.round(((zones[k]?.visitors || 0) / total) * 100)}%</b></span>`).join('');
  const src = `/api/runs/${encodeURIComponent(run)}/still?t=${p.t}&w=960`;
  return `<figure class="pf-fig pf-heat">
    <div class="pf-shot" role="button" tabindex="0" aria-label="Enlarge the frame">
      <img decoding="async" src="${src}" alt="${esc(p.caption)}">
      <svg class="pf-over pf-heatmap" viewBox="0 0 ${h.cols * 10} ${h.rows * 10}" preserveAspectRatio="none" aria-hidden="true">
        <g class="cells">${cells}</g>
        <line x1="${(h.cols * 10) / 3}" y1="0" x2="${(h.cols * 10) / 3}" y2="${h.rows * 10}" vector-effect="non-scaling-stroke"/>
        <line x1="${(h.cols * 20) / 3}" y1="0" x2="${(h.cols * 20) / 3}" y2="${h.rows * 10}" vector-effect="non-scaling-stroke"/>
      </svg>
      ${strips}
    </div>
    <figcaption><span class="pf-keys"><span class="pf-key"><i class="k-heat"></i>Brighter: more time spent there</span><span class="pf-key">Share of people in each third</span></span></figcaption>
  </figure>`;
}

/* How a person's gender was decided: the pictures that voted for the label
   out of all that voted ("11/12 votes"), or the split when it was unsure. */
function votes(v, label) {
  const n = Object.values(v || {}).reduce((a, b) => a + b, 0);
  if (!n) return null;
  if (label) return `${v[label] || 0}/${n} votes`;
  return Object.entries(v).map(([k, c]) => `${c} ${k === 'female' ? 'woman' : 'man'}`).join(' vs ');
}

function personTile(id, show, doc) {
  const t = (doc?.tracks || []).find((x) => x.id === id) || {};
  const g = GENDER[t.gender];
  const dot = `<i style="background:${t.gender ? `var(--${t.gender})` : 'var(--muted)'}"></i>`;
  let lb, sm;
  if (show === 'stay') { lb = `Stayed ${shortDur(t.dwell_sec)}`; sm = `from ${clockAt(doc, t.first_sec || 0)}`; }
  else if (show === 'arrived') { lb = `Arrived ${clockAt(doc, t.first_sec || 0)}`; sm = [g || 'Not sure', AGE[t.age]].filter(Boolean).join(' · '); }
  else if (show === 'span') { lb = `Track ${id}`; sm = `${pos(t.first_sec || 0)} to ${pos(t.last_sec || 0)}`; }
  else {
    lb = `${dot}${g || 'Not sure'}`;
    sm = [AGE[t.age] || 'age not sure', votes(t.gender_votes, t.gender)].filter(Boolean).join(' · ');
  }
  return `<button class="pf-person" data-seek="${t.first_sec || 0}" title="Play from when this person appears">
    <span class="ph"><img alt="Person ${id}" data-portrait="${id}"></span>
    <span class="lb">${lb}</span><span class="sm">${esc(sm)}</span>
  </button>`;
}

function peopleProof(p, doc) {
  if (!p.tracks?.length) return '';
  const hint = p.kind === 'split' ? ' Compare the pictures: if it is one person, they were counted again each time.' : '';
  return `<figure class="pf-fig pf-crowd">
    <div class="pf-people">${p.tracks.map((id) => personTile(id, p.show, doc)).join('')}</div>
    <figcaption><span class="pf-cap">${esc(p.caption)}.${esc(hint)}</span></figcaption>
  </figure>`;
}

function handoffProof(p) {
  return `<figure class="pf-fig pf-handoffs" hidden>
    <div class="pf-pairs"></div>
    <figcaption><span class="pf-cap"></span></figcaption>
  </figure>`;
}

function handoffRows(h, doc) {
  const side = (track, t, url, what) => `<button class="pf-snap" data-seek="${Math.max(0, t - 1)}" title="Play from here">
      <img src="${esc(url)}" alt="Track ${track}, ${what} ${pos(t)}">
      <span class="lb">Track ${track}</span><span class="sm">${what} ${clockAt(doc, t)}</span>
    </button>`;
  return h.examples.map((e) => `<div class="pf-pair">
      ${side(e.a, e.a_t, e.a_url, 'last seen')}
      <span class="pf-gap"><b>${e.gap.toFixed(0)} s</b> later,<br>same spot<br><small>${Math.round(e.overlap * 100)}% overlap</small></span>
      ${side(e.b, e.b_t, e.b_url, 'first seen')}
    </div>`).join('');
}

function proof(p, run, doc) {
  if (!p) return '';
  if (p.kind === 'handoffs') return handoffProof(p);
  if (p.kind === 'frame') return frameProof(p, run);
  if (p.kind === 'heat') return heatProof(p, run, doc);
  if (p.kind === 'people' || p.kind === 'split') return peopleProof(p, doc);
  return '';
}

/* What a frame proof draws: everyone counted at that moment (seen in the
   frame, or hidden for a moment and shown where last seen), or the seats
   with their state. The boxes add up to the number the claim reports. */
function overlay(m, show, focus) {
  const W = m.width, H = m.height;
  const line = (p) => (p.seen
    ? `Person ${p.track} · ${GENDER[p.gender] || 'gender not sure'}${AGE[p.age] ? ` · ${AGE[p.age]}` : ''}`
    : `Person ${p.track} · not detected in this frame, last seen ${p.ago} s before`);
  const boxes = (cls, list) => list.map((p) => {
    const [x1, y1, x2, y2] = p.box;
    return `<rect class="${cls}" x="${x1}" y="${y1}" width="${x2 - x1}" height="${y2 - y1}" vector-effect="non-scaling-stroke"><title>${esc(line(p))}</title></rect>`;
  }).join('');
  if (show === 'zones') {
    const shown = m.zones.filter((z) => (focus ? z.id === focus : z.kind === 'seat' || z.kind === 'table'));
    // Taken seats only, each with how long it had been taken at that moment.
    const drawn = m.zones.filter((z) => z.taken || z.kind === 'queue');
    const polys = drawn.map((z) => {
      const on = shown.includes(z);
      const cls = `${z.kind === 'queue' ? 'queue' : 'taken'}${on ? '' : ' dim'}`;
      const pts = z.polygon.map(([x, y]) => `${(x * W).toFixed(1)},${(y * H).toFixed(1)}`).join(' ');
      return `<polygon class="${cls}" points="${pts}" vector-effect="non-scaling-stroke"><title>${esc(z.name)} · taken for ${pos(m.t - z.since)}</title></polygon>`;
    }).join('');
    const timers = drawn.filter((z) => z.taken && shown.includes(z)).map((z) => {
      const xs = z.polygon.map((p) => p[0] * W), ys = z.polygon.map((p) => p[1] * H);
      const cx = xs.reduce((a, v) => a + v, 0) / xs.length, cy = ys.reduce((a, v) => a + v, 0) / ys.length;
      return `<text class="timer" x="${cx.toFixed(1)}" y="${cy.toFixed(1)}">${pos(m.t - z.since)}</text>`;
    }).join('');
    const taken = shown.filter((z) => z.taken).length;
    const keys = focus
      ? `<span class="pf-key"><i class="k-taken"></i>${esc(shown[0]?.name || '')}</span><span class="pf-key"><i class="k-seen"></i>${m.seen} people seen</span>`
      : `<span class="pf-key"><i class="k-taken"></i>${taken} of ${shown.length} taken, with how long</span>`;
    return { W, H, svg: polys + timers, keys };
  }
  const seen = m.people.filter((p) => p.seen), hidden = m.people.filter((p) => !p.seen);
  const keys = `<span class="pf-key"><i class="k-seen"></i>${seen.length} seen in this frame</span>`
    + (hidden.length ? `<span class="pf-key"><i class="k-hidden"></i>${hidden.length} hidden for a moment</span>` : '')
    + `<span class="pf-sum">${m.people.length} counted</span>`;
  return { W, H, svg: boxes('seen', seen) + boxes('hidden', hidden), keys };
}

/* ------------------------------------------------------------- cards */

function stat(s) {
  if (!s.stat) return `<p class="pf-line">${esc(s.headline)}</p>`;
  return `<div class="pf-stat"><b>${esc(s.stat.value)}</b><span>${esc(s.stat.label)}</span></div>
    ${s.sub ? `<p class="pf-sub">${esc(s.sub)}</p>` : ''}`;
}

function more(s, c) {
  const rest = (s.do || []).filter((d) => d !== s.tip);
  return `<details class="pf-more"><summary>Why and how</summary>
    <p>${esc(s.headline)}</p>
    ${s.explain ? `<p>${esc(s.explain)}</p>` : ''}
    ${s.why ? `<p><b>${esc(c ? c.label : 'Note')}:</b> ${esc(s.why)}</p>` : ''}
    ${rest.length ? `<ul>${rest.map((d) => `<li>${esc(d)}</li>`).join('')}</ul>` : ''}
    ${numbers(s.numbers)}
  </details>`;
}

function card(s, run, doc) {
  const c = CONF[s.confidence];
  const ev = proof(s.proof, run, doc);
  const chart = s.id === 'busy' ? strip(s.series, s.proof?.t ?? null) : s.id === 'arrivals' ? strip(s.series)
    : s.shares ? stayBar(s.shares) : s.id === 'who' ? genderBar(doc) : '';
  return `<article class="card pf${ev ? '' : ' pf-solo'}" id="guide-${esc(s.id)}">
    ${ev ? `<div class="pf-proof">${ev}</div>` : ''}
    <div class="pf-body">
      <header class="pf-top">${ico(s.id)}<h3>${esc(s.title)}</h3>
        ${c ? `<span class="g-conf g-${esc(s.confidence)}" title="${esc(s.brief || c.note)}"><i aria-hidden="true"></i>${esc(c.label)}</span>` : ''}</header>
      ${stat(s)}
      ${chart}
      ${s.tip ? `<p class="pf-tip"><svg viewBox="0 0 18 18" aria-hidden="true">${ICON.tip}</svg><span>${esc(s.tip)}</span></p>` : ''}
      ${more(s, c)}
    </div>
  </article>`;
}

function weakCard(list, run, doc) {
  const ev = list.map((s) => proof(s.proof, run, doc)).filter(Boolean).join('');
  // Handoffs arrive later and may be none: their column opens when they do.
  const later = list.every((s) => !s.proof || s.proof.kind === 'handoffs');
  return `<article class="card pf pf-warn${ev && !later ? '' : ' pf-solo'}" id="guide-unreliable">
    ${ev ? `<div class="pf-proof"${later ? ' hidden' : ''}>${ev}</div>` : ''}
    <div class="pf-body">
      <header class="pf-top">${ico('warn')}<h3>Not reliable in this recording</h3></header>
      <ul class="pf-unrel">${list.map((s) => `<li><s>${esc(s.stat?.value ?? '?')}</s><b>${esc(s.title)}</b><span>${esc(s.brief || s.why || '')}</span></li>`).join('')}</ul>
      ${list[0].tip ? `<p class="pf-tip"><svg viewBox="0 0 18 18" aria-hidden="true">${ICON.tip}</svg><span>${esc(list[0].tip)}</span></p>` : ''}
      <details class="pf-more"><summary>Why and how</summary>
        ${list.map((s) => `<p><b>${esc(s.title)}.</b> ${esc(s.headline)} ${esc(s.why || '')}</p>`).join('')}
      </details>
    </div>
  </article>`;
}

function mindCard(s) {
  return `<section class="card pf-mind" id="guide-trust">
    <header class="pf-top">${ico('trust')}<h3>Keep in mind</h3></header>
    <ul>${(s.do || []).map((d) => `<li>${esc(d)}</li>`).join('')}</ul>
  </section>`;
}

/* ------------------------------------------------------------- view */

function enlarge(shot) {
  let dlg = document.getElementById('pf-dialog');
  if (!dlg) {
    dlg = document.createElement('dialog');
    dlg.id = 'pf-dialog';
    dlg.className = 'pf-dialog';
    dlg.setAttribute('aria-label', 'Frame, enlarged');
    dlg.addEventListener('click', () => dlg.close());
    document.body.append(dlg);
  }
  const big = shot.cloneNode(true);
  big.removeAttribute('role');
  big.removeAttribute('tabindex');
  const img = big.querySelector('img');
  img.loading = 'eager';
  img.src = img.src.replace(/w=\d+/, 'w=1600');
  dlg.replaceChildren(big);
  dlg.showModal();
}

export function createGuideView(root, { onSeek = () => {} } = {}) {
  let token = 0;

  async function hydrate(run, doc, mine) {
    const api = (path) => fetch(`/api/runs/${encodeURIComponent(run)}/${path}`).then((r) => (r.ok ? r.json() : Promise.reject(r)));
    root.querySelectorAll('.pf-shot img').forEach((img) => {
      img.addEventListener('error', () => img.closest('.pf-fig')?.classList.add('pf-noimg'), { once: true });
      // The enlarged frame is fetched now, so opening it is instant.
      new Image().src = img.src.replace(/w=\d+/, 'w=1600');
    });
    root.querySelectorAll('.pf-frame').forEach(async (fig) => {
      try {
        const m = await api(`moment?t=${fig.dataset.t}`);
        if (mine !== token) return;
        const o = overlay(m, fig.dataset.show, fig.dataset.focus);
        const svg = fig.querySelector('.pf-over');
        svg.setAttribute('viewBox', `0 0 ${o.W} ${o.H}`);
        svg.setAttribute('preserveAspectRatio', 'none');
        svg.innerHTML = o.svg;
        fig.querySelector('.pf-shot').style.aspectRatio = `${o.W} / ${o.H}`;
        fig.querySelector('.pf-keys').innerHTML = o.keys;
      } catch {
        if (mine === token) fig.querySelector('.pf-keys').innerHTML = '<span class="pf-wait">The boxes need the dashboard server.</span>';
      }
    });
    root.querySelectorAll('.pf-handoffs').forEach(async (fig) => {
      try {
        const h = await api('handoffs');
        if (mine !== token || !h.count) return;
        fig.querySelector('.pf-pairs').innerHTML = handoffRows(h, doc);
        fig.querySelector('.pf-cap').textContent = `${h.count} times a track ended and a new one began in the same spot `
          + `within ${h.max_gap_sec} s: most likely the same person, counted again. The clearest ${h.examples.length}:`;
        fig.hidden = false;
        fig.closest('.pf-proof').hidden = false;
        fig.closest('.pf')?.classList.remove('pf-solo');
      } catch { /* no track boxes: nothing to show */ }
    });
    const ids = [...new Set([...root.querySelectorAll('[data-portrait]')].map((i) => i.dataset.portrait))];
    if (!ids.length) return;
    try {
      const found = await api(`portraits?tracks=${ids.join(',')}`);
      if (mine !== token) return;
      root.querySelectorAll('[data-portrait]').forEach((img) => {
        const hit = found[img.dataset.portrait];
        if (hit) img.src = hit.url; else img.closest('.ph').classList.add('none');
      });
    } catch {
      root.querySelectorAll('[data-portrait]').forEach((img) => img.closest('.ph').classList.add('none'));
    }
  }

  function render(g, run, doc, mine) {
    if (!g) {
      root.innerHTML = '<p class="sub">No guide for this recording.</p>';
      return;
    }
    // Seats and tables are not part of this panel.
    const seatless = (text) => !/\b(seats?|tables?)\b/i.test(text);
    const head = document.getElementById('ins-headline');
    if (head && g.headline) head.textContent = g.headline.replace(/; \d+ of \d+ (seats|tables|places) were taken at the end/, '');
    const setting = document.getElementById('ins-setting');
    if (setting) {
      setting.textContent = g.setting?.label || '';
      setting.removeAttribute('title');
      setting.hidden = !g.setting?.label;
    }
    const sections = (g.sections || []).filter((s) => s.id !== 'seats' && s.id !== 'queues')
      .map((s) => (s.id === 'trust' ? { ...s, do: (s.do || []).filter(seatless) } : s));
    g = { ...g, glossary: (g.glossary || []).filter((x) => seatless(`${x.term} ${x.means}`) && !/turnover|utilisation/i.test(x.term)) };
    const strong = sections.filter((s) => s.id !== 'trust' && s.confidence !== 'unreliable');
    const weak = sections.filter((s) => s.confidence === 'unreliable');
    const trust = sections.find((s) => s.id === 'trust');
    root.innerHTML = `<div class="pf-list">
        ${strong.map((s) => card(s, run, doc)).join('')}
        ${weak.length ? weakCard(weak, run, doc) : ''}
        ${trust ? mindCard(trust) : ''}
      </div>
      <details class="card pf-gloss"><summary>Words used here</summary>
        <dl>${(g.glossary || []).map((x) => `<dt>${esc(x.term)}</dt><dd>${esc(x.means)}</dd>`).join('')}</dl>
      </details>`;
    hydrate(run, doc, mine);
  }

  root.addEventListener('click', (e) => {
    const seekBtn = e.target.closest('[data-seek]');
    if (seekBtn) { onSeek(Number(seekBtn.dataset.seek)); return; }
    const shot = e.target.closest('.pf-shot');
    if (shot) enlarge(shot);
  });
  root.addEventListener('keydown', (e) => {
    const shot = e.target.closest?.('.pf-shot');
    if (shot && (e.key === 'Enter' || e.key === ' ')) { e.preventDefault(); enlarge(shot); }
  });

  return {
    async setRun(id, doc) {
      const mine = ++token;
      // Guides stored before proofs existed (schema 1) are written again.
      if (doc?.guide?.schema >= 2) return render(doc.guide, id, doc, mine);
      root.innerHTML = '<p class="sub">Writing the guide</p>';
      try {
        const res = await fetch(`/api/runs/${encodeURIComponent(id)}/guide`);
        if (!res.ok) throw new Error(res.statusText);
        const g = await res.json();
        if (mine === token) render(g, id, doc, mine);
      } catch {
        if (mine === token) render(doc?.guide || null, id, doc, mine);
      }
    },
  };
}
