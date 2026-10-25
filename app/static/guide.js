/* The Guide view: the recording explained in plain words for whoever runs the
   place. The text is written server-side (src/inference/guide.py) and stored
   with the run, so every sentence comes with the numbers it rests on; this
   only lays it out. Runs processed before the guide existed get one from
   GET /api/runs/{id}/guide. */

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
};

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

function numbers(obj) {
  const rows = Object.entries(obj || {}).filter(([k, v]) => NUMBER_NAMES[k] && v !== null && v !== undefined && typeof v !== 'object');
  if (!rows.length) return '';
  const fmt = (k, v) => (SHARE.has(k) && typeof v === 'number' ? `${Math.round(v * 100)}%`
    : typeof v === 'number' ? (Number.isInteger(v) ? v.toLocaleString() : v.toFixed(1)) : esc(v));
  return `<details class="g-num"><summary>Numbers behind this</summary><dl>${
    rows.map(([k, v]) => `<dt>${esc(NUMBER_NAMES[k])}</dt><dd>${fmt(k, v)}</dd>`).join('')}</dl></details>`;
}

/* A strip of bars for one series over the recording: one colour, no legend
   (the caption names it), the peak labelled, every bar with a tooltip. */
function strip(series) {
  const pts = series?.points || [];
  if (pts.length < 2) return '';
  const max = Math.max(...pts.map((p) => p.v), 0) || 1;
  const W = 600, H = 64, gap = 2;
  const bw = Math.max(1, (W - gap * (pts.length - 1)) / pts.length);
  const peak = pts.reduce((a, p) => (p.v > a.v ? p : a), pts[0]);
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
      <text x="${px.toFixed(1)}" y="13" text-anchor="${anchor}">${Math.round(peak.v)} ${esc(series.unit)}</text>
    </svg>
    <figcaption><span>${esc(pts[0].clock)}</span><span>${esc(series.label)}</span><span>${esc(pts[pts.length - 1].clock)}</span></figcaption>
  </figure>`;
}

function section(s) {
  const c = CONF[s.confidence];
  const icon = ICON[s.id] || ICON.trust;
  return `<article class="card g-card${s.id === 'trust' ? ' g-trust' : ''}" id="guide-${esc(s.id)}">
    <header>
      <span class="g-ico"><svg viewBox="0 0 18 18" aria-hidden="true">${icon}</svg></span>
      <h3>${esc(s.title)}</h3>
      ${c ? `<span class="g-conf g-${esc(s.confidence)}" title="${esc(c.note)}"><i aria-hidden="true"></i>${esc(c.label)}</span>` : ''}
    </header>
    <p class="g-head">${esc(s.headline)}</p>
    ${strip(s.series)}
    <p class="g-explain">${esc(s.explain)}</p>
    ${s.do?.length ? `<div class="g-do"><h4>${s.id === 'trust' ? 'Keep in mind' : 'What to try'}</h4><ul>${s.do.map((d) => `<li>${esc(d)}</li>`).join('')}</ul></div>` : ''}
    ${s.why ? `<p class="g-why"><b>${c ? esc(c.label) : 'Note'}:</b> ${esc(s.why)}</p>` : ''}
    ${numbers(s.numbers)}
  </article>`;
}

export function createGuideView(root) {
  let token = 0;
  function render(g) {
    if (!g) {
      root.innerHTML = '<p class="sub">No guide for this recording.</p>';
      return;
    }
    const sections = g.sections || [];
    root.innerHTML = `
      <div class="card g-setting">
        <p class="g-kicker">What kind of place this looks like</p>
        <p class="g-setting-text">${esc(g.setting?.text)}</p>
        <nav class="g-toc" aria-label="Guide sections">${sections.map((s) => `<a href="#guide-${esc(s.id)}" data-jump="${esc(s.id)}">${esc(s.title)}</a>`).join('')}</nav>
      </div>
      <div class="g-grid">${sections.map(section).join('')}</div>
      <div class="card g-gloss">
        <header><h3>Words used here</h3></header>
        <dl>${(g.glossary || []).map((x) => `<dt>${esc(x.term)}</dt><dd>${esc(x.means)}</dd>`).join('')}</dl>
      </div>`;
    root.querySelectorAll('[data-jump]').forEach((a) => a.addEventListener('click', (e) => {
      e.preventDefault();
      root.querySelector(`#guide-${a.dataset.jump}`)?.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }));
  }
  return {
    async setRun(id, doc) {
      const mine = ++token;
      if (doc?.guide) return render(doc.guide);
      root.innerHTML = '<p class="sub">Writing the guide</p>';
      try {
        const res = await fetch(`/api/runs/${encodeURIComponent(id)}/guide`);
        if (!res.ok) throw new Error(res.statusText);
        const g = await res.json();
        if (mine === token) render(g);
      } catch {
        if (mine === token) render(null);
      }
    },
  };
}
