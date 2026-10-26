/* The Zones view: tables, seats and queues drawn on the camera frame.

   Everything is read from the `zones` block of analytics.json (built by
   src/inference/zones.py) and evaluated at the page's one playhead, so the
   camera overlay, the status board, the timeline and the "next to free up"
   card always agree with each other and with the other tabs.

   The overlay is a canvas over the recording. Only taken zones are drawn,
   red with a timer (a free one shows while pointed at or followed); the zone taken longest gets pulsing rings,
   because it is the best single guess for the next one to free up (the
   Validity card says how good a guess that was in this recording).

   Zones are edited here too: draw polygons or boxes, drag corners, or let the
   server propose seats from where people sat still. Saving recomputes the
   analysis from the run's track_boxes.json - no model is run again. */

import { cssVar } from './three-kit.js';

const KIND_PREFIX = { seat: 'S', table: 'T', queue: 'Q', area: 'A' };
const KIND_NAME = { seat: 'Seat', table: 'Table', queue: 'Queue', area: 'Area' };
const SEATED = (z) => z.kind === 'seat' || z.kind === 'table';

export function createZonesView({ $, $$, C, esc, getState, seek, setPlayhead, onAnalysis }) {
  const { fmt } = C;
  const cam = $('#zcam');
  const canvas = $('#zcam-overlay');
  const ctx = canvas.getContext('2d');
  const video = $('#zcam-video');
  const still = $('#zcam-still');
  let doc = null, runId = null, run = null;
  let Z = null;                  // the zones block
  let raf = 0, visible = false, lastPush = 0;
  let hover = null, focus = null;
  const opt = { timers: true, names: false };
  let edit = null;               // { zones, sel, mode, draft, drag, history }
  let preview = null;            // analysis under unsaved settings
  let labels = null;             // { annotator: doc } ground truth, for the Validity card

  /* ------------------------------------------------------------ helpers */
  const now = () => getState().playhead;
  const dur = () => doc?.meta.duration_sec || 0;
  const block = () => preview || Z;
  const colorVars = () => ({
    free: cssVar('--zone-free'), taken: cssVar('--zone-taken'), next: cssVar('--zone-next'),
    queue: cssVar('--zone-queue'), ink: cssVar('--text-1'), plane: cssVar('--surface'),
  });
  const wall = (t) => {
    const at = doc?.meta.captured_at ? new Date(doc.meta.captured_at) : null;
    if (!at || Number.isNaN(+at)) return null;
    return new Date(+at + t * 1000).toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit', second: '2-digit' });
  };
  const clockFor = (s) => (s >= 3600 ? `${Math.floor(s / 3600)}:${fmt.clock(s % 3600).padStart(5, '0')}` : fmt.clock(s));

  /* Where every zone stands at time t. */
  function statusAt(t, b = block()) {
    if (!b) return [];
    return b.zones.map((z) => {
      const ses = z.sessions;
      const cur = ses.find((s) => s.start <= t && t < s.end + 1e-6);
      const prev = [...ses].reverse().find((s) => s.end <= t);
      const people = z.kind === 'queue' && z.queue
        ? z.queue.waits.filter((w) => w.start <= t && t < w.end) : null;
      return {
        z, cur, prev,
        taken: !!cur,
        since: cur ? t - cur.start : (prev ? t - prev.end : t),
        lowerBound: cur ? cur.censored_start : !prev,
        waiting: people,
      };
    });
  }

  function nextFree(st) {
    const taken = st.filter((s) => s.taken && SEATED(s.z));
    if (taken.length < 1) return null;
    return taken.reduce((a, b) => (b.since > a.since ? b : a));
  }

  /* Expected remaining time for a session already `elapsed` long, from the
     Kaplan-Meier curve stored with the analysis (restricted to its last step). */
  function expectedLeft(elapsed, km = block()?.survival) {
    if (!km || km.t.length < 2) return null;
    const { t, s } = km;
    const horizon = t[t.length - 1];
    if (elapsed >= horizon) return null;
    const S = (x) => { let v = 1; for (let i = 0; i < t.length && t[i] <= x; i++) v = s[i]; return v; };
    const se = S(elapsed);
    if (se <= 0) return 0;
    let area = 0, prev = elapsed;
    for (const x of [...t.filter((v) => v > elapsed && v < horizon), horizon]) {
      area += S(prev) * (x - prev);
      prev = x;
    }
    return area / se;
  }

  const fmtSince = (s) => `${s.lowerBound ? '≥' : ''}${clockFor(Math.max(0, s.since))}`;

  /* ------------------------------------------------------------ camera */
  function layout() {
    if (!doc) return;
    const W = doc.meta.width || 16, H = doc.meta.height || 9;
    cam.style.aspectRatio = `${W} / ${H}`;
    const r = cam.getBoundingClientRect();
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    canvas.width = Math.round(r.width * dpr);
    canvas.height = Math.round(r.height * dpr);
    canvas.style.width = `${r.width}px`;
    canvas.style.height = `${r.height}px`;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    draw();
  }
  new ResizeObserver(() => layout()).observe(cam);

  const toPx = ([x, y]) => [x * canvas.clientWidth, y * canvas.clientHeight];
  const toNorm = (e) => {
    const r = canvas.getBoundingClientRect();
    return [Math.min(1, Math.max(0, (e.clientX - r.left) / r.width)),
      Math.min(1, Math.max(0, (e.clientY - r.top) / r.height))];
  };
  const centroid = (poly) => {
    let a = 0, cx = 0, cy = 0;
    poly.forEach(([x1, y1], i) => {
      const [x2, y2] = poly[(i + 1) % poly.length];
      const c = x1 * y2 - x2 * y1;
      a += c; cx += (x1 + x2) * c; cy += (y1 + y2) * c;
    });
    if (Math.abs(a) < 1e-9) return poly.reduce((p, q) => [p[0] + q[0] / poly.length, p[1] + q[1] / poly.length], [0, 0]);
    return [cx / (3 * a), cy / (3 * a)];
  };
  const inside = ([x, y], poly) => {
    let hit = false;
    for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
      const [xi, yi] = poly[i], [xj, yj] = poly[j];
      if ((yi > y) !== (yj > y) && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) hit = !hit;
    }
    return hit;
  };
  const tracePath = (poly) => {
    ctx.beginPath();
    poly.forEach((p, i) => { const [x, y] = toPx(p); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
    ctx.closePath();
  };
  /* A rounded label. With `placed`, it moves above, below or beside its spot
     rather than cover a label already drawn - a full room has seats close
     enough for their labels to collide. An `optional` label with nowhere clear
     to go is left out; the zone's colour and the board still say its state. */
  function chip(text, x, y, bg, fg, { strong = false, dense = false, placed = null, optional = false } = {}) {
    const size = strong ? (dense ? 12 : 13) : (dense ? 10.5 : 11.5);
    ctx.font = `${strong ? 700 : 600} ${size}px Inter, system-ui, sans-serif`;
    const w = ctx.measureText(text).width + (dense ? 9 : 12), h = strong ? (dense ? 19 : 22) : (dense ? 16 : 19);
    const fit = (cx, cy) => [Math.max(2, Math.min(canvas.clientWidth - w - 2, cx - w / 2)),
      Math.max(2, Math.min(canvas.clientHeight - h - 2, cy - h / 2))];
    let [x0, y0] = fit(x, y);
    if (placed) {
      const clear = ([ax, ay]) => !placed.some(([bx, by, bw, bh]) => ax < bx + bw + 2 && bx < ax + w + 2 && ay < by + bh + 2 && by < ay + h + 2);
      const tries = [[x, y], [x, y - h - 3], [x, y + h + 3], [x - w * 0.6, y], [x + w * 0.6, y],
        [x, y - 2 * (h + 3)], [x, y + 2 * (h + 3)]].map(([cx, cy]) => fit(cx, cy));
      const spot = tries.find(clear);
      if (!spot && optional) return;
      [x0, y0] = spot || tries[0];
      placed.push([x0, y0, w, h]);
    }
    ctx.fillStyle = bg;
    ctx.beginPath();
    ctx.roundRect(x0, y0, w, h, h / 2);
    ctx.fill();
    ctx.fillStyle = fg;
    ctx.textBaseline = 'middle';
    ctx.textAlign = 'left';
    ctx.fillText(text, x0 + (dense ? 4.5 : 6), y0 + h / 2 + 0.5);
  }

  function draw() {
    const W = canvas.clientWidth, H = canvas.clientHeight;
    ctx.clearRect(0, 0, W, H);
    if (!doc) return;
    const col = colorVars();
    const t = edit ? null : (video && !video.paused ? video.currentTime : now());
    if (edit) return drawEditor(col);
    const b = block();
    if (!b) return;
    const all = statusAt(t, b);
    const nxt = nextFree(all);
    // Free seats are left off the picture: what is taken, and for how long,
    // is what the overlay is for. One pointed at or followed still shows.
    const st = all.filter((s) => s.taken || s.z.kind === 'queue' || hover === s.z.id || focus === s.z.id);
    const pulse = (performance.now() % 1800) / 1800;
    st.forEach((s) => {
      const { z } = s;
      const c = z.kind === 'queue' ? col.queue : (s.taken ? col.taken : col.free);
      const lit = hover === z.id || focus === z.id;
      tracePath(z.polygon);
      ctx.fillStyle = c;
      ctx.globalAlpha = z.kind === 'queue' ? 0.14 : (s.taken ? (lit ? 0.34 : 0.22) : (lit ? 0.26 : 0.14));
      ctx.fill();
      ctx.globalAlpha = 1;
      ctx.lineWidth = lit ? 3 : 2;
      ctx.strokeStyle = c;
      ctx.stroke();
    });
    // Rings around the zone most likely to free up next.
    if (nxt) {
      const [cx, cy] = toPx(centroid(nxt.z.polygon));
      const base = Math.max(14, Math.min(46, Math.sqrt(polyAreaPx(nxt.z.polygon)) * 0.5));
      for (let k = 0; k < 3; k++) {
        const f = (pulse + k / 3) % 1;
        ctx.beginPath();
        ctx.ellipse(cx, cy, base + f * base * 1.6, (base + f * base * 1.6) * 0.62, 0, 0, Math.PI * 2);
        ctx.strokeStyle = col.next;
        ctx.globalAlpha = (1 - f) * 0.85;
        ctx.lineWidth = 2.5;
        ctx.stroke();
      }
      ctx.globalAlpha = 1;
    }
    // Labels last, so no polygon covers one; the next-to-free label first, so
    // it keeps its place and the others move around it.
    const small = W < 560;
    const dense = st.length > 12 && W < 1100;
    // A full room (a classroom has forty-odd seats) cannot carry a name and a
    // timer on every seat: there, labels are the timer alone, and only the
    // seat under the pointer or being followed, and the next to free up, keep
    // their names.
    const crowded = st.length * 13000 > W * H;
    const placed = [];
    const lit = (s) => hover === s.z.id || focus === s.z.id;
    const isNext = (s) => nxt && nxt.z.id === s.z.id;
    const first = st.filter((s) => isNext(s) || lit(s));
    const order = [...first, ...st.filter((s) => !first.includes(s))];
    order.forEach((s) => {
      const { z } = s;
      const [cx, cy] = toPx(centroid(z.polygon));
      const c = z.kind === 'queue' ? col.queue : (s.taken ? col.taken : col.free);
      const name = opt.names ? z.name : z.id;
      const full = !crowded || isNext(s) || lit(s);
      let text = name;
      if (z.kind === 'queue') text = `${name} · ${s.waiting?.length || 0} waiting`;
      else if (opt.timers && !small) {
        const state = s.taken ? fmtSince(s) : 'Free';
        text = full ? `${name} ${state}` : state;
      }
      chip(text, cx, cy, isNext(s) ? col.next : c, '#fff',
        { strong: isNext(s), dense: dense || crowded, placed, optional: crowded && !full });
    });
  }
  function polyAreaPx(poly) {
    const pts = poly.map(toPx);
    let a = 0;
    pts.forEach(([x1, y1], i) => { const [x2, y2] = pts[(i + 1) % pts.length]; a += x1 * y2 - x2 * y1; });
    return Math.abs(a) / 2;
  }

  /* Animate while the camera is visible: the rings pulse and a playing video
     moves the clock. Stops when the tab is hidden. */
  function loop() {
    raf = 0;
    if (!visible) return;
    if (video && !video.paused && !edit) {
      const t = video.currentTime;
      if (performance.now() - lastPush > 200) {
        lastPush = performance.now();
        setPlayhead(t, { fromZones: true });
      }
      renderClock(t);
    }
    draw();
    raf = requestAnimationFrame(loop);
  }
  const startLoop = () => { if (!raf && visible) raf = requestAnimationFrame(loop); };

  function renderClock(t = now()) {
    const w = wall(t);
    $('#zcam-clock').textContent = w || clockFor(t);
    $('#zcam-time').textContent = `${clockFor(t)} / ${clockFor(dur())}`;
    const r = $('#zcam-scrub');
    if (!r.matches(':active')) r.value = Math.round((dur() ? t / dur() : 0) * 1000);
  }

  /* Hover: what a zone is doing right now. */
  canvas.addEventListener('pointermove', (e) => {
    if (edit) return onEditMove(e);
    const p = toNorm(e);
    const b = block();
    if (!b) return;
    const hit = [...b.zones].reverse().find((z) => inside(p, z.polygon));
    const id = hit?.id || null;
    if (id !== hover) { hover = id; draw(); }
    if (!hit) { C.hideTip(); canvas.style.cursor = ''; return; }
    canvas.style.cursor = 'pointer';
    const s = statusAt(now(), b).find((x) => x.z.id === hit.id);
    const m = hit.metrics;
    C.showTip(e, `${esc(hit.name)} (${hit.id})`, [
      { label: 'Now', value: hit.kind === 'queue' ? `${s.waiting?.length || 0} waiting` : (s.taken ? `Taken ${fmtSince(s)}` : `Free ${fmtSince(s)}`),
        color: s.taken ? 'var(--zone-taken)' : 'var(--zone-free)' },
      { label: 'Taken overall', value: fmt.pct(m.utilization * 100, 0) },
      { label: 'Sessions', value: fmt.int(m.sessions) },
      { label: 'People', value: fmt.int(m.people) },
    ]);
  });
  canvas.addEventListener('pointerleave', () => { if (!edit) { hover = null; C.hideTip(); draw(); } });
  canvas.addEventListener('click', (e) => {
    if (edit) return;
    const b = block();
    const hit = b && [...b.zones].reverse().find((z) => inside(toNorm(e), z.polygon));
    focus = hit && focus !== hit.id ? hit.id : null;
    renderBoard(); renderGantt(); draw();
  });

  /* ------------------------------------------------------- media source */
  function setSource() {
    const hasVideo = !!(run?.has_video || doc?.meta.has_video);
    const frames = doc?.meta.frames || [];
    video.hidden = !hasVideo || !!edit;
    still.hidden = hasVideo && !edit;
    if (hasVideo && video.dataset.src !== runId) {
      video.dataset.src = runId;
      // The clean copy when there is one: boxes burned into the annotated
      // video would fight the zone overlay.
      const file = run?.has_clean_video ? 'clean.mp4' : 'video.mp4';
      video.src = `/api/runs/${encodeURIComponent(runId)}/${file}`;
    }
    if (!hasVideo || edit) {
      // A clean frame (no boxes drawn) is the right background for drawing zones.
      const t = now();
      const f = frames.length ? frames.reduce((a, b2) => (Math.abs(b2.t - t) < Math.abs(a.t - t) ? b2 : a)) : null;
      still.src = f ? `/api/runs/${encodeURIComponent(runId)}/${f.file}`
        : (run?.has_poster ? `/api/runs/${encodeURIComponent(runId)}/poster.jpg` : '');
    }
    $('#zcam-play').disabled = !hasVideo;
  }

  /* -------------------------------------------------------------- panels */
  function renderEmpty() {
    const hasBoxes = !!run?.has_boxes;
    const server = getState().runs.length > 0;
    $('#zones-empty').hidden = !!Z;
    $$('.zones-body').forEach((n) => { n.hidden = !Z; });
    if (Z) return;
    $('#zones-empty-text').innerHTML = !server
      ? 'Zones need the server: start it with <code>python -m app.server</code>.'
      : hasBoxes
        ? 'No zones yet. Draw the tables or seats on the camera frame, or let IFAS find seats from where people sat still.'
        : 'This recording was processed before zones existed. Process it again to measure zones: the pipeline now keeps every box it sees, so zones can be drawn afterwards.';
    $('#zones-empty-draw').hidden = !(server && hasBoxes);
    $('#zones-empty-auto').hidden = !(server && hasBoxes);
  }

  function renderBoard(t = now()) {
    const b = block();
    if (!b) return;
    const st = statusAt(t, b);
    const seated = st.filter((s) => SEATED(s.z));
    const taken = seated.filter((s) => s.taken).length;
    const kinds = [...new Set(b.zones.map((z) => z.kind))];
    $('#zboard-title').textContent = kinds.length === 1 ? `${KIND_NAME[kinds[0]]}s` : 'Zones';
    $('#zboard-count').innerHTML = seated.length
      ? `<b>${taken}</b> taken · <b>${seated.length - taken}</b> free` : '';
    const D = dur() || 1;
    const nxt = nextFree(st);
    $('#zboard-list').innerHTML = st.map((s) => {
      const z = s.z;
      const color = z.kind === 'queue' ? 'var(--zone-queue)' : s.taken ? 'var(--zone-taken)' : 'var(--zone-free)';
      const state = z.kind === 'queue' ? `${s.waiting?.length || 0} waiting`
        : s.taken ? `Taken ${fmtSince(s)}` : `Free ${fmtSince(s)}`;
      const segs = z.sessions.map((x) =>
        `<i style="left:${(x.start / D) * 100}%;width:${Math.max(0.6, ((x.end - x.start) / D) * 100)}%"></i>`).join('');
      return `<button class="zrow${focus === z.id ? ' on' : ''}${nxt?.z.id === z.id ? ' next' : ''}" data-zone="${esc(z.id)}">
        <i class="zdot" style="background:${color}"></i>
        <span class="zid">${esc(z.id)}</span>
        <span class="zstate">${state}</span>
        <span class="zstrip">${segs}<b style="left:${(t / D) * 100}%"></b></span>
      </button>`;
    }).join('');
    $$('#zboard-list .zrow').forEach((r) => {
      r.onclick = () => { focus = focus === r.dataset.zone ? null : r.dataset.zone; renderBoard(); renderGantt(); draw(); };
      r.onmouseenter = () => { hover = r.dataset.zone; draw(); };
      r.onmouseleave = () => { hover = null; draw(); };
    });
  }

  function renderNext(t = now()) {
    const b = block();
    const host = $('#znext');
    if (!b) return;
    const st = statusAt(t, b);
    const nxt = nextFree(st);
    const taken = st.filter((s) => s.taken && SEATED(s.z)).length;
    const ev = b.next_free?.evaluation || {};
    if (!nxt) {
      host.innerHTML = `<p class="empty">${st.some((s) => SEATED(s.z)) ? 'Nothing is taken at this moment.' : 'No seats or tables here, only queues and areas.'}</p>`;
      return;
    }
    const left = expectedLeft(nxt.since);
    const km = b.survival;
    host.innerHTML = `
      <div class="znext">
        <svg class="rings" viewBox="0 0 120 120" aria-hidden="true">
          <circle cx="60" cy="60" r="20"/><circle cx="60" cy="60" r="20"/><circle cx="60" cy="60" r="20"/>
        </svg>
        <div class="znext-id"><b>${esc(nxt.z.id)}</b><span>${fmtSince(nxt)}</span></div>
      </div>
      <p class="znext-why">${esc(nxt.z.name)} has been taken longest, of ${taken} taken ${taken === 1 ? 'zone' : 'zones'}${nxt.lowerBound ? ' (since before the recording began)' : ''}.
        ${left != null ? ` Kaplan–Meier puts the remaining time at about <b>${fmt.dur(left)}</b>, from ${km.events} finished ${km.events === 1 ? 'session' : 'sessions'}.` : ''}</p>
      ${ev.moments ? `<p class="znext-eval">In this recording, “longest taken” named the next zone to free up <b>${fmt.pct(ev.longest_seated * 100, 0)}</b> of the time (chance ${fmt.pct(ev.random * 100, 0)}, ${ev.moments} moments).</p>` : ''}
      ${rankLists(st)}`;
    $$('#znext [data-zone]').forEach((b) => {
      b.onclick = () => { focus = focus === b.dataset.zone ? null : b.dataset.zone; renderBoard(); renderGantt(); draw(); };
    });
  }

  /* The runners-up, and what is free right now: the queue a host would read. */
  function rankLists(st) {
    const taken = st.filter((s) => s.taken && SEATED(s.z)).sort((a, b2) => b2.since - a.since);
    const free = st.filter((s) => !s.taken && SEATED(s.z)).sort((a, b2) => b2.since - a.since);
    const row = (s, i, right) => `<button class="zrank-row" data-zone="${esc(s.z.id)}"><span class="zrank-i">${i}</span><b>${esc(s.z.id)}</b><span>${right}</span></button>`;
    const next = taken.slice(1, 5).map((s, i) => {
      const left = expectedLeft(s.since);
      return row(s, i + 2, `${fmtSince(s)}${left != null ? ` · ~${fmt.dur(left)} left` : ''}`);
    }).join('');
    return `<div class="zrank">
      ${next ? `<div><p class="zrank-h">Then</p>${next}</div>` : ''}
      <div><p class="zrank-h">Free now${free.length ? ` (${free.length})` : ''}</p>${free.length
        ? `<div class="zfree">${free.slice(0, 12).map((s) => `<button class="zfree-chip" data-zone="${esc(s.z.id)}"><b>${esc(s.z.id)}</b> ${fmtSince(s)}</button>`).join('')}</div>`
        : '<p class="sub">Every seat is taken.</p>'}</div>
    </div>`;
  }

  function renderQueue(t = now()) {
    const b = block();
    const qs = (b?.zones || []).filter((z) => z.kind === 'queue');
    $('#zqueue-card').hidden = !qs.length;
    if (!qs.length) return;
    const rows = qs.flatMap((z) => (z.queue?.waits || []).filter((w) => w.start <= t && t < w.end)
      .map((w) => ({ z, w, waited: t - w.start })));
    rows.sort((a, c) => a.w.start - c.w.start);
    const stats = qs.map((z) => `${esc(z.id)}: average wait ${fmt.dur(z.queue.mean_wait_sec)}, longest ${fmt.dur(z.queue.max_wait_sec)}`).join(' · ');
    $('#zqueue').innerHTML = (rows.length
      ? `<ol class="zq">${rows.map((r, i) => `<li><span class="zq-pos">${i + 1}</span><span>Person ${r.w.track}</span><span class="zq-wait">${clockFor(r.waited)}</span></li>`).join('')}</ol>`
      : '<p class="empty">Nobody is waiting at this moment.</p>')
      + `<p class="sub">${stats}</p>`;
  }

  function renderTiles(t = now()) {
    const b = block();
    if (!b) return;
    const st = statusAt(t, b);
    const seated = st.filter((s) => SEATED(s.z));
    const free = seated.filter((s) => !s.taken).length;
    const s = b.summary;
    $('#zk-free').innerHTML = `${free}<small>of ${seated.length}</small>`;
    $('#zk-free-note').textContent = `${seated.length - free} taken at ${clockFor(t)}`;
    const tl = b.timeline.occupied;
    C.sparkline($('#zk-free-spark'), tl.map((v) => seated.length - v), 'var(--zone-free)');
    $('#zk-util').innerHTML = `${fmt.num(s.mean_utilization * 100, 0)}<small>%</small>`;
    const busiest = b.zones.filter(SEATED).reduce((a, z) => (!a || z.metrics.utilization > a.metrics.utilization ? z : a), null);
    $('#zk-util-note').textContent = busiest ? `Busiest: ${busiest.id}, taken ${fmt.pct(busiest.metrics.utilization * 100, 0)} of the time` : '';
    C.sparkline($('#zk-util-spark'), tl, 'var(--zone-taken)');
    $('#zk-sess').textContent = fmt.int(s.sessions);
    $('#zk-sess-note').textContent = s.sessions ? `Median ${fmt.dur(s.median_session_sec)} each` : 'No one sat down';
    $('#zk-flick').innerHTML = `${fmt.num(s.raw_flips_per_min, 1)}<small>→ ${fmt.num(s.smoothed_flips_per_min, 2)}</small>`;
    $('#zk-flick-note').textContent = 'State changes per zone-minute, raw → smoothed';
  }

  function renderGantt() {
    const b = block();
    if (!b) return;
    const rows = b.zones.map((z) => ({
      id: z.id, label: z.id,
      spans: z.sessions.map((s) => ({ ...s, color: z.kind === 'queue' ? 'var(--zone-queue)' : undefined })),
    }));
    C.timelineRows($('#c-zgantt'), {
      rows, duration: dur(), rowH: rows.length > 30 ? 14 : 18, highlight: focus,
      marker: () => now(),
      onSeek: (t2) => seek(t2),
      tipRows: (row, s) => [
        { label: 'Taken for', value: `${s.censored_start || s.censored_end ? '≥' : ''}${fmt.dur(s.duration)}`, color: 'var(--zone-taken)' },
        { label: 'People', value: fmt.int(s.people.length) },
        { label: 'Most at once', value: fmt.int(s.peak) },
        ...(s.censored_start ? [{ label: 'Began', value: 'before the recording' }] : []),
        ...(s.censored_end ? [{ label: 'Ended', value: 'after the recording' }] : []),
      ],
    });
  }

  function renderUtil() {
    const b = block();
    if (!b) return;
    const zs = [...b.zones].filter((z) => z.kind !== 'queue')
      .sort((a, c) => c.metrics.utilization - a.metrics.utilization);
    C.hbars($('#c-zutil'), {
      data: zs.map((z) => ({
        label: z.id, value: z.metrics.utilization * 100, color: 'var(--zone-taken)',
        rows: [{ label: 'Taken', value: fmt.pct(z.metrics.utilization * 100, 0), color: 'var(--zone-taken)' },
          { label: 'Sessions', value: fmt.int(z.metrics.sessions) },
          { label: 'Longest', value: fmt.dur(z.metrics.max_session_sec) }],
      })),
      labelW: 46, rowH: zs.length > 24 ? 20 : 26, format: (v) => fmt.num(v, 0), unit: '%',
    });
  }

  function renderSurvival() {
    const b = block();
    const km = b?.survival;
    if (!km || km.t.length < 2) {
      $('#c-zsurv').innerHTML = `<p class="empty">${km ? `No session both began and ended inside the recording (${km.n} seen, all running past an edge), so there is no curve yet.` : 'No sessions.'}</p>`;
      return;
    }
    const nxt = nextFree(statusAt(now(), b));
    C.stepCurve($('#c-zsurv'), {
      t: km.t, s: km.s, color: 'var(--zone-taken)', height: 190,
      marker: () => (nxt ? nxt.since : null),
    });
    $('#zsurv-note').textContent = `${km.n} sessions, ${km.events} seen from start to finish; the rest began before or ran past the recording and count as at least as long as seen.`;
  }

  const EVENT = {
    occupied: (e) => [`${e.zone} taken`, e.people > 1 ? `${e.people} people` : ''],
    freed: (e) => [`${e.zone} free`, `after ${fmt.dur(e.after)}`],
    long_stay: (e) => [`${e.zone} long stay`, `taken ${fmt.dur(e.lasted)} in all`],
    queue_wait: (e) => [`Long wait in ${e.zone}`, `person ${e.track}, ${fmt.dur(e.wait)}`],
  };
  function renderEvents(t = now()) {
    const b = block();
    if (!b) return;
    const evs = b.events || [];
    const host = $('#zevents');
    if (!evs.length) { host.innerHTML = '<p class="empty">Nothing changed hands in this recording.</p>'; return; }
    host.innerHTML = evs.slice(0, 400).map((e, i) => {
      const [title, note] = (EVENT[e.type] || (() => [e.type, '']))(e);
      return `<button class="zev ${e.type}${e.t > t + 0.01 ? ' later' : ''}" data-i="${i}">
        <span class="zev-t">${clockFor(e.t)}</span><i></i><span class="zev-x"><b>${esc(title)}</b> ${esc(note)}</span></button>`;
    }).join('');
    $$('#zevents .zev').forEach((btn) => { btn.onclick = () => seek(evs[+btn.dataset.i].t); });
  }

  function showOutputs(f) {
    $$('#zparams output').forEach((o) => {
      const v = +f[o.htmlFor].value;
      o.textContent = o.dataset.unit === '%' ? `${Math.round(v * 100)}%` : `${v}s`;
    });
  }

  function renderMethod() {
    const b = block();
    if (!b) return;
    const p = b.params;
    const f = $('#zparams');
    if (!f.dataset.dirty) {
      f.anchor.value = p.anchor;
      ['on_sec', 'off_sec', 'gap_fill_sec', 'long_stay_sec'].forEach((k) => { f[k].value = p[k]; });
      f.stick.value = p.stick ?? 0;
    }
    showOutputs(f);
    $('#zparams-state').innerHTML = preview
      ? '<span class="warn">Preview: these settings are not saved.</span>'
      : `Saved settings. Taken is reported ${p.on_sec}s after someone sits and free ${p.off_sec}s after they leave; the times shown are exact either way.`;
    $('#zparams-save').disabled = !preview;
    $('#zparams-reset').disabled = !preview;
  }

  function renderValidity() {
    const b = block();
    if (!b) return;
    const s = b.summary, ev = b.next_free?.evaluation || {};
    const rows = [
      ['Zones', `${s.zones} (${b.source === 'auto' ? 'found automatically' : 'drawn by hand'})`],
      ['Anchor', { foot: 'Bottom of the box (feet)', lower: `${Math.round(b.params.lower_frac * 100)}% down the box (seated)`, center: 'Centre of the box', overlap: `Lower half ≥ ${Math.round(b.params.min_overlap * 100)}% inside` }[b.params.anchor]],
      ['Sightings in no zone', fmt.pct(s.unassigned_share * 100, 0)],
      ['Flicker', `${fmt.num(s.raw_flips_per_min, 2)} → ${fmt.num(s.smoothed_flips_per_min, 2)} changes per zone-minute`],
      ['Sample interval', `${fmt.num(b.sample_sec, 2)}s`],
      ['Live-alert latency', `${b.latency_sec.occupied}s to taken, ${b.latency_sec.freed}s to free`],
    ];
    $('#zvalid').innerHTML = rows.map(([k, v]) => `<dt>${k}</dt><dd>${esc(v)}</dd>`).join('');
    $('#znf-eval').innerHTML = ev.moments
      ? `<table><thead><tr><th>Rule for “next to free up”</th><th class="num">Right</th></tr></thead><tbody>
          <tr><td>Taken longest so far</td><td class="num">${fmt.pct(ev.longest_seated * 100, 0)}</td></tr>
          <tr><td>Shortest Kaplan–Meier remaining time${ev.km_fallbacks ? ` <span class="muted">(${ev.km_fallbacks} fell back)</span>` : ''}</td><td class="num">${fmt.pct(ev.km_residual * 100, 0)}</td></tr>
          <tr><td>Chance</td><td class="num">${fmt.pct(ev.random * 100, 0)}</td></tr></tbody></table>
         <p class="sub">Checked every ${ev.step_sec}s over ${ev.moments} moments, using only what was known at each moment.</p>`
      : '<p class="sub">Too few zones freed up in this recording to check the next-free rules.</p>';
    renderAgreement();
  }

  /* Ground truth, when someone has labelled occupancy on the clean frames. */
  function renderAgreement() {
    const host = $('#zgt');
    const b = block();
    const link = `zones-label.html?run=${encodeURIComponent(runId)}`;
    if (!labels || !Object.keys(labels).length) {
      host.innerHTML = `<p class="sub">No ground truth yet. <a href="${link}">Label zone occupancy</a> on the saved frames to measure accuracy.</p>`;
      return;
    }
    const out = Object.entries(labels).map(([who, d]) => {
      let tp = 0, fp = 0, fn = 0, tn = 0;
      Object.entries(d.frames || {}).forEach(([t, row]) => {
        const st = statusAt(+t, b);
        st.forEach((s) => {
          const v = row[s.z.id];
          if (v === null || v === undefined || !SEATED(s.z)) return;
          if (v && s.taken) tp++; else if (!v && s.taken) fp++; else if (v && !s.taken) fn++; else tn++;
        });
      });
      const n = tp + fp + fn + tn;
      const acc = n ? (tp + tn) / n : 0;
      const p = tp + fp ? tp / (tp + fp) : 0, r = tp + fn ? tp / (tp + fn) : 0;
      const f1 = p + r ? (2 * p * r) / (p + r) : 0;
      const pe = n ? (((tp + fp) * (tp + fn)) + ((fn + tn) * (fp + tn))) / (n * n) : 0;
      const kappa = pe < 1 ? (acc - pe) / (1 - pe) : 0;
      return `<tr><td>${esc(who)}</td><td class="num">${n}</td><td class="num">${fmt.pct(acc * 100, 1)}</td><td class="num">${fmt.fixed(f1)}</td><td class="num">${fmt.fixed(kappa)}</td></tr>`;
    });
    host.innerHTML = `<table><thead><tr><th>Against</th><th class="num">Zone-moments</th><th class="num">Accuracy</th><th class="num">F1 (taken)</th><th class="num">κ</th></tr></thead><tbody>${out.join('')}</tbody></table>
      <p class="sub">Each labelled zone at each labelled moment, against the smoothed state. <a href="${link}">Label more</a>.</p>`;
  }

  async function loadLabels() {
    labels = null;
    if (!runId || !getState().runs.length) return;
    try {
      const res = await fetch(`/api/runs/${encodeURIComponent(runId)}/zones`);
      if (!res.ok) return;
      const info = await res.json();
      const got = {};
      for (const a of info.annotators || []) {
        const r = await fetch(`/api/runs/${encodeURIComponent(runId)}/zone-labels/${encodeURIComponent(a)}`);
        if (r.ok) got[a] = await r.json();
      }
      labels = got;
      if (Z) renderAgreement();
    } catch { /* no server */ }
  }

  /* ------------------------------------------------------------ settings */
  let previewTimer = 0;
  async function sendParams(save) {
    const f = $('#zparams');
    const params = {
      ...block().params,
      anchor: f.anchor.value,
      on_sec: +f.on_sec.value, off_sec: +f.off_sec.value,
      gap_fill_sec: +f.gap_fill_sec.value, long_stay_sec: +f.long_stay_sec.value,
      stick: +f.stick.value,
    };
    $('#zparams-state').textContent = save ? 'Saving' : 'Recomputing';
    try {
      const res = await fetch(`/api/runs/${encodeURIComponent(runId)}/zones`, {
        method: 'PUT', headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ params, save }),
      });
      if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
      const out = await res.json();
      if (save) { Z = out; preview = null; delete f.dataset.dirty; onAnalysis(out); } else preview = out;
      renderAll();
    } catch (err) {
      $('#zparams-state').innerHTML = `<span class="warn">Could not recompute: ${esc(err.message)}</span>`;
    }
  }
  function wireParams() {
    const f = $('#zparams');
    f.addEventListener('input', () => {
      f.dataset.dirty = '1';
      showOutputs(f);
      clearTimeout(previewTimer);
      previewTimer = setTimeout(() => sendParams(false), 350);
    });
    f.addEventListener('submit', (e) => e.preventDefault());
    $('#zparams-save').onclick = () => sendParams(true);
    $('#zparams-reset').onclick = () => { preview = null; delete f.dataset.dirty; renderAll(); };
  }

  /* -------------------------------------------------------------- editor */
  const nextId = (kind, zones) => {
    const pre = KIND_PREFIX[kind] || 'Z';
    for (let i = 1; ; i++) {
      const id = `${pre}${String(i).padStart(2, '0')}`;
      if (!zones.some((z) => z.id === id)) return id;
    }
  };
  const snapshot = () => JSON.parse(JSON.stringify(edit.zones));
  const remember = () => { edit.history.push(snapshot()); if (edit.history.length > 60) edit.history.shift(); };

  function startEdit(mode = 'select') {
    if (!runId) return;
    video.pause();
    edit = {
      zones: JSON.parse(JSON.stringify((Z?.zones || []).map(({ id, name, kind, polygon, capacity }) => ({ id, name, kind, polygon, capacity })))),
      sel: null, mode, draft: [], drag: null, history: [], kind: Z?.zones?.[0]?.kind || 'table',
    };
    cam.classList.add('editing');
    $('#zedit-bar').hidden = false;
    $('#zcam-bar').hidden = true;
    $$('.zones-body').forEach((n) => { n.hidden = !Z; });
    $('#zones-empty').hidden = true;
    setSource();
    syncEditBar();
    layout();
  }
  function stopEdit() {
    edit = null;
    cam.classList.remove('editing');
    $('#zedit-bar').hidden = true;
    $('#zcam-bar').hidden = false;
    canvas.style.cursor = '';
    setSource();
    renderAll();
  }
  function syncEditBar() {
    if (!edit) return;
    $$('#zedit-mode button').forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.mode === edit.mode)));
    const z = edit.sel != null ? edit.zones[edit.sel] : null;
    $('#zedit-sel').hidden = !z;
    $('#zedit-kind').value = z ? z.kind : edit.kind;
    if (z) { $('#zedit-name').value = z.name; $('#zedit-id').value = z.id; }
    $('#zedit-count').textContent = `${edit.zones.length} ${edit.zones.length === 1 ? 'zone' : 'zones'}`;
    $('#zedit-hint').textContent = {
      select: z ? 'Drag a corner to reshape, drag inside to move. Delete removes it.' : 'Click a zone to change it.',
      poly: edit.draft.length ? 'Click to add corners. Click the first corner, double-click or press Enter to finish. Esc cancels.' : 'Click on the frame to place the first corner.',
      rect: 'Drag a box over a table or seat.',
    }[edit.mode];
    $('#zedit-undo').disabled = !edit.history.length;
  }

  function hitVertex(p, z) {
    const W = canvas.clientWidth, H = canvas.clientHeight;
    return z.polygon.findIndex(([x, y]) => Math.hypot((x - p[0]) * W, (y - p[1]) * H) < 9);
  }
  function finishDraft() {
    if (edit.draft.length >= 3) {
      remember();
      const kind = $('#zedit-kind').value;
      const id = nextId(kind, edit.zones);
      edit.zones.push({ id, name: `${KIND_NAME[kind]} ${id.slice(1).replace(/^0/, '')}`, kind, polygon: edit.draft.map((p) => p.map((v) => +v.toFixed(5))), capacity: null });
      edit.sel = edit.zones.length - 1;
      edit.mode = 'select';
    }
    edit.draft = [];
    syncEditBar(); draw();
  }

  canvas.addEventListener('pointerdown', (e) => {
    if (!edit) return;
    const p = toNorm(e);
    canvas.setPointerCapture(e.pointerId);
    if (edit.mode === 'poly') {
      if (edit.draft.length >= 3) {
        const [fx, fy] = edit.draft[0];
        if (Math.hypot((fx - p[0]) * canvas.clientWidth, (fy - p[1]) * canvas.clientHeight) < 10) return finishDraft();
      }
      edit.draft.push(p);
      syncEditBar(); draw();
      return;
    }
    if (edit.mode === 'rect') {
      edit.drag = { kind: 'rect', from: p, to: p };
      return;
    }
    const sel = edit.sel != null ? edit.zones[edit.sel] : null;
    const vi = sel ? hitVertex(p, sel) : -1;
    if (vi >= 0) {
      remember();
      edit.drag = { kind: 'vertex', vi };
      return;
    }
    const hit = edit.zones.map((z, i) => [z, i]).reverse().find(([z]) => inside(p, z.polygon));
    edit.sel = hit ? hit[1] : null;
    if (hit) {
      remember();
      edit.drag = { kind: 'move', from: p, orig: JSON.parse(JSON.stringify(hit[0].polygon)) };
    }
    syncEditBar(); draw();
  });
  function onEditMove(e) {
    const p = toNorm(e);
    edit.cursor = p;
    const d = edit.drag;
    if (d?.kind === 'vertex') {
      edit.zones[edit.sel].polygon[d.vi] = p;
    } else if (d?.kind === 'move') {
      const dx = p[0] - d.from[0], dy = p[1] - d.from[1];
      edit.zones[edit.sel].polygon = d.orig.map(([x, y]) => [Math.min(1, Math.max(0, x + dx)), Math.min(1, Math.max(0, y + dy))]);
    } else if (d?.kind === 'rect') {
      d.to = p;
    }
    const sel = edit.sel != null ? edit.zones[edit.sel] : null;
    canvas.style.cursor = edit.mode !== 'select' ? 'crosshair'
      : sel && hitVertex(p, sel) >= 0 ? 'grab'
        : edit.zones.some((z) => inside(p, z.polygon)) ? 'move' : 'default';
    draw();
  }
  canvas.addEventListener('pointerup', () => {
    if (!edit?.drag) return;
    const d = edit.drag;
    edit.drag = null;
    if (d.kind === 'rect') {
      const [x1, y1] = d.from, [x2, y2] = d.to;
      if (Math.abs(x2 - x1) * canvas.clientWidth > 8 && Math.abs(y2 - y1) * canvas.clientHeight > 8) {
        edit.draft = [[Math.min(x1, x2), Math.min(y1, y2)], [Math.max(x1, x2), Math.min(y1, y2)],
          [Math.max(x1, x2), Math.max(y1, y2)], [Math.min(x1, x2), Math.max(y1, y2)]];
        finishDraft();
        return;
      }
    }
    syncEditBar(); draw();
  });
  canvas.addEventListener('dblclick', () => { if (edit?.mode === 'poly') finishDraft(); });
  document.addEventListener('keydown', (e) => {
    if (!edit || !visible) return;
    if (e.target instanceof Element && e.target.closest('input, select, textarea')) return;
    if (e.key === 'Enter' && edit.mode === 'poly') { e.preventDefault(); finishDraft(); }
    else if (e.key === 'Escape') { e.preventDefault(); if (edit.draft.length) { edit.draft = []; } else edit.sel = null; syncEditBar(); draw(); }
    else if ((e.key === 'Delete' || e.key === 'Backspace') && edit.mode === 'poly' && edit.draft.length) { e.preventDefault(); edit.draft.pop(); syncEditBar(); draw(); }
    else if ((e.key === 'Delete' || e.key === 'Backspace') && edit.sel != null) { e.preventDefault(); remember(); edit.zones.splice(edit.sel, 1); edit.sel = null; syncEditBar(); draw(); }
    else if (e.key === 'z' && (e.ctrlKey || e.metaKey)) { e.preventDefault(); undo(); }
  });
  function undo() {
    if (!edit?.history.length) return;
    edit.zones = edit.history.pop();
    edit.sel = null;
    syncEditBar(); draw();
  }

  function drawEditor(col) {
    const W = canvas.clientWidth, H = canvas.clientHeight;
    ctx.fillStyle = 'rgba(0,0,0,.18)';
    ctx.fillRect(0, 0, W, H);
    edit.zones.forEach((z, i) => {
      const sel = i === edit.sel;
      tracePath(z.polygon);
      ctx.fillStyle = z.kind === 'queue' ? col.queue : col.free;
      ctx.globalAlpha = sel ? 0.32 : 0.18;
      ctx.fill();
      ctx.globalAlpha = 1;
      ctx.setLineDash(sel ? [] : [6, 4]);
      ctx.lineWidth = sel ? 2.5 : 1.8;
      ctx.strokeStyle = sel ? '#fff' : (z.kind === 'queue' ? col.queue : col.free);
      ctx.stroke();
      ctx.setLineDash([]);
      const [cx, cy] = toPx(centroid(z.polygon));
      chip(z.id, cx, cy, sel ? '#fff' : (z.kind === 'queue' ? col.queue : col.free), sel ? '#111' : '#fff');
      if (sel) {
        z.polygon.forEach((p) => {
          const [x, y] = toPx(p);
          ctx.beginPath(); ctx.arc(x, y, 5.5, 0, Math.PI * 2);
          ctx.fillStyle = '#fff'; ctx.fill();
          ctx.lineWidth = 2; ctx.strokeStyle = col.next; ctx.stroke();
        });
      }
    });
    const pts = [...edit.draft];
    if (edit.drag?.kind === 'rect') {
      const [x1, y1] = edit.drag.from, [x2, y2] = edit.drag.to;
      pts.push([x1, y1], [x2, y1], [x2, y2], [x1, y2]);
    }
    if (pts.length) {
      ctx.beginPath();
      pts.forEach((p, i) => { const [x, y] = toPx(p); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
      if (edit.mode === 'poly' && edit.cursor) { const [x, y] = toPx(edit.cursor); ctx.lineTo(x, y); }
      if (edit.drag?.kind === 'rect') ctx.closePath();
      ctx.lineWidth = 2; ctx.strokeStyle = col.next; ctx.setLineDash([5, 4]); ctx.stroke(); ctx.setLineDash([]);
      pts.forEach((p, i) => {
        const [x, y] = toPx(p);
        ctx.beginPath(); ctx.arc(x, y, i === 0 ? 6.5 : 4.5, 0, Math.PI * 2);
        ctx.fillStyle = i === 0 ? col.next : '#fff'; ctx.fill();
      });
    }
  }

  async function propose() {
    $('#zedit-hint').textContent = 'Looking for places where people sat still';
    try {
      const res = await fetch(`/api/runs/${encodeURIComponent(runId)}/zones/propose`, {
        method: 'POST', headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ min_stay_sec: +$('#zedit-stay').value || 20, params: Z?.params || {} }),
      });
      if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
      const prop = await res.json();
      if (!edit) startEdit();
      remember();
      edit.zones = prop.zones.map(({ id, name, kind, polygon, capacity }) => ({ id, name, kind, polygon, capacity }));
      edit.sel = null;
      syncEditBar(); draw();
      $('#zedit-hint').textContent = prop.zones.length
        ? `Found ${prop.zones.length} seats. Adjust or delete any, then save. Undo brings back the previous zones.`
        : 'Nobody sat still long enough to make a seat. Try a shorter stay, or draw the zones.';
    } catch (err) {
      $('#zedit-hint').textContent = `Could not find seats: ${err.message}`;
    }
  }

  async function save() {
    const btn = $('#zedit-save');
    btn.disabled = true;
    $('#zedit-hint').textContent = 'Saving and measuring';
    try {
      const res = await fetch(`/api/runs/${encodeURIComponent(runId)}/zones`, {
        method: 'PUT', headers: { 'content-type': 'application/json' },
        body: JSON.stringify({
          zones: { zones: edit.zones, source: 'user' },
          params: Z?.params || {}, save: true, camera: $('#zedit-camera').checked,
        }),
      });
      if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
      Z = await res.json();
      preview = null;
      onAnalysis(Z);
      stopEdit();
    } catch (err) {
      $('#zedit-hint').textContent = `Not saved: ${err.message}`;
    } finally {
      btn.disabled = false;
    }
  }

  function wireEditor() {
    $('#zones-edit').onclick = () => startEdit('select');
    $('#zones-empty-draw').onclick = () => startEdit('rect');
    $('#zones-empty-auto').onclick = () => { startEdit('select'); propose(); };
    $$('#zedit-mode button').forEach((b) => {
      b.onclick = () => { edit.mode = b.dataset.mode; edit.draft = []; syncEditBar(); draw(); };
    });
    $('#zedit-kind').onchange = (e) => {
      edit.kind = e.target.value;
      if (edit.sel != null) {
        remember();
        const z = edit.zones[edit.sel];
        z.kind = e.target.value;
        const id = nextId(z.kind, edit.zones.filter((x) => x !== z));
        if (z.id[0] !== KIND_PREFIX[z.kind]) { z.id = id; z.name = `${KIND_NAME[z.kind]} ${id.slice(1).replace(/^0/, '')}`; }
      }
      syncEditBar(); draw();
    };
    $('#zedit-name').oninput = (e) => { if (edit.sel != null) edit.zones[edit.sel].name = e.target.value.slice(0, 48); };
    $('#zedit-id').onchange = (e) => {
      const v = e.target.value.trim();
      if (edit.sel == null) return;
      if (!/^[A-Za-z0-9_.-]{1,24}$/.test(v) || edit.zones.some((z, i) => i !== edit.sel && z.id === v)) {
        e.target.value = edit.zones[edit.sel].id;
        $('#zedit-hint').textContent = 'Ids are up to 24 letters, digits, ., _ or -, and each must be unique.';
        return;
      }
      remember(); edit.zones[edit.sel].id = v; draw();
    };
    $('#zedit-del').onclick = () => { if (edit.sel != null) { remember(); edit.zones.splice(edit.sel, 1); edit.sel = null; syncEditBar(); draw(); } };
    $('#zedit-clear').onclick = () => { remember(); edit.zones = []; edit.sel = null; syncEditBar(); draw(); };
    $('#zedit-undo').onclick = undo;
    $('#zedit-auto').onclick = propose;
    $('#zedit-cancel').onclick = stopEdit;
    $('#zedit-save').onclick = save;
  }

  /* ------------------------------------------------------------ playback */
  function wirePlayback() {
    $('#zcam-play').onclick = () => {
      if (video.paused) {
        if (Math.abs(video.currentTime - now()) > 0.3) video.currentTime = now();
        video.play().catch(() => {});
      } else video.pause();
    };
    video.addEventListener('play', () => { cam.classList.add('playing'); startLoop(); });
    video.addEventListener('pause', () => { cam.classList.remove('playing'); setPlayhead(video.currentTime, { fromZones: true }); });
    video.addEventListener('seeked', () => draw());
    $('#zcam-scrub').addEventListener('input', (e) => {
      const t = (+e.target.value / 1000) * dur();
      if (!video.paused) video.currentTime = t;
      seek(t);
    });
    $('#zopt-timers').onchange = (e) => { opt.timers = e.target.checked; draw(); };
    $('#zopt-names').onchange = (e) => { opt.names = e.target.checked; draw(); };
  }

  /* --------------------------------------------------------------- public */
  function renderAll() {
    renderEmpty();
    if (!Z && !edit) return;
    if (Z) {
      renderBoard(); renderNext(); renderQueue(); renderTiles(); renderGantt();
      renderUtil(); renderSurvival(); renderEvents(); renderMethod(); renderValidity();
    }
    $('#zones-edit').hidden = !(run?.has_boxes && getState().runs.length);
    renderClock();
    layout();
  }

  let tickTimer = 0;
  wireEditor();
  wirePlayback();
  wireParams();
  return {
    setRun(id, d, r) {
      if (edit) stopEdit();
      runId = id; doc = d; run = r; Z = d.zones || null; preview = null; focus = null;
      const f = $('#zparams');
      delete f.dataset.dirty;
      video.pause();
      setSource();
      renderAll();
      loadLabels();
    },
    setAnalysis(b) { Z = b; preview = null; renderAll(); },
    rerender() { renderAll(); },
    /* The page's playhead moved. */
    tick(t, { fromZones = false } = {}) {
      if (!doc) return;
      if (!fromZones && video && !video.paused && Math.abs(video.currentTime - t) > 0.5) video.currentTime = t;
      if (!fromZones && video?.paused && Number.isFinite(video.duration) && Math.abs(video.currentTime - t) > 0.05) {
        video.currentTime = Math.min(t, video.duration);
      }
      renderClock(t);
      if (!raf) draw();
      if (!tickTimer) {
        tickTimer = setTimeout(() => {
          tickTimer = 0;
          if (!Z) return;
          renderBoard(); renderNext(); renderQueue(); renderTiles(); renderEvents();
          $('#c-zgantt').__render?.();
          $('#c-zsurv').__render?.();
        }, 120);
      }
    },
    show(on) {
      visible = on;
      if (on) { layout(); startLoop(); } else { video.pause(); if (edit) stopEdit(); }
    },
    statusAt: (t) => statusAt(t),
    nextFree: (t) => nextFree(statusAt(t)),
  };
}
