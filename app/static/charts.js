/* Hand-rolled SVG chart primitives (the 3D views live in stage3d.js and
   skyline3d.js).
   No chart library: every mark is drawn to the same spec (<=24px bars, 4px
   rounded data-ends, 2px surface gaps, 2px lines, hairline solid grid) and
   every colour comes from a CSS variable, so light/dark swap in one place.
   Each chart renders at its container's measured width and re-renders on
   resize; every chart has a hover tooltip and a table-view twin in the DOM. */

const NS = 'http://www.w3.org/2000/svg';
const SURFACE = () => getComputedStyle(document.body).getPropertyValue('--surface').trim();

/* Resolve a colour (possibly `var(--x)`) and pick legible ink for text on it. */
function resolve(color) {
  const m = /^var\((--[\w-]+)\)$/.exec((color || '').trim());
  return m ? getComputedStyle(document.body).getPropertyValue(m[1]).trim() : color;
}
function inkOn(color) {
  const hex = resolve(color);
  const m = /^#?([0-9a-f]{6})$/i.exec(hex);
  if (!m) return '#fff';
  const n = parseInt(m[1], 16);
  const [r, g, b] = [(n >> 16) & 255, (n >> 8) & 255, n & 255].map((v) => {
    const c = v / 255;
    return c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
  });
  const L = 0.2126 * r + 0.7152 * g + 0.0722 * b;
  // Pick whichever ink has the higher WCAG contrast against the fill.
  return (L + 0.05) / 0.05 > 1.05 / (L + 0.05) ? '#0b0b0b' : '#ffffff';
}

export const fmt = {
  int: (n) => Math.round(n).toLocaleString(),
  num: (n, d = 1) => (Math.round(n * 10 ** d) / 10 ** d).toLocaleString(),
  pct: (n, d = 1) => `${fmt.num(n, d)}%`,
  fixed: (n, d = 2) => (n ?? 0).toFixed(d),   // columns need a stable decimal count
  dur(s) {
    if (s == null) return '--';
    if (s < 60) return `${fmt.num(s, 1)}s`;
    const m = Math.floor(s / 60);
    return `${m}m ${Math.round(s - m * 60)}s`;
  },
  clock(s) {
    const m = Math.floor(s / 60), r = Math.floor(s % 60);
    return `${m}:${String(r).padStart(2, '0')}`;
  },
};

function el(name, attrs = {}, parent = null) {
  const node = document.createElementNS(NS, name);
  for (const [k, v] of Object.entries(attrs)) {
    if (v !== null && v !== undefined) node.setAttribute(k, v);
  }
  if (parent) parent.appendChild(node);
  return node;
}

/* A bar with rounded data-end and a square baseline end. */
function barPath(x, y, w, h, r = 4, dir = 'up') {
  r = Math.max(0, Math.min(r, w / 2, h));
  if (h <= 0.5) return `M${x} ${y + h} h${w}`;
  if (dir === 'up') {
    return `M${x} ${y + h} V${y + r} a${r} ${r} 0 0 1 ${r} ${-r} h${w - 2 * r} a${r} ${r} 0 0 1 ${r} ${r} V${y + h} Z`;
  }
  // horizontal, growing right
  return `M${x} ${y} h${w - r} a${r} ${r} 0 0 1 ${r} ${r} v${h - 2 * r} a${r} ${r} 0 0 1 ${-r} ${r} H${x} Z`;
}

function niceTicks(max, count = 4) {
  if (max <= 0) return [0, 1];
  const raw = max / count;
  const mag = 10 ** Math.floor(Math.log10(raw));
  let step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw) || mag * 10;
  // Counts are integers: fractional ticks would render as duplicate labels.
  if (step < 1 && Number.isInteger(max)) step = 1;
  // Round the top tick UP past the max - otherwise the tallest mark is scaled
  // against a ceiling below its own value and overflows the plot.
  const top = Math.ceil(max / step - 1e-9) * step;
  const ticks = [];
  for (let v = 0; v <= top + step * 0.001; v += step) ticks.push(+v.toFixed(6));
  return ticks;
}

/* ------------------------------------------------------------- tooltip */
const tip = document.getElementById('tip');
export function showTip(evt, title, rows) {
  tip.innerHTML =
    `<div class="t-title">${title}</div>` +
    rows.map((r) => `<div class="t-row"><span>${r.color ? `<i class="t-key" style="display:inline-block;background:${r.color}"></i>` : ''}${r.label}</span><b>${r.value}</b></div>`).join('');
  tip.style.opacity = 1;
  moveTip(evt);
}
export function moveTip(evt) {
  const pad = 12;
  const r = tip.getBoundingClientRect();
  let x = evt.clientX, y = evt.clientY - pad;
  x = Math.min(Math.max(x, r.width / 2 + 4), window.innerWidth - r.width / 2 - 4);
  if (y - r.height < 4) y = evt.clientY + r.height + pad;
  tip.style.left = `${x}px`;
  tip.style.top = `${y}px`;
}
export function hideTip() { tip.style.opacity = 0; }

function hoverable(node, title, rows, onClick) {
  node.addEventListener('pointerenter', (e) => showTip(e, title, rows));
  node.addEventListener('pointermove', moveTip);
  node.addEventListener('pointerleave', hideTip);
  if (onClick) node.addEventListener('click', onClick);
  node.setAttribute('tabindex', '0');
  node.addEventListener('focus', (e) => {
    const b = node.getBoundingClientRect();
    showTip({ clientX: b.x + b.width / 2, clientY: b.y }, title, rows);
  });
  node.addEventListener('blur', hideTip);
}

/* Mount a render function that re-runs on container resize. */
const observed = new WeakMap();
export function mount(container, render) {
  const run = () => {
    const w = container.clientWidth;
    if (!w) return;
    container.innerHTML = '';
    render(container, w);
  };
  if (!observed.has(container)) {
    let t;
    // Call through the element: this closure belongs to the first mount, and
    // mounting again with new data must not be undone by the next resize.
    const ro = new ResizeObserver(() => { clearTimeout(t); t = setTimeout(() => container.__render(), 60); });
    ro.observe(container);
    observed.set(container, ro);
  }
  container.__render = run;
  run();
}

function svgRoot(container, w, h) {
  const svg = el('svg', { class: 'chart', width: w, height: h, viewBox: `0 0 ${w} ${h}` }, container);
  return svg;
}

/* ------------------------------------------------------------- columns */
export function columns(container, opts) {
  mount(container, (root, w) => {
    const { data, height = 200, format = fmt.int, onClick, labelEvery = 1, unit = '' } = opts;
    const pad = { t: 24, r: 8, b: 26, l: 38 };
    const h = height;
    const svg = svgRoot(root, w, h);
    const iw = w - pad.l - pad.r, ih = h - pad.t - pad.b;
    const max = Math.max(...data.map((d) => d.value), 0);
    const ticks = niceTicks(max);
    const top = ticks[ticks.length - 1] || 1;
    const y = (v) => pad.t + ih - (v / top) * ih;

    for (const t of ticks) {
      el('line', { class: 'grid-line', x1: pad.l, x2: w - pad.r, y1: y(t), y2: y(t) }, svg);
      const tx = el('text', { x: pad.l - 8, y: y(t) + 4, 'text-anchor': 'end' }, svg);
      tx.textContent = format(t);
    }
    const band = iw / data.length;
    const bw = Math.min(24, band - 8);
    data.forEach((d, i) => {
      const x = pad.l + band * i + (band - bw) / 2;
      const bh = Math.max(0, ih - (y(d.value) - pad.t));
      el('path', { class: 'mark', d: barPath(x, y(d.value), bw, bh), fill: d.color || 'var(--neutral)' }, svg);
      const hit = el('rect', { class: 'hit', x: pad.l + band * i, y: pad.t, width: band, height: ih }, svg);
      if (onClick) hit.style.cursor = 'pointer';
      hoverable(hit, d.label, [{ label: d.hint || 'Count', value: `${format(d.value)}${unit}`, color: d.color }],
        onClick && (() => onClick(d)));
      if (i % labelEvery === 0) {
        const tx = el('text', { class: 'lbl', x: pad.l + band * i + band / 2, y: h - 8, 'text-anchor': 'middle' }, svg);
        tx.textContent = d.label;
      }
      if (d.value === max && max > 0) {
        const vl = el('text', { class: 'val', x: x + bw / 2, y: y(d.value) - 6, 'text-anchor': 'middle' }, svg);
        vl.textContent = format(d.value);
      }
    });
  });
}

/* ------------------------------------------------- grouped / stacked columns */
export function groupedColumns(container, opts) {
  mount(container, (root, w) => {
    const { categories, series, height = 230, stacked = false, format = fmt.int, onClick } = opts;
    const pad = { t: 24, r: 8, b: 30, l: 40 };
    const svg = svgRoot(root, w, height);
    const iw = w - pad.l - pad.r, ih = height - pad.t - pad.b;
    const max = stacked
      ? Math.max(...categories.map((_, i) => series.reduce((s, ser) => s + ser.values[i], 0)))
      : Math.max(...series.flatMap((s) => s.values));
    const ticks = niceTicks(max);
    const top = ticks[ticks.length - 1] || 1;
    const y = (v) => pad.t + ih - (v / top) * ih;

    for (const t of ticks) {
      el('line', { class: 'grid-line', x1: pad.l, x2: w - pad.r, y1: y(t), y2: y(t) }, svg);
      const tx = el('text', { x: pad.l - 8, y: y(t) + 4, 'text-anchor': 'end' }, svg);
      tx.textContent = format(t);
    }
    const band = iw / categories.length;
    const labelStep = Math.max(1, Math.ceil(categories.length / Math.max(2, Math.floor(iw / 46))));
    const groupW = Math.min(band - 12, stacked ? 24 : series.length * 26);
    const bw = stacked ? groupW : Math.min(22, groupW / series.length - 2);

    categories.forEach((cat, i) => {
      const gx = pad.l + band * i + (band - groupW) / 2;
      // Only the topmost segment that actually has a value gets the rounded end.
      let topSeg = -1;
      series.forEach((ser, j) => { if (ser.values[i] > 0) topSeg = j; });
      let acc = 0;
      series.forEach((ser, j) => {
        const v = ser.values[i];
        let x, yTop, bh;
        if (stacked) {
          x = gx;
          const y0 = y(acc), y1 = y(acc + v);
          bh = Math.max(0, y0 - y1 - (j ? 2 : 0)); // 2px surface gap between segments
          yTop = y1;
          acc += v;
        } else {
          x = gx + j * (bw + 2);
          yTop = y(v);
          bh = ih - (yTop - pad.t);
        }
        if (v > 0) {
          el('path', {
            class: 'mark', fill: ser.color,
            d: barPath(x, yTop, bw, bh, stacked && j !== topSeg ? 0 : 4),
          }, svg);
        }
      });
      const hit = el('rect', { class: 'hit', x: pad.l + band * i, y: pad.t, width: band, height: ih }, svg);
      hoverable(hit, cat, series.map((s) => ({ label: s.name, value: format(s.values[i]), color: s.color })),
        onClick && (() => onClick(cat)));
      if (i % labelStep === 0) {
        const tx = el('text', { class: 'lbl', x: pad.l + band * i + band / 2, y: height - 10, 'text-anchor': 'middle' }, svg);
        tx.textContent = cat;
      }
    });
  });
}

/* ---------------------------------------------------------- lines / area */
export function lines(container, opts) {
  mount(container, (root, w) => {
    const { x, series, height = 220, format = fmt.int, xFormat = fmt.clock, area = false,
      onSeek = null, marker = null } = opts;
    const pad = { t: 16, r: 12, b: 26, l: 40 };
    const svg = svgRoot(root, w, height);
    const iw = w - pad.l - pad.r, ih = height - pad.t - pad.b;
    const max = Math.max(...series.flatMap((s) => s.values), 0);
    const ticks = niceTicks(max);
    const top = ticks[ticks.length - 1] || 1;
    const X = (i) => pad.l + (x.length < 2 ? iw / 2 : (i / (x.length - 1)) * iw);
    const Y = (v) => pad.t + ih - (v / top) * ih;

    for (const t of ticks) {
      el('line', { class: 'grid-line', x1: pad.l, x2: w - pad.r, y1: Y(t), y2: Y(t) }, svg);
      const tx = el('text', { x: pad.l - 8, y: Y(t) + 4, 'text-anchor': 'end' }, svg);
      tx.textContent = format(t);
    }
    const step = Math.max(1, Math.ceil(x.length / Math.max(2, Math.floor(iw / 70))));
    x.forEach((xv, i) => {
      if (i % step) return;
      const tx = el('text', { class: 'lbl', x: X(i), y: height - 8, 'text-anchor': 'middle' }, svg);
      tx.textContent = xFormat(xv);
    });

    series.forEach((s) => {
      const pts = s.values.map((v, i) => `${X(i)},${Y(v)}`).join(' L');
      if (area) {
        el('path', {
          class: 'mark', fill: s.color, 'fill-opacity': 0.1,
          d: `M${X(0)},${Y(0)} L${pts} L${X(s.values.length - 1)},${Y(0)} Z`,
        }, svg);
      }
      el('path', {
        class: 'mark', d: `M${pts}`, fill: 'none', stroke: s.color,
        'stroke-width': 2, 'stroke-linejoin': 'round', 'stroke-linecap': 'round',
      }, svg);
      const li = s.values.length - 1;
      el('circle', { class: 'mark', cx: X(li), cy: Y(s.values[li]), r: 4, fill: s.color, stroke: SURFACE(), 'stroke-width': 2 }, svg);
    });

    // A function is read on every draw, so a moving playhead needs no re-mount.
    const at = typeof marker === 'function' ? marker() : marker;
    if (at != null && x.length > 1) {
      const frac = Math.max(0, Math.min(1, (at - x[0]) / (x[x.length - 1] - x[0] || 1)));
      const mx = pad.l + frac * iw;
      el('line', { x1: mx, x2: mx, y1: pad.t - 4, y2: pad.t + ih, stroke: 'var(--text-1)', 'stroke-width': 1.5, opacity: .55 }, svg);
    }

    // Parked with opacity, not off-canvas: charts render with overflow visible,
    // so an off-canvas mark would draw over the neighbouring card.
    const cross = el('line', { class: 'crosshair', y1: pad.t, y2: pad.t + ih, x1: pad.l, x2: pad.l, opacity: 0 }, svg);
    const dots = series.map((s) => el('circle', { r: 4.5, fill: s.color, stroke: SURFACE(), 'stroke-width': 2, opacity: 0 }, svg));
    const hit = el('rect', { class: 'hit', x: pad.l, y: pad.t, width: iw, height: ih }, svg);
    if (onSeek) hit.style.cursor = 'pointer';
    hit.addEventListener('pointermove', (e) => {
      const bb = svg.getBoundingClientRect();
      const rel = (e.clientX - bb.left - pad.l) / iw;
      const i = Math.max(0, Math.min(x.length - 1, Math.round(rel * (x.length - 1))));
      cross.setAttribute('x1', X(i)); cross.setAttribute('x2', X(i)); cross.setAttribute('opacity', 1);
      dots.forEach((d, k) => {
        d.setAttribute('cx', X(i)); d.setAttribute('cy', Y(series[k].values[i])); d.setAttribute('opacity', 1);
      });
      showTip(e, xFormat(x[i]), series.map((s) => ({ label: s.name, value: format(s.values[i]), color: s.color })));
    });
    hit.addEventListener('pointerleave', () => {
      cross.setAttribute('opacity', 0);
      dots.forEach((d) => d.setAttribute('opacity', 0));
      hideTip();
    });
    if (onSeek) {
      hit.addEventListener('click', (e) => {
        const bb = svg.getBoundingClientRect();
        const rel = (e.clientX - bb.left - pad.l) / iw;
        const i = Math.max(0, Math.min(x.length - 1, Math.round(rel * (x.length - 1))));
        onSeek(x[i]);
      });
    }
  });
}

/* ----------------------------------------------------- horizontal bars */
export function hbars(container, opts) {
  mount(container, (root, w) => {
    const { data, rowH = 30, format = fmt.int, labelW = 110, color = 'var(--neutral)', unit = '' } = opts;
    const height = data.length * rowH + 8;
    const svg = svgRoot(root, w, height);
    const iw = w - labelW - 54;
    const max = Math.max(...data.map((d) => d.value), 0) || 1;
    data.forEach((d, i) => {
      const y = i * rowH + 4;
      const t = el('text', { class: 'lbl', x: 0, y: y + rowH / 2 + 1, 'dominant-baseline': 'middle' }, svg);
      t.textContent = d.label;
      const bh = 12;
      el('rect', { x: labelW, y: y + (rowH - bh) / 2, width: iw, height: bh, rx: 6, fill: 'var(--surface-3)' }, svg);
      const bw = Math.max(2, (d.value / max) * iw);
      el('path', { class: 'mark', d: barPath(labelW, y + (rowH - bh) / 2, bw, bh, 6, 'right'), fill: d.color || color }, svg);
      const v = el('text', { class: 'val', x: w, y: y + rowH / 2 + 1, 'text-anchor': 'end', 'dominant-baseline': 'middle' }, svg);
      v.textContent = `${format(d.value)}${unit}`;
      const hit = el('rect', { class: 'hit', x: 0, y, width: w, height: rowH }, svg);
      hoverable(hit, d.label, d.rows || [{ label: d.hint || 'Value', value: `${format(d.value)}${unit}`, color: d.color || color }]);
    });
  });
}

/* --------------------------------------------- stacked horizontal rows */
export function stackedRows(container, opts) {
  mount(container, (root, w) => {
    const { categories, series, rowH = 44, labelW = 70, format = fmt.pct } = opts;
    const height = categories.length * rowH + 4;
    const svg = svgRoot(root, w, height);
    const iw = w - labelW - 8;
    categories.forEach((cat, i) => {
      const y = i * rowH + 8;
      const bh = 18;
      const t = el('text', { class: 'lbl', x: 0, y: y + bh / 2 + 1, 'dominant-baseline': 'middle' }, svg);
      t.textContent = cat;
      const total = series.reduce((sum, s) => sum + s.values[i], 0) || 1;
      el('rect', { x: labelW, y, width: iw, height: bh, rx: 5, fill: 'var(--surface-3)' }, svg);
      let acc = 0;
      series.forEach((s) => {
        const frac = s.values[i] / total;
        if (frac <= 0) return;
        const x = labelW + (acc / total) * iw;
        const segW = Math.max(1, frac * iw - 2);   // 2px surface gap between segments
        const seg = el('rect', { class: 'mark', x, y, width: segW, height: bh, rx: 3, fill: s.color }, svg);
        hoverable(seg, `${cat} · ${s.name}`, [{ label: 'Share', value: format(frac * 100), color: s.color },
          { label: 'Value', value: fmt.num(s.values[i], 1) }]);
        if (segW > 34) {
          const lb = el('text', { x: x + segW / 2, y: y + bh / 2 + 1, 'text-anchor': 'middle',
            'dominant-baseline': 'middle', 'font-size': 10.5, 'font-weight': 600 }, svg);
          // Inline style, not a fill attribute: `.chart text` in the stylesheet
          // would otherwise win and paint these muted.
          lb.style.fill = inkOn(s.color);
          lb.textContent = format(frac * 100, 0);
        }
        acc += s.values[i];
      });
    });
  });
}

/* -------------------------------------------------------------- heatmap */
export function heatgrid(container, opts) {
  mount(container, (root, w) => {
    // paths: [{ points: [[x, y], ...] (0-1), color }]; heat: false draws only the paths.
    const { grid, cols, rows, aspect = 16 / 9, onCell = null, paths = null, heat = true } = opts;
    const height = Math.round(w / aspect);
    const svg = svgRoot(root, w, height);
    const cw = w / cols, ch = height / rows;
    const ramp = ['--seq-0', '--seq-1', '--seq-2', '--seq-3', '--seq-4', '--seq-5', '--seq-6'];
    el('rect', { x: 0, y: 0, width: w, height, fill: 'var(--seq-0)', rx: 10 }, svg);
    for (let r = 0; heat && r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const v = grid[r][c];
        if (v <= 0.001) continue;
        const step = Math.min(ramp.length - 1, Math.max(1, Math.ceil(v * (ramp.length - 1))));
        const cell = el('rect', {
          class: 'mark', x: c * cw + 1, y: r * ch + 1, width: cw - 2, height: ch - 2, rx: 3,
          fill: `var(${ramp[step]})`,
        }, svg);
        hoverable(cell, 'This spot', [{ label: 'Against the busiest spot', value: fmt.pct(v * 100, 0) }],
          onCell && (() => onCell(c, r)));
      }
    }
    if (paths) {
      for (const { points, color } of paths) {
        if (points.length < 2) continue;
        el('path', {
          d: `M${points.map(([x, y]) => `${x * w},${y * height}`).join(' L')}`,
          fill: 'none', stroke: color || 'var(--text-1)', 'stroke-width': color ? 1.5 : 1.25,
          'stroke-opacity': color ? 0.7 : 0.28, 'stroke-linecap': 'round', 'stroke-linejoin': 'round',
        }, svg);
      }
    }
    for (let c = 1; c < 3; c++) {
      el('line', { x1: (w / 3) * c, x2: (w / 3) * c, y1: 0, y2: height, stroke: 'var(--border)', 'stroke-dasharray': '0' }, svg);
    }
    ['Left', 'Center', 'Right'].forEach((z, i) => {
      const t = el('text', { class: 'lbl', x: (w / 3) * i + w / 6, y: height - 8, 'text-anchor': 'middle' }, svg);
      t.textContent = z;
    });
  });
}

/* --------------------------------------------------------- occupancy rows */
/* One row per zone, one bar per occupied stretch: a Gantt chart of who held
   which table when. A session that ran past either end of the recording is
   drawn open on that side (squared, faded), because its true length is not
   known. The playhead is read on every draw, like `lines`. */
export function timelineRows(container, opts) {
  mount(container, (root, w) => {
    const { rows, duration, rowH = 16, labelW = 54, color = 'var(--zone-taken)',
      marker = null, onSeek = null, xFormat = fmt.clock, tipRows = null, highlight = null } = opts;
    const pad = { t: 6, r: 10, b: 24 };
    const height = pad.t + rows.length * rowH + pad.b;
    const svg = svgRoot(root, w, height);
    const iw = Math.max(10, w - labelW - pad.r);
    const X = (t) => labelW + (Math.max(0, Math.min(duration, t)) / (duration || 1)) * iw;

    const ticks = niceTicks(duration, Math.max(2, Math.floor(iw / 90)));
    for (const t of ticks) {
      if (t > duration + 1e-6) continue;
      el('line', { class: 'grid-line', x1: X(t), x2: X(t), y1: pad.t, y2: pad.t + rows.length * rowH }, svg);
      const tx = el('text', { class: 'lbl', x: X(t), y: height - 7, 'text-anchor': 'middle' }, svg);
      tx.textContent = xFormat(t);
    }
    rows.forEach((row, i) => {
      const y = pad.t + i * rowH;
      const on = highlight && highlight === row.id;
      const tx = el('text', { class: on ? 'val' : 'lbl', x: labelW - 8, y: y + rowH / 2 + 4, 'text-anchor': 'end' }, svg);
      tx.textContent = row.label;
      el('rect', { x: labelW, y: y + rowH / 2 - 0.5, width: iw, height: 1, fill: 'var(--grid)' }, svg);
      row.spans.forEach((s, k) => {
        const x1 = X(s.start), x2 = Math.max(X(s.end), x1 + 1.5);
        const h = Math.max(4, rowH - 6);
        const bar = el('rect', {
          class: 'mark', x: x1, y: y + (rowH - h) / 2, width: x2 - x1, height: h,
          rx: Math.min(3, h / 2), fill: s.color || color,
          'fill-opacity': s.censored_start || s.censored_end ? 0.62 : 0.92,
        }, svg);
        const title = `${row.label}: ${xFormat(s.start)} to ${xFormat(s.end)}`;
        hoverable(bar, title, tipRows ? tipRows(row, s, k) : [{ label: 'Taken for', value: fmt.dur(s.end - s.start) }],
          onSeek && (() => onSeek(s.start, row, s)));
      });
    });
    const at = typeof marker === 'function' ? marker() : marker;
    if (at != null) {
      el('line', { x1: X(at), x2: X(at), y1: pad.t - 4, y2: pad.t + rows.length * rowH + 2,
        stroke: 'var(--text-1)', 'stroke-width': 1.5, opacity: 0.6 }, svg);
    }
    if (onSeek) {
      const hit = el('rect', { class: 'hit', x: labelW, y: pad.t + rows.length * rowH, width: iw, height: pad.b }, svg);
      hit.style.cursor = 'pointer';
      hit.addEventListener('click', (e) => {
        const bb = svg.getBoundingClientRect();
        onSeek(((e.clientX - bb.left - labelW) / iw) * duration);
      });
    }
  });
}

/* ------------------------------------------------------------- step curve */
/* A right-continuous step function, e.g. a Kaplan-Meier survival curve:
   the share of sessions still running after t seconds. */
export function stepCurve(container, opts) {
  mount(container, (root, w) => {
    const { t, s, height = 200, color = 'var(--neutral)', xFormat = fmt.dur, marker = null,
      yLabel = (v) => fmt.pct(v * 100, 0), name = 'Still taken' } = opts;
    const pad = { t: 14, r: 14, b: 26, l: 42 };
    const svg = svgRoot(root, w, height);
    const iw = w - pad.l - pad.r, ih = height - pad.t - pad.b;
    const tmax = Math.max(1, t[t.length - 1] * 1.04);
    const X = (v) => pad.l + (v / tmax) * iw;
    const Y = (v) => pad.t + ih - v * ih;
    for (const v of [0, 0.25, 0.5, 0.75, 1]) {
      el('line', { class: 'grid-line', x1: pad.l, x2: w - pad.r, y1: Y(v), y2: Y(v) }, svg);
      const tx = el('text', { x: pad.l - 8, y: Y(v) + 4, 'text-anchor': 'end' }, svg);
      tx.textContent = yLabel(v);
    }
    for (const v of niceTicks(tmax, Math.max(2, Math.floor(iw / 80)))) {
      if (v > tmax) continue;
      const tx = el('text', { class: 'lbl', x: X(v), y: height - 8, 'text-anchor': 'middle' }, svg);
      tx.textContent = xFormat(v);
    }
    let d = `M${X(0)},${Y(s[0])}`;
    for (let i = 1; i < t.length; i++) d += ` H${X(t[i])} V${Y(s[i])}`;
    d += ` H${X(tmax)}`;
    el('path', { class: 'mark', d: `${d} V${Y(0)} H${X(0)} Z`, fill: color, 'fill-opacity': 0.08 }, svg);
    el('path', { class: 'mark', d, fill: 'none', stroke: color, 'stroke-width': 2, 'stroke-linejoin': 'round' }, svg);
    const at = typeof marker === 'function' ? marker() : marker;
    if (at != null && at <= tmax) {
      el('line', { x1: X(at), x2: X(at), y1: pad.t, y2: pad.t + ih, stroke: 'var(--zone-next)', 'stroke-width': 1.5, 'stroke-dasharray': '3 3' }, svg);
    }
    const valAt = (x) => { let v = s[0]; for (let i = 0; i < t.length && t[i] <= x; i++) v = s[i]; return v; };
    const cross = el('line', { class: 'crosshair', y1: pad.t, y2: pad.t + ih, x1: pad.l, x2: pad.l, opacity: 0 }, svg);
    const hit = el('rect', { class: 'hit', x: pad.l, y: pad.t, width: iw, height: ih }, svg);
    hit.addEventListener('pointermove', (e) => {
      const bb = svg.getBoundingClientRect();
      const x = Math.max(0, ((e.clientX - bb.left - pad.l) / iw) * tmax);
      cross.setAttribute('x1', X(x)); cross.setAttribute('x2', X(x)); cross.setAttribute('opacity', 1);
      showTip(e, `After ${xFormat(x)}`, [{ label: name, value: yLabel(valAt(x)), color }]);
    });
    hit.addEventListener('pointerleave', () => { cross.setAttribute('opacity', 0); hideTip(); });
  });
}

/* ------------------------------------------------------------ split bar */
export function splitBar(container, parts, format = fmt.pct) {
  const total = parts.reduce((s, p) => s + p.value, 0) || 1;
  container.innerHTML = '';
  const row = document.createElement('div');
  row.style.cssText = 'display:flex;gap:2px;height:12px;border-radius:6px;overflow:hidden;margin:12px 0 4px';
  parts.forEach((p) => {
    const i = document.createElement('i');
    i.style.cssText = `flex:${p.value};background:${p.color};display:block`;
    i.title = `${p.label}: ${format((p.value / total) * 100)}`;
    row.appendChild(i);
  });
  container.appendChild(row);
  const legend = document.createElement('div');
  legend.className = 'legend';
  legend.innerHTML = parts.map((p) =>
    `<span class="item"><i class="key" style="background:${p.color}"></i>${p.label} <b style="color:var(--text-1);font-weight:600">${format((p.value / total) * 100)}</b></span>`).join('');
  container.appendChild(legend);
}

/* ------------------------------------------------------------- sparkline */
export function sparkline(container, values, color = 'var(--neutral)') {
  mount(container, (root, w) => {
    const h = 34;
    const svg = svgRoot(root, w, h);
    const max = Math.max(...values, 1);
    const pts = values.map((v, i) => `${(i / Math.max(values.length - 1, 1)) * w},${h - 2 - (v / max) * (h - 6)}`).join(' L');
    el('path', { d: `M${pts} L${w},${h} L0,${h} Z`, fill: color, 'fill-opacity': .1 }, svg);
    el('path', { d: `M${pts}`, fill: 'none', stroke: color, 'stroke-width': 2, 'stroke-linejoin': 'round' }, svg);
  });
}

export function rampLegend(container, lo = 'Low', hi = 'High') {
  const steps = [1, 2, 3, 4, 5, 6].map((i) =>
    `<i style="width:18px;height:10px;border-radius:2px;background:var(--seq-${i})"></i>`).join('');
  container.innerHTML =
    `<span class="item" style="gap:4px">${lo} ${steps} ${hi}</span>`;
}

export function legend(container, items) {
  container.innerHTML = items.map((i) =>
    `<span class="item"><i class="key${i.line ? ' line' : ''}" style="background:${i.color}"></i>${i.label}</span>`).join('');
}
