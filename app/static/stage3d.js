/* The floor: the camera frame laid flat and replayed in 3D.

   Three layers, all built from the filtered track list:
     heat    one column per heat-map cell, as tall as the time people spent there
     paths   every person's trail across the frame
     people  a marker per person, moving along their trail while the clock runs

   Time is the fourth dimension. At the end of the recording the columns are at
   full height; scrub back (or replay) and they grow as people stand around. */

import { T, makeStage, color, rampFrom, litBox, fitCamera, flyTo, hoverPicker, reducedMotion, easeOut } from './three-kit.js';

const FLOOR_W = 16;        // world units across; depth follows the frame's aspect ratio
const MAX_H = 3.3;         // height of the busiest cell
const RAMP = ['--seq-1', '--seq-2', '--seq-3', '--seq-4', '--seq-5', '--seq-6'];
const MAX_PEOPLE = 600;

export function createFloor(host, { showTip, hideTip, fmt, describe, insets, onTime, onPlayState }) {
  const stage = makeStage(host, { fov: 30 });
  const { scene, camera, controls } = stage;
  controls.minPolarAngle = 0.02;
  controls.maxPolarAngle = 1.4;          // never below the floor
  controls.minDistance = 7;
  controls.maxDistance = 220;            // a tall phone screen needs the camera a long way back

  scene.add(new T.HemisphereLight(0xffffff, 0x8a88b0, 2.2));
  const sun = new T.DirectionalLight(0xffffff, 1.3);
  sun.position.set(5, 12, 7);
  scene.add(sun);

  const root = new T.Group();
  scene.add(root);

  let S = null;                          // everything derived from the current data
  let t = 0, playing = false, intro = 1;
  let lift = 1;                          // 1 = full columns; near 0 in the top view, where they lie flat
  let mode = 'hero', layout = '3d', touched = false;
  const layers = { heat: true, paths: true, frame: false };
  const dummy = new T.Object3D();
  const tmp = new T.Color();

  /* Plain wheel scrolls the page; Ctrl/⌘ + wheel (which is also what a
     trackpad pinch sends) zooms. Otherwise the stage swallows page scrolling. */
  host.addEventListener('wheel', (e) => {
    if (!(e.ctrlKey || e.metaKey) || mode === 'hero') e.stopPropagation();
  }, { capture: true, passive: true });
  controls.addEventListener('start', () => { touched = true; });

  /* ------------------------------------------------------------- build */
  function clear() {
    stage.clearLabels();
    root.traverse((o) => {
      o.geometry?.dispose();
      if (o.material) {
        o.material.map?.dispose();
        o.material.dispose();
      }
    });
    root.clear();
  }

  /* --------------------------------------------------------------- footage */
  /* The camera frame follows the clock. A still would show one moment under
     trails made over the whole recording, so a trail could cross empty floor.
     One hidden video is kept and asked for the frame at the playhead; while a
     seek is in flight the newest request waits, so scrubbing never queues up. */
  const film = { video: null, url: null, want: 0 };
  function chase() {
    const v = film.video;
    if (!v || v.readyState < 1 || v.seeking) return;
    const want = Math.min(film.want, Math.max(0, (v.duration || film.want) - 0.05));
    if (Math.abs(v.currentTime - want) > 0.03) v.currentTime = want;
  }
  function filmFrame() {
    const v = film.video;
    if (!S || !v || S.filmUrl !== film.url || v.readyState < 2 || !v.videoWidth) return;
    if (!S.filmTex) {
      const c = document.createElement('canvas');
      const k = Math.min(1, 1280 / v.videoWidth);
      c.width = Math.round(v.videoWidth * k);
      c.height = Math.round(v.videoHeight * k);
      S.filmCtx = c.getContext('2d');
      S.filmTex = new T.CanvasTexture(c);
      S.filmTex.colorSpace = T.SRGBColorSpace;
      S.poster.material.map?.dispose();
      S.poster.material.map = S.filmTex;
      S.poster.material.needsUpdate = true;
      S.hasPoster = true;
      S.poster.material.visible = layers.frame;
    }
    S.filmCtx.drawImage(v, 0, 0, S.filmTex.image.width, S.filmTex.image.height);
    S.filmTex.needsUpdate = true;
    stage.invalidate();
    chase();
  }
  function syncFilm() {
    if (!S?.filmUrl || !layers.frame) return;
    if (film.url !== S.filmUrl) {
      film.video ||= Object.assign(document.createElement('video'), {
        muted: true, playsInline: true, preload: 'auto',
        onloadedmetadata: chase, onloadeddata: filmFrame, onseeked: filmFrame, oncanplay: filmFrame,
      });
      film.url = S.filmUrl;
      film.video.src = film.url;
    }
    film.want = t;
    if (!S.filmTex) filmFrame();     // same video as before, new data: show it at once
    chase();
  }

  function setData({ doc, rows, colorOf, posterUrl, videoUrl }) {
    clear();
    const W = doc.meta.width || 16, H = doc.meta.height || 9;
    const depth = FLOOR_W / (W / H);
    const cols = doc.heatmap.cols, nRows = doc.heatmap.rows;
    const duration = Math.max(doc.meta.duration_sec || 0, 0.001);
    const toX = (px) => (px / W - 0.5) * FLOOR_W;
    const toZ = (py) => (py / H - 0.5) * depth;
    const clampInt = (v, hi) => Math.min(hi, Math.max(0, Math.floor(v)));
    const cellOf = (px, py) => clampInt((py / H) * nRows, nRows - 1) * cols + clampInt((px / W) * cols, cols - 1);

    const tracks = rows.slice(0, MAX_PEOPLE).map((r) => ({
      r, pts: r.path.map(([pt, x, y]) => [pt, toX(x), toZ(y)]),
    }));

    // One segment per step of a path: how long it lasted and which cell it was in.
    const segs = [];
    rows.forEach((r, k) => {
      const p = r.path;
      if (p.length === 1) segs.push({ t0: p[0][0], dt: r.dwell_sec || 0, cell: cellOf(p[0][1], p[0][2]), k, i: -1 });
      for (let i = 0; i < p.length - 1; i++) {
        segs.push({ t0: p[i][0], dt: Math.max(0, p[i + 1][0] - p[i][0]), cell: cellOf(p[i][1], p[i][2]), k, i });
      }
    });
    segs.sort((a, b) => a.t0 - b.t0);
    const total = new Float32Array(cols * nRows);
    segs.forEach((s) => { total[s.cell] += s.dt; });
    const peak = Math.max(...total, 0) || 1;

    // ---- floor slab, cell grid, zone dividers
    const slab = new T.Mesh(
      new T.BoxGeometry(FLOOR_W + 0.7, 0.14, depth + 0.7),
      new T.MeshBasicMaterial({ transparent: true, depthWrite: false }),
    );
    slab.position.y = -0.075;
    slab.renderOrder = -2;
    root.add(slab);

    const poster = new T.Mesh(
      new T.PlaneGeometry(FLOOR_W, depth),
      new T.MeshBasicMaterial({ transparent: true, opacity: 0.6, depthWrite: false, visible: false }),
    );
    poster.rotation.x = -Math.PI / 2;
    poster.position.y = 0.004;
    poster.renderOrder = -1;
    root.add(poster);
    if (posterUrl) {
      new T.TextureLoader().load(posterUrl, (tex) => {
        // Too late if the data changed while loading, or the footage got there first.
        if (S?.poster !== poster || S.filmTex) { tex.dispose(); return; }
        tex.colorSpace = T.SRGBColorSpace;
        poster.material.map = tex;
        poster.material.needsUpdate = true;
        S.hasPoster = true;
        applyLayers();
      }, undefined, () => { /* no poster for this run */ });
    }

    const lines = (pts, opacity) => {
      const g = new T.BufferGeometry();
      g.setAttribute('position', new T.Float32BufferAttribute(pts, 3));
      const m = new T.LineSegments(g, new T.LineBasicMaterial({ transparent: true, opacity, depthWrite: false }));
      m.userData.baseOpacity = opacity;
      root.add(m);
      return m;
    };
    const hw = FLOOR_W / 2, hd = depth / 2, y = 0.012;
    const gridPts = [];
    for (let c = 1; c < cols; c++) { const x = -hw + (c * FLOOR_W) / cols; gridPts.push(x, y, -hd, x, y, hd); }
    for (let r = 1; r < nRows; r++) { const z = -hd + (r * depth) / nRows; gridPts.push(-hw, y, z, hw, y, z); }
    const grid = lines(gridPts, 0.07);
    const zonePts = [];
    [-1, 1].forEach((s) => { const x = (s * FLOOR_W) / 6; zonePts.push(x, y, -hd, x, y, hd); });
    zonePts.push(-hw, y, -hd, hw, y, -hd, hw, y, -hd, hw, y, hd, hw, y, hd, -hw, y, hd, -hw, y, hd, -hw, y, -hd);
    const zones = lines(zonePts, 0.28);

    ['Left', 'Center', 'Right'].forEach((name, i) => {
      stage.label(name, new T.Vector3(((i - 1) * FLOOR_W) / 3, 0, hd + 0.6), { cls: 'axis' });
    });

    // ---- heat columns
    const columns = new T.InstancedMesh(litBox(), new T.MeshBasicMaterial({ vertexColors: true }), cols * nRows);
    columns.instanceMatrix.setUsage(T.DynamicDrawUsage);
    columns.frustumCulled = false;
    root.add(columns);

    // ---- paths, in time order so a draw range reveals them as the clock runs
    const lineSegs = segs.filter((s) => s.i >= 0 && s.k < tracks.length);
    const pos = new Float32Array(lineSegs.length * 6);
    lineSegs.forEach((s, n) => {
      const a = tracks[s.k].pts[s.i], b = tracks[s.k].pts[s.i + 1];
      pos.set([a[1], 0.04, a[2], b[1], 0.04, b[2]], n * 6);
    });
    const pathGeo = new T.BufferGeometry();
    pathGeo.setAttribute('position', new T.BufferAttribute(pos, 3));
    pathGeo.setAttribute('color', new T.BufferAttribute(new Float32Array(lineSegs.length * 6), 3));
    const paths = new T.LineSegments(pathGeo, new T.LineBasicMaterial({
      vertexColors: true, transparent: true, opacity: 0.62, depthWrite: false,
    }));
    paths.frustumCulled = false;
    root.add(paths);

    // ---- people: a pill standing on a soft disc
    const n = Math.max(1, tracks.length);
    const bodies = new T.InstancedMesh(
      new T.CapsuleGeometry(0.15, 0.42, 4, 14),
      new T.MeshStandardMaterial({ roughness: 0.42, metalness: 0 }), n,
    );
    bodies.geometry.translate(0, 0.38, 0);
    const discs = new T.InstancedMesh(
      new T.CircleGeometry(0.34, 28),
      new T.MeshBasicMaterial({ transparent: true, opacity: 0.3, depthWrite: false }), n,
    );
    discs.geometry.rotateX(-Math.PI / 2);
    discs.geometry.translate(0, 0.02, 0);
    [bodies, discs].forEach((m) => {
      m.instanceMatrix.setUsage(T.DynamicDrawUsage);
      m.frustumCulled = false;
      root.add(m);
    });

    S = {
      doc, rows, tracks, segs, lineSegs, total, peak, cols, nRows, depth, duration,
      cum: new Float32Array(cols * nRows),
      slab, poster, grid, zones, columns, paths, bodies, discs,
      hasPoster: false, colorOf,
      filmUrl: videoUrl || null, filmTex: null, filmCtx: null,
    };
    paint();
    t = Math.min(t, duration);
    applyLayers();
    if (!touched) frameCamera(0);
    apply();
  }

  /* --------------------------------------------------------- theme/colour */
  function paint() {
    if (!S) return;
    S.ramp = rampFrom(RAMP);
    const dark = getComputedStyle(document.documentElement).colorScheme.includes('dark');
    S.slab.material.color.copy(color('--floor'));
    S.slab.material.opacity = dark ? 0.07 : 0.62;
    S.poster.material.opacity = dark ? 0.55 : 0.8;
    [S.grid, S.zones].forEach((m) => m.material.color.copy(color('--floor-line')));
    S.grid.material.opacity = dark ? 0.08 : 0.07;
    S.zones.material.opacity = dark ? 0.3 : 0.24;

    const col = S.paths.geometry.getAttribute('color');
    S.lineSegs.forEach((s, n) => {
      color(S.colorOf(S.tracks[s.k].r), tmp);
      col.setXYZ(n * 2, tmp.r, tmp.g, tmp.b);
      col.setXYZ(n * 2 + 1, tmp.r, tmp.g, tmp.b);
    });
    col.needsUpdate = true;
    S.paths.material.opacity = dark ? 0.7 : 0.55;

    S.tracks.forEach((tr, k) => {
      color(S.colorOf(tr.r), tmp);
      S.bodies.setColorAt(k, tmp);
      S.discs.setColorAt(k, tmp);
    });
    [S.bodies, S.discs].forEach((m) => { if (m.instanceColor) m.instanceColor.needsUpdate = true; });
  }

  function applyLayers() {
    if (!S) return;
    S.columns.visible = layers.heat;
    S.paths.visible = layers.paths;
    S.poster.material.visible = layers.frame && S.hasPoster;
    syncFilm();
    stage.invalidate();
  }

  /* ------------------------------------------------------- state at time t */
  function apply() {
    if (!S) return;
    const { segs, cum, peak, cols, nRows, depth, columns, tracks, bodies, discs, lineSegs, paths, duration } = S;

    cum.fill(0);
    for (const s of segs) {
      if (s.t0 >= t) break;
      cum[s.cell] += Math.min(s.dt, t - s.t0);
    }
    const cw = FLOOR_W / cols, cd = depth / nRows;
    for (let c = 0; c < cols * nRows; c++) {
      const v = cum[c] / peak;
      const col = c % cols, row = (c - col) / cols;
      // The opening animation sweeps across the floor, left to right.
      const k = intro >= 1 ? 1 : easeOut(Math.max(0, Math.min(1, intro * 1.6 - (col / cols) * 0.6)));
      const h = v * MAX_H * k * lift;
      dummy.position.set((col + 0.5) * cw - FLOOR_W / 2, 0, (row + 0.5) * cd - depth / 2);
      if (v * k < 0.0015) dummy.scale.set(0, 0, 0);
      else dummy.scale.set(cw * 0.76, Math.max(h, 0.05), cd * 0.76);
      dummy.updateMatrix();
      columns.setMatrixAt(c, dummy.matrix);
      columns.setColorAt(c, S.ramp(v, tmp));
    }
    columns.instanceMatrix.needsUpdate = true;
    if (columns.instanceColor) columns.instanceColor.needsUpdate = true;

    // Paths drawn so far.
    let lo = 0, hi = lineSegs.length;
    while (lo < hi) { const mid = (lo + hi) >> 1; if (lineSegs[mid].t0 < t) lo = mid + 1; else hi = mid; }
    paths.geometry.setDrawRange(0, lo * 2);
    paths.position.y = (1 - lift) * 0.1;                     // stay above the flattened tiles

    // People are only on the floor while the clock is somewhere inside the recording.
    const live = t < duration - 1e-3;
    dummy.scale.set(1, 1, 1);
    tracks.forEach((tr, k) => {
      const { r, pts } = tr;
      const here = live && t >= r.first_sec && t <= r.last_sec && pts.length;
      if (!here) {
        dummy.position.set(0, -50, 0);
        dummy.scale.set(0, 0, 0);
      } else {
        let i = 0;
        while (i < pts.length - 2 && pts[i + 1][0] <= t) i++;
        const a = pts[i], b = pts[Math.min(i + 1, pts.length - 1)];
        const f = b[0] > a[0] ? Math.max(0, Math.min(1, (t - a[0]) / (b[0] - a[0]))) : 0;
        dummy.position.set(a[1] + (b[1] - a[1]) * f, 0, a[2] + (b[2] - a[2]) * f);
        dummy.scale.set(1, 1, 1);
      }
      dummy.updateMatrix();
      bodies.setMatrixAt(k, dummy.matrix);
      discs.setMatrixAt(k, dummy.matrix);
    });
    bodies.instanceMatrix.needsUpdate = true;
    discs.instanceMatrix.needsUpdate = true;
    syncFilm();
    stage.invalidate();
  }

  /* ---------------------------------------------------------------- camera */
  /* The part of the stage the floor may use. In the overview the headline
     covers the left of the stage, and the time bar always covers the bottom. */
  function region() {
    const W = stage.width, H = stage.height;
    if (mode === 'hero' && W > 900) {
      const x = Math.min(470, W * 0.44) + 8;
      return { x, y: 56, w: W - x - 28, h: H - 56 - 86 };
    }
    if (mode === 'hero') return { x: 18, y: 18, w: W - 36, h: H - 18 - 84 };
    const pad = insets?.() || { top: 70, bottom: 92 };
    return { x: 28, y: pad.top, w: W - 56, h: Math.max(120, H - pad.top - pad.bottom) };
  }
  function frameCamera(ms = 700) {
    if (!stage.width) return;
    const hw = FLOOR_W / 2, hd = (S?.depth || 9) / 2;
    const top = layout === 'top';
    // Everything that has to stay in view: the slab, the tallest column, the zone names.
    const points = [];
    [-hw, hw].forEach((x) => [-hd, hd].forEach((z) => {
      points.push(new T.Vector3(x, 0, z));
      if (!top) points.push(new T.Vector3(x * 0.8, MAX_H * 0.7, z * 0.8));
    }));
    if (!top) points.push(new T.Vector3(0, MAX_H, 0));
    points.push(new T.Vector3(-hw, 0, hd + 0.9), new T.Vector3(hw, 0, hd + 0.9));
    const target = new T.Vector3(0, top ? 0 : MAX_H * 0.3, top ? 0.4 : 0.3);
    const pos = fitCamera(stage, {
      az: top ? 0 : -0.42, el: top ? Math.PI / 2 - 0.002 : 0.6, target, points, region: region(),
    });
    flyTo(stage, pos, target, ms);
  }
  stage.onResize(() => { if (!touched || mode === 'hero') frameCamera(0); });

  /* ------------------------------------------------------------- playback */
  function setTime(next, { emit = false } = {}) {
    if (!S) return;
    t = Math.max(0, Math.min(S.duration, next));
    apply();
    if (emit) onTime?.(t);
  }
  function pause() {
    if (!playing) return;
    playing = false;
    onPlayState?.(false);
  }
  function play() {
    if (!S || playing) return;
    if (t >= S.duration - 1e-3) t = 0;
    playing = true;
    onPlayState?.(true);
    // Long recordings are compressed so a replay always takes about half a minute.
    const speed = Math.max(1, S.duration / 28);
    let last = 0;
    stage.animate((now) => {
      if (!playing || !S) return false;
      const dt = last ? Math.min(0.1, (now - last) / 1000) : 0;
      last = now;
      t = Math.min(S.duration, t + dt * speed);
      apply();
      onTime?.(t);
      if (t >= S.duration) { pause(); return false; }
      return true;
    });
  }

  /* One opening moment: the columns rise in a sweep across the floor. */
  function playIntro() {
    if (reducedMotion() || !S) return;
    intro = 0;
    let start = 0;
    stage.animate((now) => {
      start ||= now;
      intro = Math.min(1, (now - start) / 1300);
      apply();
      return intro < 1;
    });
  }

  /* ---------------------------------------------------------------- hover */
  hoverPicker(stage, () => (S ? [S.bodies, ...(layers.heat ? [S.columns] : [])] : []), {
    onHover(hit, evt) {
      if (hit.object === S.columns) {
        const sec = S.cum[hit.id];
        if (sec <= 0) { hideTip(); return; }
        showTip(evt, 'This spot', [
          { label: 'Time people spent here', value: fmt.dur(sec) },
          { label: 'Against the busiest spot', value: fmt.pct((sec / S.peak) * 100, 0) },
        ]);
      } else {
        const tr = S.tracks[hit.id];
        if (tr) showTip(evt, `Visitor ${tr.r.id}`, describe(tr.r));
      }
    },
    onLeave: hideTip,
  });

  return {
    setData,
    setTime,
    play,
    pause,
    toggle() { (playing ? pause : play)(); },
    get playing() { return playing; },
    get time() { return t; },
    intro: playIntro,
    refreshTheme() { paint(); apply(); },
    setColors(colorOf) { if (S) { S.colorOf = colorOf; paint(); stage.invalidate(); } },
    setLayers(next) { Object.assign(layers, next); applyLayers(); },
    get hasPoster() { return !!S?.hasPoster; },
    setLayout(next) {
      layout = next;
      touched = false;
      frameCamera();
      // Seen from straight above, tall columns lean with perspective and hide
      // their neighbours, so they sink into flat tiles as the camera rises.
      const from = lift, to = next === 'top' ? 0.015 : 1;
      if (reducedMotion()) { lift = to; apply(); return; }
      let start = 0;
      stage.animate((now) => {
        start ||= now;
        const k = Math.min(1, (now - start) / 600);
        lift = from + (to - from) * easeOut(k);
        apply();
        return k < 1;
      });
    },
    setMode(next) {
      mode = next;
      touched = false;
      controls.enableZoom = next === 'full';
      frameCamera(0);
    },
  };
}
