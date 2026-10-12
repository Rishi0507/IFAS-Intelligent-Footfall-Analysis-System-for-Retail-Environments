/* Shared three.js plumbing for the two 3D views.

   A "stage" is one renderer + camera + orbit controls bound to a host element.
   It draws on demand: nothing is rendered while the picture is not changing,
   and nothing at all while the host is hidden (another tab is showing). */

import * as T from './vendor/three.bundle.min.js';

export { T };

export function hasWebGL() {
  try {
    const c = document.createElement('canvas');
    return !!(window.WebGLRenderingContext && (c.getContext('webgl2') || c.getContext('webgl')));
  } catch {
    return false;
  }
}

export const reducedMotion = () => matchMedia('(prefers-reduced-motion: reduce)').matches;

export function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

/* A theme token (`--male` or `var(--male)`) or any CSS colour string, as a
   three.js Color. */
export function color(value, out = new T.Color()) {
  const token = /^var\((--[\w-]+)\)$/.exec(value)?.[1] || (value.startsWith('--') ? value : null);
  return out.set((token ? cssVar(token) : value) || '#888888');
}

/* Interpolate through a list of theme tokens: ramp(0..1) -> Color. */
export function rampFrom(tokens) {
  const stops = tokens.map((t) => color(t));
  return (v, out = new T.Color()) => {
    const x = Math.max(0, Math.min(1, v)) * (stops.length - 1);
    const i = Math.min(stops.length - 2, Math.floor(x));
    return out.copy(stops[i]).lerp(stops[i + 1], x - i);
  };
}

/* A unit box sitting on the ground (y from 0 to 1) with its shading baked into
   vertex colours: lit top, graded sides. Drawn unlit, so instance colours stay
   exactly the theme colours in both light and dark. */
export function litBox() {
  const g = new T.BoxGeometry(1, 1, 1);
  g.translate(0, 0.5, 0);
  const pos = g.getAttribute('position');
  const nrm = g.getAttribute('normal');
  const shade = new Float32Array(pos.count * 3);
  for (let i = 0; i < pos.count; i++) {
    const ny = nrm.getY(i), nx = nrm.getX(i), nz = nrm.getZ(i);
    let s;
    if (ny > 0.5) s = 1;                                   // top
    else if (nz > 0.5) s = 0.86;                           // facing the viewer
    else if (nx > 0.5) s = 0.74;
    else if (nx < -0.5) s = 0.66;
    else s = 0.58;
    if (ny <= 0.5) s *= 0.72 + 0.28 * pos.getY(i);          // sides darken toward the floor
    shade[i * 3] = shade[i * 3 + 1] = shade[i * 3 + 2] = s;
  }
  g.setAttribute('color', new T.BufferAttribute(shade, 3));
  return g;
}

export const easeOut = (x) => 1 - (1 - x) ** 3;
export const easeInOut = (x) => (x < 0.5 ? 4 * x * x * x : 1 - (-2 * x + 2) ** 3 / 2);

export function makeStage(host, { fov = 32 } = {}) {
  const renderer = new T.WebGLRenderer({ antialias: true, alpha: true, powerPreference: 'high-performance' });
  renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
  renderer.setClearColor(0x000000, 0);
  host.appendChild(renderer.domElement);

  const labelLayer = document.createElement('div');
  labelLayer.className = 'stage-labels';
  host.appendChild(labelLayer);

  const scene = new T.Scene();
  const camera = new T.PerspectiveCamera(fov, 1, 0.1, 400);
  const controls = new T.OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.09;
  controls.enablePan = false;
  controls.rotateSpeed = 0.7;
  controls.zoomSpeed = 0.8;

  const labels = [];
  const tickers = new Set();
  const resizers = new Set();
  let dirty = true, raf = 0, interacting = false, width = 0, height = 0;

  const projected = new T.Vector3();
  function placeLabels() {
    for (const l of labels) {
      projected.copy(l.pos).project(camera);
      const off = projected.z > 1 || l.hidden;
      l.el.style.display = off ? 'none' : '';
      if (off) continue;
      const x = (projected.x * 0.5 + 0.5) * width;
      const y = (-projected.y * 0.5 + 0.5) * height;
      l.el.style.transform = `translate(${x.toFixed(1)}px, ${y.toFixed(1)}px) translate(${l.ax}, ${l.ay})`;
    }
  }

  function frame(now) {
    raf = 0;
    if (!width || !height) return;                          // hidden: wait for a resize
    let busy = interacting;
    for (const fn of tickers) busy = fn(now) || busy;
    busy = controls.update() || busy;
    if (dirty || busy) {
      renderer.render(scene, camera);
      placeLabels();
      dirty = false;
    }
    if (busy) request();
  }
  function request() {
    if (!raf) raf = requestAnimationFrame(frame);
  }
  function invalidate() {
    dirty = true;
    request();
  }

  controls.addEventListener('change', invalidate);
  controls.addEventListener('start', () => { interacting = true; request(); });
  controls.addEventListener('end', () => { interacting = false; request(); });

  const ro = new ResizeObserver(() => {
    const w = host.clientWidth, h = host.clientHeight;
    if (w === width && h === height) return;
    width = w; height = h;
    if (!w || !h) return;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    for (const fn of resizers) fn(w, h);
    invalidate();
  });
  ro.observe(host);

  return {
    T, renderer, scene, camera, controls, invalidate,
    flight: 0,
    get width() { return width; },
    get height() { return height; },
    canvas: renderer.domElement,
    /* fn(now) runs each frame while it keeps returning true. */
    animate(fn) {
      const wrapped = (now) => {
        const more = fn(now);
        if (!more) tickers.delete(wrapped);
        return more;
      };
      tickers.add(wrapped);
      request();
      return () => tickers.delete(wrapped);
    },
    onResize(fn) { resizers.add(fn); },
    /* An HTML label pinned to a point in the scene. */
    label(text, pos, { cls = '', ax = '-50%', ay = '-50%' } = {}) {
      const el = document.createElement('span');
      if (cls) el.className = cls;
      el.textContent = text;
      labelLayer.appendChild(el);
      const l = { el, pos: pos.clone(), ax, ay, hidden: false };
      labels.push(l);
      return {
        set text(v) { el.textContent = v; },
        set hidden(v) { l.hidden = v; },
        pos: l.pos,
        remove() { el.remove(); labels.splice(labels.indexOf(l), 1); },
      };
    },
    clearLabels() {
      labels.splice(0).forEach((l) => l.el.remove());
    },
  };
}

/* Find the closest camera position, looking at `target` from the direction
   (az, el), that keeps every one of `points` inside `region` - a pixel box
   within the stage. The view is offset so the camera's axis lands on the
   centre of that box, which is how a scene shares a stage with HTML laid over
   part of it. Returns the position; the offset is applied to the camera. */
export function fitCamera(stage, { az, el, target, points, region }) {
  const { camera } = stage;
  const W = stage.width, H = stage.height;
  const box = region || { x: 0, y: 0, w: W, h: H };
  const tanV = Math.tan((camera.fov * Math.PI) / 360);
  const limY = tanV * (box.h / H), limX = tanV * (box.w / H);

  const dir = new T.Vector3(Math.sin(az) * Math.cos(el), Math.sin(el), Math.cos(az) * Math.cos(el));
  const forward = dir.clone().negate();
  const right = new T.Vector3().crossVectors(forward, new T.Vector3(0, 1, 0));
  if (right.lengthSq() < 1e-6) right.set(1, 0, 0);          // looking straight down
  right.normalize();
  const up = new T.Vector3().crossVectors(right, forward);

  const v = new T.Vector3();
  const fits = (r) => points.every((p) => {
    v.copy(p).sub(target).addScaledVector(dir, -r);          // point relative to the camera
    const z = v.dot(forward);
    return z > 0.1 && Math.abs(v.dot(right)) <= limX * z && Math.abs(v.dot(up)) <= limY * z;
  });
  let lo = 1, hi = 400;
  for (let i = 0; i < 40; i++) {
    const mid = (lo + hi) / 2;
    if (fits(mid)) hi = mid; else lo = mid;
  }

  const dx = box.x + box.w / 2 - W / 2, dy = box.y + box.h / 2 - H / 2;
  if (Math.abs(dx) > 0.5 || Math.abs(dy) > 0.5) camera.setViewOffset(W, H, -dx, -dy, W, H);
  else camera.clearViewOffset();
  return target.clone().addScaledVector(dir, hi);
}

/* Ease the camera to a new position/target. */
export function flyTo(stage, pos, target, ms = 700) {
  const { camera, controls } = stage;
  const flight = (stage.flight = (stage.flight || 0) + 1);   // a newer flight cancels this one
  if (reducedMotion() || ms <= 0) {
    camera.position.copy(pos);
    controls.target.copy(target);
    controls.update();
    stage.invalidate();
    return;
  }
  const p0 = camera.position.clone(), t0 = controls.target.clone();
  let start = 0;
  stage.animate((now) => {
    if (stage.flight !== flight) return false;
    start ||= now;
    const k = Math.min(1, (now - start) / ms);
    const e = easeInOut(k);
    camera.position.lerpVectors(p0, pos, e);
    controls.target.lerpVectors(t0, target, e);
    stage.invalidate();
    return k < 1;
  });
}

/* Pointer -> instanced mesh hit testing with hover/leave callbacks. */
export function hoverPicker(stage, getTargets, { onHover, onLeave, onClick }) {
  const ray = new T.Raycaster();
  const ndc = new T.Vector2();
  const el = stage.canvas;
  let last = null, downAt = null;

  function pick(evt) {
    const r = el.getBoundingClientRect();
    ndc.set(((evt.clientX - r.left) / r.width) * 2 - 1, -((evt.clientY - r.top) / r.height) * 2 + 1);
    ray.setFromCamera(ndc, stage.camera);
    const hits = ray.intersectObjects(getTargets(), false);
    return hits.length ? { object: hits[0].object, id: hits[0].instanceId } : null;
  }
  el.addEventListener('pointermove', (evt) => {
    if (evt.buttons) return;                                // dragging the camera
    const hit = pick(evt);
    const key = hit ? `${hit.object.uuid}:${hit.id}` : null;
    if (hit) onHover(hit, evt, key !== last);
    else if (last) onLeave();
    last = key;
  });
  el.addEventListener('pointerleave', () => { if (last) onLeave(); last = null; });
  el.addEventListener('pointerdown', (evt) => { downAt = [evt.clientX, evt.clientY]; });
  el.addEventListener('pointerup', (evt) => {
    if (!onClick || !downAt) return;
    const moved = Math.hypot(evt.clientX - downAt[0], evt.clientY - downAt[1]);
    downAt = null;
    if (moved > 5) return;                                  // that was an orbit, not a click
    const hit = pick(evt);
    if (hit) onClick(hit, evt);
  });
}
