/* The skyline: one block per group of people, on a small grid.

   Groups run left to right (age buckets), series run front to back (women,
   men). Height is whatever the caller is comparing, so the same scene shows
   head counts or time stayed. Blocks ease to their new height when the data
   or a filter changes, which is the only motion here. */

import { T, makeStage, color, litBox, fitCamera, hoverPicker, reducedMotion } from './three-kit.js';

const MAX_H = 3.2;
const STEP_X = 1.55, STEP_Z = 1.45, FOOT = 1.02;

export function createSkyline(host, { showTip, hideTip, onPick }) {
  const stage = makeStage(host, { fov: 28 });
  const { scene, camera, controls } = stage;
  controls.enableZoom = false;
  controls.minPolarAngle = 0.5;
  controls.maxPolarAngle = 1.35;
  controls.minAzimuthAngle = -0.2;
  controls.maxAzimuthAngle = 1.25;

  const root = new T.Group();
  scene.add(root);
  const dummy = new T.Object3D();
  const tmp = new T.Color();

  let D = null;            // current data
  let bars = null, slab = null;
  let heights = [], targets = [], valueLabels = [];
  let shape = '', hovered = -1, touched = false, animating = false;
  controls.addEventListener('start', () => { touched = true; });

  const xOf = (g) => (g - (D.groups.length - 1) / 2) * STEP_X;
  const zOf = (s) => (s - (D.series.length - 1) / 2) * STEP_Z;

  function rebuild() {
    stage.clearLabels();
    root.traverse((o) => { o.geometry?.dispose(); o.material?.dispose(); });
    root.clear();
    const nG = D.groups.length, nS = D.series.length;

    slab = new T.Mesh(
      new T.BoxGeometry(nG * STEP_X + 0.5, 0.12, nS * STEP_Z + 0.5),
      new T.MeshBasicMaterial({ transparent: true, depthWrite: false }),
    );
    slab.position.y = -0.065;
    slab.renderOrder = -1;
    root.add(slab);

    bars = new T.InstancedMesh(litBox(), new T.MeshBasicMaterial({ vertexColors: true }), nG * nS);
    bars.instanceMatrix.setUsage(T.DynamicDrawUsage);
    bars.frustumCulled = false;
    root.add(bars);

    // The camera sits front right, so names go on the two near edges.
    const front = (nS * STEP_Z) / 2 + 0.55, side = (nG * STEP_X) / 2 + 0.45;
    D.groups.forEach((g, i) => stage.label(g.label, new T.Vector3(xOf(i), 0, front), { cls: 'axis' }));
    D.series.forEach((s, j) => stage.label(s.label, new T.Vector3(side, 0, zOf(j)), { cls: 'axis', ax: '0%' }));

    heights = Array(nG * nS).fill(0);
    valueLabels = heights.map((_, n) => stage.label('', new T.Vector3(xOf(n % nG), 0, zOf((n - (n % nG)) / nG)),
      { cls: 'strong', ay: '-130%' }));
    if (!touched) frameCamera();
  }

  function frameCamera() {
    if (!D || !stage.width) return;
    const hw = (D.groups.length * STEP_X) / 2 + 0.3, hd = (D.series.length * STEP_Z) / 2 + 0.3;
    // Keep the slab, the tallest block with its number, and both rows of names in view.
    const points = [];
    [-hw, hw].forEach((x) => [-hd, hd].forEach((z) => {
      points.push(new T.Vector3(x, 0, z), new T.Vector3(x * 0.8, MAX_H + 0.4, z * 0.8));
    }));
    points.push(new T.Vector3(hw + 1.3, 0, hd), new T.Vector3(hw + 1.3, 0, -hd), new T.Vector3(-hw, 0, hd + 0.8));
    const target = new T.Vector3(0, MAX_H * 0.4, 0);
    camera.position.copy(fitCamera(stage, {
      az: 0.46, el: 0.5, target, points,
      region: { x: 10, y: 6, w: stage.width - 20, h: stage.height - 12 },
    }));
    controls.target.copy(target);
    controls.update();
    stage.invalidate();
  }
  stage.onResize(() => { if (!touched) frameCamera(); });

  function paint() {
    if (!D) return;
    const dark = getComputedStyle(document.documentElement).colorScheme.includes('dark');
    slab.material.color.copy(color('--floor'));
    slab.material.opacity = dark ? 0.07 : 0.62;
    const nG = D.groups.length;
    heights.forEach((_, n) => {
      const s = (n - (n % nG)) / nG;
      color(D.series[s].color, tmp);
      // Hovering one block lets the rest fall back.
      if (hovered >= 0 && hovered !== n) tmp.lerp(color('--plane'), 0.55);
      bars.setColorAt(n, tmp);
    });
    if (bars.instanceColor) bars.instanceColor.needsUpdate = true;
    stage.invalidate();
  }

  function place() {
    const nG = D.groups.length;
    heights.forEach((h, n) => {
      const g = n % nG, s = (n - g) / nG;
      dummy.position.set(xOf(g), 0, zOf(s));
      dummy.scale.set(FOOT, Math.max(h, 0.02), FOOT);
      dummy.updateMatrix();
      bars.setMatrixAt(n, dummy.matrix);
      valueLabels[n].pos.y = Math.max(h, 0.02);
    });
    bars.instanceMatrix.needsUpdate = true;
    stage.invalidate();
  }

  function settle() {
    if (reducedMotion()) {
      heights = targets.slice();
      place();
      return;
    }
    if (animating) return;
    animating = true;
    stage.animate(() => {
      let moving = false;
      heights = heights.map((h, n) => {
        const d = targets[n] - h;
        if (Math.abs(d) < 0.002) return targets[n];
        moving = true;
        return h + d * 0.16;
      });
      place();
      animating = moving;
      return moving;
    });
  }

  /* data: { groups: [{key, label}], series: [{key, label, color}],
             values: [series][group] numbers, text: [series][group] strings,
             tips: [series][group] tooltip rows } */
  function setData(data) {
    const nextShape = `${data.groups.map((g) => g.key)}|${data.series.map((s) => s.key)}`;
    D = data;
    if (nextShape !== shape) { shape = nextShape; rebuild(); }
    const nG = D.groups.length;
    const max = Math.max(0, ...D.values.flat()) || 1;
    targets = heights.map((_, n) => (D.values[(n - (n % nG)) / nG][n % nG] / max) * MAX_H);
    valueLabels.forEach((l, n) => { l.text = D.text[(n - (n % nG)) / nG][n % nG]; });
    paint();
    settle();
  }

  hoverPicker(stage, () => (bars ? [bars] : []), {
    onHover(hit, evt, changed) {
      const nG = D.groups.length, n = hit.id;
      const g = n % nG, s = (n - g) / nG;
      showTip(evt, `${D.series[s].label}, ${D.groups[g].label}`, D.tips[s][g]);
      if (changed) { hovered = n; paint(); }
    },
    onLeave() {
      hideTip();
      hovered = -1;
      paint();
    },
    onClick(hit) {
      const nG = D.groups.length, g = hit.id % nG, s = (hit.id - g) / nG;
      onPick?.(D.series[s].key, D.groups[g].key);
    },
  });

  return { setData, refreshTheme: paint };
}
