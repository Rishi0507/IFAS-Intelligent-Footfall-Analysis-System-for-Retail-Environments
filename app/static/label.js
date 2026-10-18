// Labelling page: one person at a time, every crop of them in a strip, keyboard
// first. Writes labels_<annotator>.json in the run folder through the API, one
// file per annotator so the two can be compared (docs/research-plan.md, S2).
'use strict';

const $ = (id) => document.getElementById(id);
const AGES = ['Age16-30', 'Age31-45', 'Age46-60', 'AgeAbove61'];
const NAME_OK = /^[A-Za-z0-9_-]{1,32}$/;

const state = {
  run: null,        // run id
  people: [],       // [{track, crops: [{frame, t, url}]}]
  labels: {},       // track -> {gender, age, note}
  index: 0,
  annotator: '',
  saveTimer: null,
  saving: null,
  loading: null,    // promise while this annotator's earlier labels load
};

function store(key, value) {
  try { value == null ? localStorage.removeItem(key) : localStorage.setItem(key, value); }
  catch { /* storage blocked: the page still works, it just forgets */ }
}
function recall(key) {
  try { return localStorage.getItem(key); } catch { return null; }
}

async function getJSON(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || res.statusText);
  return res.json();
}

function labelOf(track) {
  return state.labels[track] || (state.labels[track] = { gender: null, age: null, note: '' });
}
function status(track) {
  const l = state.labels[track];
  if (!l || (!l.gender && !l.age)) return 'empty';
  return l.gender && l.age ? 'done' : 'partial';
}

// ------------------------------------------------------------------ saving
function setSaveState(text, kind) {
  const el = $('save-state');
  el.textContent = text;
  el.dataset.state = kind || '';
}

function scheduleSave() {
  clearTimeout(state.saveTimer);
  setSaveState('Saving…');
  state.saveTimer = setTimeout(save, 300);
}

async function save() {
  if (!state.run || !NAME_OK.test(state.annotator)) return;
  const labels = {};
  for (const [track, l] of Object.entries(state.labels)) {
    if (l.gender || l.age || l.note) labels[track] = l;
  }
  const body = JSON.stringify({ annotator: state.annotator, labels });
  const run = state.run;
  state.saving = (async () => {
    try {
      const res = await fetch(`/api/runs/${encodeURIComponent(run)}/labels`, {
        method: 'POST', headers: { 'content-type': 'application/json' }, body,
      });
      if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || res.statusText);
      if (run === state.run) setSaveState('Saved', 'saved');
    } catch (err) {
      setSaveState(`Not saved: ${err.message}`, 'error');
    }
  })();
  await state.saving;
}

async function flush() {
  if (state.saveTimer) { clearTimeout(state.saveTimer); state.saveTimer = null; await save(); }
  if (state.saving) await state.saving;
}

// ---------------------------------------------------------------- drawing
function renderPerson() {
  const person = state.people[state.index];
  if (!person) return;
  const l = labelOf(person.track);
  const first = person.crops[0], last = person.crops[person.crops.length - 1];
  $('person-title').textContent = `Person ${person.track}`;
  $('person-meta').textContent =
    `${state.index + 1} of ${state.people.length} · ${person.crops.length} crops · ` +
    `${first.t.toFixed(1)}–${last.t.toFixed(1)} s`;

  const strip = $('strip');
  strip.replaceChildren(...person.crops.map((c) => {
    const fig = document.createElement('figure');
    const img = document.createElement('img');
    img.src = c.url;
    img.alt = `Person ${person.track} at ${c.t.toFixed(1)} s`;
    img.loading = 'lazy';
    const cap = document.createElement('figcaption');
    cap.textContent = `${c.t.toFixed(1)} s`;
    fig.append(img, cap);
    return fig;
  }));
  strip.scrollLeft = 0;

  for (const b of $('seg-gender').querySelectorAll('button')) {
    b.setAttribute('aria-pressed', String(b.dataset.value === l.gender));
  }
  for (const b of $('seg-age').querySelectorAll('button')) {
    b.setAttribute('aria-pressed', String(b.dataset.value === l.age));
  }
  $('note').value = l.note || '';
  $('prev').disabled = state.index === 0;
  $('next').textContent = state.index === state.people.length - 1 ? 'Finish' : 'Next →';
  renderRoster();
}

function renderRoster() {
  const roster = $('roster');
  if (roster.childElementCount !== state.people.length) {
    roster.replaceChildren(...state.people.map((p, i) => {
      const b = document.createElement('button');
      b.type = 'button';
      b.setAttribute('role', 'listitem');
      b.textContent = `#${p.track}`;
      b.addEventListener('click', () => go(i));
      return b;
    }));
  }
  let done = 0;
  state.people.forEach((p, i) => {
    const b = roster.children[i];
    const s = status(p.track);
    if (s === 'done') done += 1;
    b.dataset.state = s;
    b.setAttribute('aria-current', String(i === state.index));
    b.title = `Person ${p.track}: ${s === 'done' ? 'labelled' : s === 'partial' ? 'half labelled' : 'not labelled'}`;
  });
  $('progress').textContent = `${done} of ${state.people.length} labelled`;
  $('meter').style.width = `${state.people.length ? (100 * done) / state.people.length : 0}%`;
  // Keep the current person in view inside the list, without scrolling the page.
  const cur = roster.children[state.index];
  if (cur) {
    const top = cur.offsetTop - roster.offsetTop;
    if (top < roster.scrollTop) roster.scrollTop = top;
    else if (top + cur.offsetHeight > roster.scrollTop + roster.clientHeight) {
      roster.scrollTop = top + cur.offsetHeight - roster.clientHeight;
    }
  }
}

function go(i) {
  state.index = Math.max(0, Math.min(state.people.length - 1, i));
  renderPerson();
}

function nextUnlabelled() {
  const n = state.people.length;
  for (let k = 1; k <= n; k += 1) {
    const i = (state.index + k) % n;
    if (status(state.people[i].track) !== 'done') return i;
  }
  return Math.min(state.index + 1, n - 1);
}

// --------------------------------------------------------------- answering
function requireName() {
  if (NAME_OK.test(state.annotator)) return true;
  setSaveState('Enter your name first, so your labels go in your own file.', 'error');
  $('annotator').focus();
  return false;
}

async function answer(field, value) {
  if (state.loading) await state.loading;   // never let arriving labels overwrite a new answer
  if (!requireName()) return;
  const person = state.people[state.index];
  const l = labelOf(person.track);
  const wasDone = status(person.track) === 'done';
  l[field] = l[field] === value ? null : value;          // pressing it again clears it
  scheduleSave();
  renderPerson();
  if (!wasDone && status(person.track) === 'done' && $('auto').checked) {
    setTimeout(() => go(nextUnlabelled()), 220);
  }
}

// ------------------------------------------------------------------ loading
async function loadRun(runId) {
  await flush();
  state.run = runId;
  state.index = 0;
  const url = new URL(location.href);
  url.searchParams.set('run', runId);
  history.replaceState(null, '', url);
  const data = await getJSON(`/api/runs/${encodeURIComponent(runId)}/crops`);
  state.people = data.people;
  $('roster').replaceChildren();
  await loadLabels();
  $('work').hidden = false;
  if (!state.people.length) {
    $('person-title').textContent = 'No people were classified in this recording';
    $('person-meta').textContent = '';
    $('strip').replaceChildren();
    $('progress').textContent = '';
    return;
  }
  const firstOpen = state.people.findIndex((p) => status(p.track) !== 'done');
  go(firstOpen >= 0 ? firstOpen : 0);
}

function loadLabels() {
  state.loading = (async () => {
    let labels = {};
    if (state.run && NAME_OK.test(state.annotator)) {
      const doc = await getJSON(
        `/api/runs/${encodeURIComponent(state.run)}/labels/${encodeURIComponent(state.annotator)}`);
      labels = doc.labels || {};
      setSaveState(Object.keys(labels).length ? 'Loaded your earlier labels' : '');
    } else {
      setSaveState('Enter your name to start.');
    }
    state.labels = labels;
  })().finally(() => { state.loading = null; });
  return state.loading;
}

async function init() {
  const params = new URLSearchParams(location.search);
  state.annotator = params.get('annotator') || recall('ifas-annotator') || '';
  $('annotator').value = state.annotator;

  const runs = (await getJSON('/api/runs')).filter((r) => r.has_crops);
  if (!runs.length) { $('empty').hidden = false; return; }
  const select = $('run');
  select.replaceChildren(...runs.map((r) => {
    const o = document.createElement('option');
    o.value = r.id;
    o.textContent = `${r.source || r.id} · ${r.visitors} people`;
    return o;
  }));
  const wanted = params.get('run');
  select.value = runs.some((r) => r.id === wanted) ? wanted : runs[0].id;
  await loadRun(select.value);
  select.addEventListener('change', () => loadRun(select.value));
}

// ------------------------------------------------------------------ events
$('annotator').addEventListener('change', async (e) => {
  await flush();
  const name = e.target.value.trim();
  state.annotator = NAME_OK.test(name) ? name : '';
  store('ifas-annotator', state.annotator || null);
  await loadLabels();
  renderPerson();
});

$('seg-gender').addEventListener('click', (e) => {
  const b = e.target.closest('button');
  if (b) answer('gender', b.dataset.value);
});
$('seg-age').addEventListener('click', (e) => {
  const b = e.target.closest('button');
  if (b) answer('age', b.dataset.value);
});
$('note').addEventListener('input', (e) => {
  if (state.loading || !requireName()) return;
  labelOf(state.people[state.index].track).note = e.target.value;
  scheduleSave();
});
$('prev').addEventListener('click', () => go(state.index - 1));
$('next').addEventListener('click', () => {
  if (state.index === state.people.length - 1) { flush(); renderRoster(); return; }
  go(state.index + 1);
});

document.addEventListener('keydown', (e) => {
  if (e.ctrlKey || e.metaKey || e.altKey || $('work').hidden) return;
  const typing = e.target.matches('input, select, textarea');
  if (typing) {
    if (e.key === 'Escape' || (e.key === 'Enter' && e.target.id === 'note')) e.target.blur();
    return;
  }
  const k = e.key.toLowerCase();
  const actions = {
    f: () => answer('gender', 'female'),
    m: () => answer('gender', 'male'),
    u: () => answer('gender', 'unsure'),
    0: () => answer('age', 'unsure'),
    arrowleft: () => go(state.index - 1),
    arrowright: () => go(state.index + 1),
    n: () => $('note').focus(),
  };
  AGES.forEach((a, i) => { actions[String(i + 1)] = () => answer('age', a); });
  if (actions[k]) { e.preventDefault(); actions[k](); }
});

window.addEventListener('beforeunload', (e) => {
  if (state.saveTimer) { save(); e.preventDefault(); }
});

init().catch((err) => {
  $('empty').hidden = false;
  $('empty').querySelector('.sub').textContent = `Could not load the recordings: ${err.message}`;
});
