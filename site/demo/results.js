// Fills the page's result sections from assets/results/*.json (written from the project's aggregate results,
// runs/web, by tools/web_figures.py --web). Values the files mark "pending" are shown as pending; nothing is hard-coded.

const url = (f) => new URL(`../assets/results/${f}`, import.meta.url);
const get = (f) => fetch(url(f)).then((r) => (r.ok ? r.json() : Promise.reject(new Error(`${f}: HTTP ${r.status}`))));
const $ = (id) => document.getElementById(id);
const n = (v) => (typeof v === 'number' ? v.toLocaleString('en') : String(v));
const ret = (v) => (typeof v === 'number' ? v.toFixed(4) : String(v));
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c]);
const pending = (v) => v === 'pending' || v === null || v === undefined;
const cell = (v, fmt = n) => (pending(v) ? '<td class="pending">pending</td>' : `<td>${esc(fmt(v))}</td>`);

function table(head, rows, cls = '') {
  return `<table class="${cls}"><thead><tr>${head.map((h) => `<th>${esc(h)}</th>`).join('')}</tr></thead><tbody>${rows.join('')}</tbody></table>`;
}

function headline(r) {
  const rel = r.released, seed = r.second_seed, tail = r.kernel_tail_run, sonic = r.sonic_release, paper = r.paper;
  // The evaluator column names the GPU that scored each run (kept on one line).
  const scored = (g) => esc(g ? `this project (${g.replace(/ /g, ' ')})` : 'this project');
  // "final, 86k" with the last three checkpoints' failures on a grey line under it.
  const lastOf = (o) => {
    const its = Object.keys(o).map((it) => `${Number(it) / 1000}k`).join(', ');
    return `the last three saved models (${its}): ${Object.values(o).map((v) => (pending(v) ? 'pending' : n(v))).join(', ')} failures`;
  };
  const ckpt = (o) => `after 86,000 iterations<span class="sub">${esc(lastOf(o))}</span>`;
  const units = (t) => esc(t).replace(/(\d) (GB|h)\b/g, '$1\u00a0$2'); // "24 GB", "10.5 h" on one line
  // Completion first: completed of all, as a percentage rounded down (never up), failures second.
  const pct = (failedN, clips) => `${(Math.floor((1e5 * (clips - failedN)) / clips) / 1e3).toFixed(3)}%`;
  $('res-pct').textContent = pct(rel.failed, rel.clips);
  $('res-done').textContent = n(rel.clips - rel.failed);
  $('res-clips').textContent = n(rel.clips);
  $('res-failed').textContent = n(rel.failed);
  $('res-return').textContent = ret(rel.return);
  $('res-seed-pct').textContent = `second training seed: ${n(seed.failed)} failed (${pct(seed.failed, rel.clips)})`;
  $('res-sonic-pct').textContent = pct(sonic.failed, rel.clips);
  $('res-sonic-done').textContent = n(rel.clips - sonic.failed);
  $('res-sonic-clips').textContent = n(rel.clips);
  $('res-sonic').textContent = n(sonic.failed);
  $('res-paper-pct').textContent = pct(paper.failed, paper.clips); // the paper's filter on its release of the data
  $('res-paper-done').textContent = n(paper.clips - paper.failed);
  $('res-paper-clips').textContent = n(paper.clips);
  $('res-paper').textContent = n(paper.failed);
  $('res-table').innerHTML = table(['Policy', 'Model', 'Scored by', 'Training', 'Tracking score', 'Failed'], [
    `<tr><td>realmimic, released</td><td>${ckpt(rel.last_three)}</td><td>${scored(rel.scored_on)}</td><td>${units(rel.training)}</td>${cell(rel.return, ret)}${cell(rel.failed)}</tr>`,
    `<tr><td>${esc(seed.label)}</td><td>${ckpt(seed.last_three)}</td><td>${scored(seed.scored_on)}</td><td>${units(seed.training)}</td>${cell(seed.return, ret)}${cell(seed.failed)}</tr>`,
    `<tr><td>${esc(tail.label)}<span class="sub">${esc(tail.note)}</span></td><td>final saved model</td><td>${scored(tail.scored_on)}</td><td>${units(tail.training)}</td>${cell(tail.return, ret)}${cell(tail.failed)}</tr>`,
    `<tr><td>SONIC release, its own gains</td><td>NVIDIA's release</td><td>${scored(sonic.scored_on)}</td><td>NVIDIA</td>${cell(sonic.return, ret)}${cell(sonic.failed)}</tr>`,
    `<tr><td>SuperMimic, as reported</td><td>after 3 h</td><td>the paper's own evaluation</td><td>${units('one H100 80 GB, 3 h')}</td><td>—</td>${cell(paper.failed)}</tr>`,
  ], 'results');
  $('res-hw-note').textContent = r.hardware_note;
  $('res-paper-note').textContent = `The paper's evaluation counts ${n(paper.sonic_failed)} failures for SONIC; ours counts ${n(sonic.failed)}.`;
}

// The motion-type table as the README has it (tools/web_figures.py writes its columns and rows).
function groups(g) {
  const ours = g.header.indexOf('realmimic');
  $('res-groups').innerHTML = table(g.header, g.rows.map((row, i) => `<tr class="${i === g.rows.length - 1 ? 'total' : ''}">${row.map((c, j) => `<td class="${j === ours ? 'ours' : ''}">${esc(c)}</td>`).join('')}</tr>`), 'results groups');
}

function lafan(r) {
  const l = r.lafan1;
  // Both training seeds of ours (the released weights are seed 0) next to SONIC's release; failures with their
  // share of the clips.
  const failed = (d, clips) => (pending(d?.failed) ? undefined : `${n(d.failed)} (${((100 * d.failed) / clips).toFixed(1)}%)`);
  const row = (name, k) => {
    const d = l[k];
    return `<tr><td>${esc(name)} (${n(d.clips)})</td>${cell(d.ours.return, ret)}${cell(d.ours_seed1?.return, ret)}${cell(d.sonic_release.return, ret)}`
      + `${cell(failed(d.ours, d.clips), String)}${cell(failed(d.ours_seed1, d.clips), String)}${cell(failed(d.sonic_release, d.clips), String)}</tr>`;
  };
  $('res-lafan').innerHTML = table(['LAFAN1, never trained on (clips)', 'Tracking score, seed 0', 'Tracking score, seed 1', 'Tracking score, SONIC', 'Failed, seed 0', 'Failed, seed 1', 'Failed, SONIC'],
    [row('whole clips', 'whole_clips'), row('10-second segments', 'segments_10s')], 'results lafan');
  $('res-lafan-note').textContent = `${l.note} ${l.hardware_note}`;
}

function findings(r) {
  // Flips still failing on the 130 hardest clips at 12k iterations, old estimate against the fix, for two pairs of
  // runs that differ in the estimate alone.
  const rows = Object.entries(r.practice_fix).flatMap(([name, p]) => [[`${name}, old estimate`, p.old_estimate, 'warn'], [`${name}, with the fix`, p.with_the_fix, 'ok']]);
  const top = Math.max(...rows.map(([, v]) => v), 1);
  $('res-sampler-bars').innerHTML = rows.map(([label, v, cls]) => `<div class="bar-row"><span class="bar-label">${esc(label)}</span><span class="bar-track"><span class="bar-val ${cls}" style="width: ${Math.max((100 * v) / top, 0.8)}%"></span></span><span class="bar-num">${n(v)}</span></div>`).join('');
  $('res-sampler-bars').setAttribute('aria-label', `Flips still failing on the 130 hardest clips at 12k iterations: ${rows.map(([l, v]) => `${l} ${v}`).join(', ')}.`);
  $('res-notmeasurable').innerHTML = r.not_measurable.map((f) => `<li><strong>${esc(f.title)}.</strong> ${esc(f.text)}</li>`).join('');
}

function compute(r) {
  const c = r.compute;
  $('res-compute').innerHTML = table(['What', 'Hardware', 'GPU-hours'], [
    ...c.rows.map(([what, hw, h]) => `<tr><td>${esc(what)}</td><td>${esc(hw)}</td><td>${esc(h)}</td></tr>`),
    `<tr class="total"><td>Total</td><td></td><td>${esc(c.total)}</td></tr>`,
  ], 'compute');
  $('res-compute-excl').textContent = c.not_included;
}

// Each cell of a result table gets its column's header as data-label: on phones the rows become cards that
// show the label next to the value (style.css).
function labelCells() {
  for (const t of document.querySelectorAll('table.results')) {
    const head = [...t.querySelectorAll('thead th')].map((th) => th.textContent.trim());
    for (const tr of t.querySelectorAll('tbody tr')) [...tr.children].forEach((td, i) => td.setAttribute('data-label', head[i] ?? ''));
  }
}

Promise.all([get('results.json'), get('groups.json')])
  .then(([r, g]) => {
    for (const fill of [() => headline(r), () => groups(g), () => lafan(r), () => findings(r), () => compute(r)]) {
      try {
        fill();
      } catch (e) {
        console.error(e);
      }
    }
    labelCells();
    document.dispatchEvent(new Event('results-ready')); // reveal.js counts the percentages up from here
  })
  .catch((e) => console.error('results:', e));

// Comparison videos (in section 02, Results): a video starts from its beginning when at least half of it comes on
// screen (so the visitor sees the whole move, SONIC's fall included), loops while it stays, and pauses when it leaves.
// It is downloaded once it comes within about a screen of view (preload="none" until then), so it is ready when
// reached. Reduced-motion and Save-Data visitors get the browser's own controls instead and start videos themselves.
const videos = [...document.querySelectorAll('.video-card video')];
const autoplay = !matchMedia('(prefers-reduced-motion: reduce)').matches && !navigator.connection?.saveData;
if (autoplay) {
  const near = new IntersectionObserver((entries) => {
    for (const e of entries) {
      if (!e.isIntersecting) continue;
      const v = e.target;
      near.unobserve(v);
      if (v.dataset.started) continue; // already playing (load() would cancel it)
      v.preload = 'auto';
      v.load();
    }
  }, { rootMargin: '100% 100%' }); // a screen ahead, also sideways (the phone row)
  const seen = new IntersectionObserver((entries) => {
    for (const e of entries) {
      const v = e.target;
      if (e.isIntersecting) {
        v.dataset.started = '1';
        v.currentTime = 0;
        v.play().catch(() => {});
      } else v.pause();
    }
  }, { threshold: 0.5 });
  for (const v of videos) {
    v.loop = true;
    near.observe(v);
    seen.observe(v);
  }
} else for (const v of videos) v.controls = true;

// The strip under each comparison video: per pane, the latest event so far ("on track", "off track 2.3 s",
// "fell 2.4 s"; assets/videos/cues.json), back to "on track" when the video loops.
get('../videos/cues.json').then((c) => {
  for (const v of document.querySelectorAll('.video-card video')) {
    const cue = c.videos.find((x) => v.getAttribute('src').endsWith(x.file));
    const ems = v.closest('.video-card').querySelectorAll('.vs em');
    if (!cue || ems.length !== cue.panes.length) continue;
    const show = () => {
      const t = v.currentTime;
      cue.panes.forEach((p, i) => {
        let state = 'ok', text = 'on track';
        if (p.off_track !== null && t >= p.off_track) [state, text] = ['off', `off track ${p.off_track.toFixed(1)} s`];
        if (p.fell !== null && t >= p.fell) [state, text] = ['fell', `fell ${p.fell.toFixed(1)} s`];
        if (ems[i].textContent !== text) Object.assign(ems[i], { textContent: text }).dataset.state = state;
      });
    };
    if ('requestVideoFrameCallback' in v) {
      const frame = () => (show(), v.requestVideoFrameCallback(frame));
      v.requestVideoFrameCallback(frame);
    }
    v.addEventListener('timeupdate', show);
    v.addEventListener('seeked', show);
  }
}).catch((e) => console.error('video cues:', e));
