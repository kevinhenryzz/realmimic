// Top bar: mark the section being read; on phones, where the links are a sideways-scrolling row, keep the
// marked one in view.

const links = new Map(
  [...document.querySelectorAll('.topbar nav a[href^="#"]')].map((a) => [a.getAttribute('href').slice(1), a]),
);
const sections = [...links.keys()].map((id) => document.getElementById(id)).filter(Boolean);
const nav = document.querySelector('.topbar nav');
let current = null;

function mark(id) {
  if (id === current) return;
  current = id;
  for (const [key, a] of links) a.classList.toggle('active', key === id);
  const a = links.get(id);
  if (a && nav.scrollWidth > nav.clientWidth) {
    nav.scrollTo({ left: a.offsetLeft - (nav.clientWidth - a.offsetWidth) / 2, behavior: 'smooth' });
  }
}

// The section whose top has passed a line a third of the way down the screen.
function update() {
  const line = window.innerHeight / 3;
  let id = null;
  for (const s of sections) if (s.getBoundingClientRect().top <= line) id = s.id;
  mark(id);
}

let queued = false;
window.addEventListener('scroll', () => {
  if (queued) return;
  queued = true;
  requestAnimationFrame(() => {
    queued = false;
    update();
  });
}, { passive: true });
window.addEventListener('resize', update);
update();

// Desktop: while the montage plays, the demo frame's height is what is left of the first screen below its top
// (--fit-h; style.css bounds it so the cropped montage keeps the robot whole). Measured again once fonts load.
const stage = document.getElementById('stage');
function fit() {
  const top = stage.getBoundingClientRect().top + window.scrollY;
  document.documentElement.style.setProperty('--fit-h', `${Math.max(0, window.innerHeight - top - 16)}px`);
}
if (stage) {
  window.addEventListener('resize', fit);
  document.fonts?.ready.then(fit);
  fit();
}

// Phones: the secondary tables start folded ("Show the table"), and long grey notes and explanations show three lines
// with a "more" link (style.css). Run again once results.js has filled its notes.
const phone = matchMedia('(max-width: 600px)');
const CLAMP = '#results .note:not(.credit), .finding > p, #res-notmeasurable li, #deploy > p, #deploy .note, #deviations .note'; // licence credits stay whole
function clamp(el) {
  if (el.dataset.clamp || !el.textContent.trim() || !el.offsetParent) return;
  el.classList.add('clamp');
  if (el.scrollHeight <= el.clientHeight + 2) return void el.classList.remove('clamp'); // three lines or fewer
  el.dataset.clamp = '1';
  const more = Object.assign(document.createElement('button'), { type: 'button', className: 'more-btn', textContent: 'more' });
  more.addEventListener('click', () => { more.textContent = el.classList.toggle('clamp') ? 'more' : 'less'; });
  el.after(more);
}
if (phone.matches) {
  for (const d of document.querySelectorAll('details.fold')) {
    d.open = false;
    const label = d.querySelector('summary');
    d.addEventListener('toggle', () => { label.textContent = d.open ? 'Hide the table' : 'Show the table'; });
  }
  const clampAll = () => { for (const el of document.querySelectorAll(CLAMP)) clamp(el); };
  document.addEventListener('results-ready', clampAll);
  document.fonts?.ready.then(clampAll);
  clampAll();
}
