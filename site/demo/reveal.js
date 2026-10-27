// Scroll-in motion for the results. Line charts are inlined (the SVG fetched in place of the <img>) so their parts
// move separately: axes and grid stay put, each series draws along its length, its points appear as the line reaches
// them, labels and reference lines fade in (style.css; classes from tools/web_figures.py). The sampler bars grow one
// after another, tables fade up, the completion percentages count up. Each plays once, when it comes on screen;
// none of it under reduced motion (style.css animates only below html.reveal).

const COUNT_MS = 1100;

function countUp(el) {
  const text = el.textContent.trim();
  const m = text.match(/^(\d+)\.(\d+)%$/);
  if (!m) return;
  const target = Number(`${m[1]}.${m[2]}`), digits = m[2].length;
  const t0 = performance.now();
  const step = (now) => {
    const k = Math.min((now - t0) / COUNT_MS, 1);
    const ease = 1 - (1 - k) ** 4;
    el.textContent = k < 1 ? `${(target * ease).toFixed(digits)}%` : text; // ends on the exact original text
    if (k < 1) requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}

// Swap a figure's <picture> for its SVG inline (the phone or the wide version, as the <picture> would pick, again
// when that changes). On any failure the image stays.
async function inline(fig) {
  const pic = fig.querySelector('picture');
  const img = pic?.querySelector('img.chart'), source = pic?.querySelector('source');
  if (!img || !img.getAttribute('src').endsWith('.svg')) return;
  const holder = document.createElement('div');
  const query = source ? matchMedia(source.media) : null;
  const load = async () => {
    const res = await fetch(query?.matches ? source.getAttribute('srcset') : img.getAttribute('src'));
    if (!res.ok) throw new Error(`${res.url}: HTTP ${res.status}`);
    const svg = new DOMParser().parseFromString(await res.text(), 'image/svg+xml').documentElement;
    if (svg.nodeName !== 'svg') throw new Error('not an SVG');
    svg.classList.add('chart');
    svg.setAttribute('aria-label', img.alt);
    holder.replaceChildren(document.importNode(svg, true));
    if (pic.isConnected) pic.replaceWith(holder);
  };
  query?.addEventListener('change', () => load().catch((e) => console.warn('chart inline:', e)));
  return load().catch((e) => console.warn('chart inline:', e));
}

if (!matchMedia('(prefers-reduced-motion: reduce)').matches && 'IntersectionObserver' in window) {
  document.documentElement.classList.add('reveal');
  let ready = false; // results.js has filled the numbers
  const waiting = new Set();
  const io = new IntersectionObserver((entries) => {
    for (const e of entries) {
      if (!e.isIntersecting) continue;
      io.unobserve(e.target);
      e.target.classList.add('in');
      if (e.target.classList.contains('stat-num')) (ready ? countUp(e.target) : waiting.add(e.target));
    }
  }, { rootMargin: '0px 0px -12% 0px' }); // any part of it past the bottom eighth of the screen (tall tables too)
  for (const el of document.querySelectorAll('.table-panel, .stat-num')) io.observe(el);
  // A chart is watched once it is inline, so it never comes on screen already drawn.
  for (const fig of document.querySelectorAll('.fig')) inline(fig).finally(() => io.observe(fig));
  document.addEventListener('results-ready', () => {
    ready = true;
    for (const el of waiting) countUp(el);
    waiting.clear();
  });
}
