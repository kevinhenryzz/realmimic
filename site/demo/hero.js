// The opening view: a recorded montage of the released policy, rendered in this page's own scene with the
// demo's default camera (tools/web_hero_render.mjs), plays while the live demo loads behind it. "Try it live"
// crossfades to the canvas once the policy is ready; clicked earlier, the button shows the loading progress
// and switches by itself when loading is done.

const FADE_MS = 500;

export class Hero {
  constructor(root, { cuesUrl }) {
    this.root = root;
    this.video = root.querySelector('video');
    this.caption = root.querySelector('.hero-move');
    this.button = root.querySelector('.try-live');
    this.fill = root.querySelector('.try-fill');
    this.label = root.querySelector('.try-label');
    this.active = true;
    this.isReady = false;
    this.pending = false; // clicked before the demo was ready
    this.onStart = null; // called once, when the live demo takes over
    this.onClick = null; // called on the first click (phones start loading only then)
    this.cues = [];
    fetch(cuesUrl).then((r) => r.json()).then((c) => (this.cues = c.cues)).catch(() => {});

    const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;
    const saveData = navigator.connection?.saveData;
    if (reduced || saveData) {
      this.video.removeAttribute('autoplay');
      this.video.preload = 'none';
      this.video.pause();
    } else {
      // Play only while on screen.
      new IntersectionObserver(([e]) => {
        if (!this.active) return;
        if (e.isIntersecting) this.video.play().catch(() => {});
        else this.video.pause();
      }).observe(root);
    }
    const tick = () => {
      if (!this.active) return;
      const t = this.video.currentTime;
      const cue = this.cues.find((c) => t >= c.start && t < c.end);
      const text = cue ? cue.caption : '';
      if (this.caption.textContent !== text) this.caption.textContent = text;
      requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);

    this.button.addEventListener('click', () => {
      this.onClick?.();
      this.onClick = null;
      if (this.isReady) this.start();
      else {
        this.pending = true;
        this.button.classList.add('loading');
        this.label.textContent = 'Loading 0 %';
      }
    });
  }

  // Loading progress, 0..1 (all downloads and compiles).
  progress(f) {
    this.fill.style.width = `${Math.round(100 * f)}%`;
    if (this.pending && !this.isReady) this.label.textContent = `Loading ${Math.floor(100 * f)} %`;
  }

  ready() {
    this.isReady = true;
    this.fill.style.width = '100%';
    this.button.classList.add('ready');
    if (this.pending) this.start();
  }

  // The demo cannot start: the button says so, and the reason is shown on the video.
  fail(message) {
    this.button.classList.remove('loading');
    this.button.classList.add('failed');
    this.label.textContent = 'Could not start the demo';
    this.button.title = message;
    const note = this.root.querySelector('.hero-error') ?? this.root.appendChild(Object.assign(document.createElement('p'), { className: 'hero-error' }));
    note.setAttribute('role', 'alert');
    note.textContent = `Could not start the demo: ${message}`;
  }

  start() {
    if (!this.active) return;
    this.active = false;
    this.onStart?.();
    this.root.classList.add('leaving');
    this.root.closest('.stage')?.classList.remove('hero-on');
    setTimeout(() => {
      this.video.pause();
      this.root.hidden = true;
    }, FADE_MS);
  }
}
