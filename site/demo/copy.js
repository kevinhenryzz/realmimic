// A "Copy" button on every code block (the Get started commands, the citation): selecting long text on a phone is
// fiddly. It copies the block exactly as shown.

for (const pre of document.querySelectorAll('pre')) {
  const code = pre.querySelector('code') ?? pre;
  const btn = Object.assign(document.createElement('button'), { type: 'button', className: 'copy', textContent: 'Copy' });
  btn.addEventListener('click', async () => {
    try {
      await navigator.clipboard.writeText(code.textContent);
      btn.textContent = 'Copied';
    } catch {
      btn.textContent = 'Select and copy';
    }
    setTimeout(() => (btn.textContent = 'Copy'), 1600);
  });
  pre.append(btn);
}
