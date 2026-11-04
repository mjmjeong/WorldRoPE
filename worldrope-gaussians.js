(() => {
  const root = document.getElementById('gaussian-explorer');
  if (!root) return;

  const viewer = document.getElementById('gaussian-viewer');
  const status = document.getElementById('gaussian-viewer-status');
  const title = document.getElementById('gaussian-title');
  const description = document.getElementById('gaussian-description');
  const tabs = [...root.querySelectorAll('[data-gaussian-view]')];
  const views = {
    query: ['Query View', 'The query camera receiving projected Gaussian support from the key view.'],
    key: ['Key View', 'The key camera whose patch tokens are lifted into world-centric Gaussian supports.'],
    layer1: ['Layer 1', 'Early attention support before deeper geometric refinement.'],
    layer2: ['Layer 2', 'Intermediate Gaussian support after the second attention layer.'],
    layer3: ['Layer 3', 'Refined world-space support after the third attention layer.'],
    layer4: ['Layer 4', 'Final layer support before projection into the query view.']
  };

  tabs.forEach(tab => tab.addEventListener('click', () => {
    const key = tab.dataset.gaussianView;
    const [label, copy] = views[key];
    tabs.forEach(item => item.setAttribute('aria-selected', String(item === tab)));
    viewer.dataset.activeView = key;
    title.textContent = label;
    description.textContent = copy;
    status.textContent = `3DGS renderer will load here · ${label}`;
  }));
})();
