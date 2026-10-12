/* Illustrative projection of a two-component world-centric Gaussian token.
 * The query-view panel displays projected probability density, not attention.
 */
(function () {
  'use strict';

  const NX = 21;
  const NY = 15;
  const FOCAL = 1.25;
  const PATCH_U = 1.8 / NX;
  const PATCH_V = 1.3 / NY;
  const VIEW_U = 0.9;
  const VIEW_V = 0.65;
  const DEFAULTS = { baselineX: 0.7, baselineY: 0.35, baselineZ: 0.1, yaw: -10, pitch: -8, depth: 2.2, split: 1.4, nearSigma: 0.14, farSigma: 1.0, spatialExtent: true };

  function keyCoordinate(i, j) {
    return { u: -0.9 + 1.8 * i / (NX - 1), v: -0.65 + 1.3 * j / (NY - 1) };
  }

  function rotateCamera(x, y, z, yawDegrees, pitchDegrees) {
    const yaw = yawDegrees * Math.PI / 180;
    const pitch = pitchDegrees * Math.PI / 180;
    const cy = Math.cos(yaw), sy = Math.sin(yaw);
    const cp = Math.cos(pitch), sp = Math.sin(pitch);
    const yawX = cy * x + sy * z;
    const yawZ = -sy * x + cy * z;
    return [yawX, cp * y + sp * yawZ, -sp * y + cp * yawZ];
  }

  function projectedHypothesis(u, v, depth, depthSigma, params) {
    const ray = rotateCamera(u / FOCAL, v / FOCAL, 1, params.yaw, params.pitch);
    const world = [params.baselineX + depth * ray[0], params.baselineY + depth * ray[1], params.baselineZ + depth * ray[2]];
    const [x, y, z] = world;
    const meanX = FOCAL * x / z;
    const meanY = FOCAL * y / z;

    // The local 3D covariance is the sum of patch-footprint and depth terms.
    // Projecting each generating direction by the pinhole Jacobian yields
    // the full 2D covariance of the distribution in the query view.
    function projectedDirection(direction) {
      return [
        FOCAL * (direction[0] * z - x * direction[2]) / (z * z),
        FOCAL * (direction[1] * z - y * direction[2]) / (z * z),
      ];
    }
    const du = projectedDirection(rotateCamera(depth / FOCAL, 0, 0, params.yaw, params.pitch));
    const dv = projectedDirection(rotateCamera(0, depth / FOCAL, 0, params.yaw, params.pitch));
    const dd = projectedDirection(ray);
    // Deliberately amplified for this interactive illustration: the world
    // footprint grows with depth, making its projected spread easy to see.
    const footprintScale = params.spatialExtent !== false ? 1.8 * depth / 1.5 : 0;
    const pixelVarianceU = PATCH_U * PATCH_U * footprintScale ** 2 / 12;
    const pixelVarianceV = PATCH_V * PATCH_V * footprintScale ** 2 / 12;
    const depthVariance = depthSigma * depthSigma;
    // Half a display pixel prevents singular density when depth alone
    // projects to a line or point after spatial extent is disabled.
    const varianceX = pixelVarianceU * du[0] ** 2 + pixelVarianceV * dv[0] ** 2 + depthVariance * dd[0] ** 2 + (VIEW_U / 420) ** 2;
    const varianceY = pixelVarianceU * du[1] ** 2 + pixelVarianceV * dv[1] ** 2 + depthVariance * dd[1] ** 2 + (VIEW_V / 300) ** 2;
    const covarianceXY = pixelVarianceU * du[0] * du[1] + pixelVarianceV * dv[0] * dv[1] + depthVariance * dd[0] * dd[1];
    return { world, meanX, meanY, varianceX, varianceY, covarianceXY, depth, depthSigma };
  }

  function hypotheses(u, v, params) {
    const near = params.depth - params.split / 2;
    const far = params.depth + params.split / 2;
    return [
      projectedHypothesis(u, v, near, params.nearSigma, params),
      projectedHypothesis(u, v, far, params.farSigma, params),
    ];
  }

  function gaussianDensity(x, y, component) {
    const dx = x - component.meanX;
    const dy = y - component.meanY;
    const determinant = component.varianceX * component.varianceY - component.covarianceXY ** 2;
    const distance = (component.varianceY * dx * dx - 2 * component.covarianceXY * dx * dy + component.varianceX * dy * dy) / determinant;
    return Math.exp(-0.5 * distance) / (2 * Math.PI * Math.sqrt(determinant));
  }

  function projectedDensity(x, y, components) {
    return 0.5 * (gaussianDensity(x, y, components[0]) + gaussianDensity(x, y, components[1]));
  }

  if (typeof module === 'object' && module.exports) {
    module.exports = { keyCoordinate, projectedHypothesis, hypotheses, gaussianDensity, projectedDensity, DEFAULTS };
  }
  if (typeof document === 'undefined') return;

  const root = document.getElementById('worldrope-demo');
  if (!root) return;
  const controls = {
    yaw: document.getElementById('demo-yaw'),
    pitch: document.getElementById('demo-pitch'),
    depth: document.getElementById('demo-depth'),
    split: document.getElementById('demo-split'),
    nearSigma: document.getElementById('demo-nearSigma'),
    farSigma: document.getElementById('demo-farSigma'),
  };
  const canvas = document.getElementById('demo-heatmap');
  const context = canvas.getContext('2d');
  const worldView = document.getElementById('demo-world');
  const status = document.getElementById('demo-status');
  const extentControl = document.getElementById('demo-spatial-extent');
  const cameraYControl = document.getElementById('demo-camera-y');
  const centerPatch = keyCoordinate(Math.floor(NX / 2), Math.floor(NY / 2));
  const camera = { x: DEFAULTS.baselineX, y: DEFAULTS.baselineY, z: DEFAULTS.baselineZ };
  let drag = null;

  function params() {
    return { ...Object.fromEntries(Object.entries(controls).map(([key, input]) => [key, Number(input.value)])), baselineX: camera.x, baselineY: camera.y, baselineZ: camera.z, spatialExtent: extentControl.checked };
  }

  function formatValues(p) {
    for (const key of Object.keys(controls)) {
      const output = document.getElementById(`demo-${key}-value`);
      output.textContent = key === 'yaw' || key === 'pitch' ? `${p[key]}°` : p[key].toFixed(2);
    }
    document.getElementById('demo-spatial-extent-value').textContent = p.spatialExtent ? 'On' : 'Off';
    document.getElementById('demo-camera-y-value').textContent = `${p.baselineY >= 0 ? '+' : ''}${p.baselineY.toFixed(2)}`;
    document.getElementById('demo-camera-position').textContent = `X ${p.baselineX.toFixed(2)} · Y ${p.baselineY >= 0 ? '+' : ''}${p.baselineY.toFixed(2)} · Z ${p.baselineZ.toFixed(2)}`;
  }

  function mix(a, b, t) { return Math.round(a + (b - a) * t); }
  function queryPixelX(u) { return (u + VIEW_U) * canvas.width / (2 * VIEW_U); }
  function queryPixelY(v) { return (VIEW_V - v) * canvas.height / (2 * VIEW_V); }

  function drawDistribution(components, p) {
    const width = canvas.width;
    const height = canvas.height;
    const values = new Float32Array(width * height);
    let peak = 0;
    for (let py = 0; py < height; py++) {
      const v = VIEW_V - (py + 0.5) * 2 * VIEW_V / height;
      for (let px = 0; px < width; px++) {
        const u = -VIEW_U + (px + 0.5) * 2 * VIEW_U / width;
        const density = projectedDensity(u, v, components);
        values[py * width + px] = density;
        if (density > peak) peak = density;
      }
    }
    const image = context.createImageData(width, height);
    const low = [250, 249, 252];
    const high = [91, 76, 120];
    for (let pixel = 0; pixel < values.length; pixel++) {
      const relative = peak > 0 ? values[pixel] / peak : 0;
      const offset = pixel * 4;
      image.data[offset] = mix(low[0], high[0], relative);
      image.data[offset + 1] = mix(low[1], high[1], relative);
      image.data[offset + 2] = mix(low[2], high[2], relative);
      image.data[offset + 3] = 255;
    }
    context.putImageData(image, 0, 0);
    const ray = rotateCamera(centerPatch.u / FOCAL, centerPatch.v / FOCAL, 1, p.yaw, p.pitch);
    const alongRay = depth => {
      const z = p.baselineZ + depth * ray[2];
      return [queryPixelX(FOCAL * (p.baselineX + depth * ray[0]) / z), queryPixelY(FOCAL * (p.baselineY + depth * ray[1]) / z)];
    };
    const a = alongRay(0.4);
    const b = alongRay(10);
    const dx = b[0] - a[0], dy = b[1] - a[1];
    if (Math.hypot(dx, dy) > 1) {
      context.save();
      context.beginPath();
      context.rect(0, 0, width, height);
      context.clip();
      context.strokeStyle = '#dc8658';
      context.lineWidth = 1.7;
      context.setLineDash([7, 5]);
      context.beginPath();
      context.moveTo(a[0] - 100 * dx, a[1] - 100 * dy);
      context.lineTo(a[0] + 100 * dx, a[1] + 100 * dy);
      context.stroke();
      const intersections = [];
      if (Math.abs(dx) > 1e-6) for (const x of [0, width]) {
        const y = a[1] + (x - a[0]) * dy / dx;
        if (y >= 0 && y <= height) intersections.push([x, y]);
      }
      if (Math.abs(dy) > 1e-6) for (const y of [0, height]) {
        const x = a[0] + (y - a[1]) * dx / dy;
        if (x >= 0 && x <= width) intersections.push([x, y]);
      }
      if (intersections.length >= 2) {
        intersections.sort((first, second) => Math.abs(dx) > Math.abs(dy) ? first[0] - second[0] : first[1] - second[1]);
        const start = intersections[0], end = intersections[intersections.length - 1];
        const labelX = start[0] + 0.82 * (end[0] - start[0]);
        const labelY = start[1] + 0.82 * (end[1] - start[1]);
        let angle = Math.atan2(end[1] - start[1], end[0] - start[0]);
        if (angle > Math.PI / 2) angle -= Math.PI;
        if (angle < -Math.PI / 2) angle += Math.PI;
        context.translate(labelX, labelY);
        context.rotate(angle);
        context.setLineDash([]);
        context.font = 'bold 11px Arial, sans-serif';
        context.textAlign = 'right';
        context.lineWidth = 3;
        context.strokeStyle = 'rgba(255,255,255,.95)';
        context.strokeText('Epipolar line', -4, -7);
        context.fillStyle = '#b56338';
        context.fillText('Epipolar line', -4, -7);
      }
      context.restore();
    }
    context.strokeStyle = 'rgba(85, 112, 119, .22)';
    context.lineWidth = 1;
    context.beginPath();
    context.moveTo(queryPixelX(0) + 0.5, 0);
    context.lineTo(queryPixelX(0) + 0.5, height);
    context.moveTo(0, queryPixelY(0) + 0.5);
    context.lineTo(width, queryPixelY(0) + 0.5);
    context.stroke();
    context.font = '11px Arial, sans-serif';
    context.fillStyle = '#6a777b';
    context.fillText('QUERY IMAGE  u / v', 13, 20);
    components.forEach((item, index) => {
      const px = queryPixelX(item.meanX);
      const py = queryPixelY(item.meanY);
      if (px < 0 || px >= width || py < 0 || py >= height) return;
      const color = index ? '#80609b' : '#477b9d';
      const scaleX = width / (2 * VIEW_U);
      const scaleY = height / (2 * VIEW_V);
      const xx = item.varianceX * scaleX * scaleX;
      const yy = item.varianceY * scaleY * scaleY;
      const xy = -item.covarianceXY * scaleX * scaleY;
      const halfDifference = (xx - yy) / 2;
      const root = Math.hypot(halfDifference, xy);
      const major = Math.sqrt(Math.max(0, (xx + yy) / 2 + root));
      const minor = Math.sqrt(Math.max(0, (xx + yy) / 2 - root));
      const angle = Math.atan2(2 * xy, xx - yy) / 2;
      context.strokeStyle = color;
      context.lineWidth = 1;
      for (const radius of [1, 2]) {
        context.globalAlpha = radius === 1 ? 0.75 : 0.4;
        context.beginPath();
        context.ellipse(px, py, major * radius, minor * radius, angle, 0, 2 * Math.PI);
        context.stroke();
      }
      context.globalAlpha = 1;
      context.beginPath();
      context.arc(px, py, 5, 0, 2 * Math.PI);
      context.fillStyle = '#fff';
      context.fill();
      context.strokeStyle = color;
      context.lineWidth = 2;
      context.stroke();
      context.font = 'bold 12px Arial, sans-serif';
      context.fillStyle = color;
      context.fillText(index ? 'Far' : 'Near', Math.min(px + 9, width - 31), Math.max(py - 8, 33));
    });
    return peak;
  }

  // Keep both positive and negative camera heights inside the 3D diagram.
  function worldPixel(x, y, z) { return [210 + 55 * x + 15 * z, 216 - 42 * y - 30 * z]; }

  function drawWorld(p, components) {
    const point = xyz => worldPixel(...xyz);
    const line = (a, b, stroke, width = 1, dash = '') => {
      const [x1, y1] = point(a), [x2, y2] = point(b);
      return `<path d="M${x1} ${y1} L${x2} ${y2}" fill="none" stroke="${stroke}" stroke-width="${width}" ${dash ? `stroke-dasharray="${dash}"` : ''}/>`;
    };
    const grid = [-2, -1, 0, 1, 2].map(x => line([x, 0, 0], [x, 0, 5], '#edf0f1')).join('')
      + [1, 2, 3, 4, 5].map(z => line([-2.5, 0, z], [2.5, 0, z], '#edf0f1')).join('');
    const cameraPosition = [p.baselineX, p.baselineY, p.baselineZ];
    const cameraGround = [p.baselineX, 0, p.baselineZ];
    const [kx, ky] = point(cameraPosition);
    const [gx, gy] = point(cameraGround);
    const [hx, hy] = point([p.baselineX, p.baselineY + 0.75, p.baselineZ]);
    const planeHandleX = gx + 39, planeHandleY = gy + 22;
    const ray = rotateCamera(centerPatch.u / FOCAL, centerPatch.v / FOCAL, 1, p.yaw, p.pitch);
    const end = [p.baselineX + 5.2 * ray[0], p.baselineY + 5.2 * ray[1], p.baselineZ + 5.2 * ray[2]];
    const patch = components.map((item, index) => {
      const [cx, cy] = point(item.world);
      const scale = p.spatialExtent ? 1.8 * item.depth / 1.5 : 0;
    const rayLength = Math.hypot(55 * ray[0] + 15 * ray[2], -42 * ray[1] - 30 * ray[2]);
      const rx = 8 + item.depthSigma * rayLength + scale * 2;
      const ry = 5 + scale * 3;
      const angle = Math.atan2(-42 * ray[1] - 30 * ray[2], 55 * ray[0] + 15 * ray[2]) * 180 / Math.PI;
      const stroke = index ? '#9a7caa' : '#477b9d';
      const fill = index ? '#eee8f1' : '#e8f0f5';
      return `<ellipse cx="${cx}" cy="${cy}" rx="${rx}" ry="${ry}" transform="rotate(${angle} ${cx} ${cy})" fill="${fill}" fill-opacity=".75" stroke="${stroke}" stroke-width="2"/><circle cx="${cx}" cy="${cy}" r="3" fill="${stroke}"/><text x="${cx + 10}" y="${cy - 10}" fill="${stroke}" font-size="12" font-weight="700">${index ? 'Far' : 'Near'}</text>`;
    }).join('');
    const [qx, qy] = point([0, 0, 0]);
    worldView.innerHTML = `<rect width="480" height="300" fill="#fff"/><text x="16" y="23" fill="#5b4c78" font-size="13" font-weight="700" letter-spacing=".5">WORLD-CENTRIC 3D GAUSSIANS</text>${grid}
      ${line([0, 0, 0], [2.8, 0, 0], '#c5d0d3', 1.5)}${line([0, 0, 0], [0, 1.5, 0], '#c5d0d3', 1.5)}${line([0, 0, 0], [0, 0, 5.2], '#c5d0d3', 1.5)}
      <text x="370" y="224" fill="#869397" font-size="11">X</text><text x="220" y="146" fill="#869397" font-size="11">Y</text><text x="292" y="52" fill="#869397" font-size="11">Z</text>
      ${line(cameraGround, cameraPosition, '#a0b9c3', 1.5, '4 4')}${line(cameraPosition, end, '#6999ae', 2, '5 5')}
      ${patch}
      <circle cx="${qx}" cy="${qy}" r="8" fill="#e09462" stroke="#fff" stroke-width="2"/><text x="${qx + 12}" y="${qy + 5}" fill="#384750" font-size="13" font-weight="600">Q</text>
      <circle cx="${gx}" cy="${gy}" r="3" fill="#8fa9b2"/>
      <g data-drag="plane" style="cursor:move"><path d="M${gx} ${gy} L${planeHandleX} ${planeHandleY}" fill="none" stroke="#9ab2bc" stroke-width="1.5"/><circle cx="${planeHandleX}" cy="${planeHandleY}" r="14" fill="#fff" stroke="#427c9b" stroke-width="1.6"/><text x="${planeHandleX}" y="${planeHandleY + 4}" text-anchor="middle" fill="#427c9b" font-size="10" font-weight="700">XZ</text></g>
      <g data-drag="camera" style="cursor:move"><circle cx="${kx}" cy="${ky}" r="21" fill="transparent"/><circle cx="${kx}" cy="${ky}" r="10" fill="#427c9b" stroke="#fff" stroke-width="2"/><text x="${kx + 13}" y="${ky + 5}" fill="#384750" font-size="13" font-weight="600">K</text></g>
      <g data-drag="height" style="cursor:ns-resize"><path d="M${kx} ${ky - 12} L${hx} ${hy}" fill="none" stroke="#427c9b" stroke-width="2"/><circle cx="${hx}" cy="${hy}" r="9" fill="#fff" stroke="#427c9b" stroke-width="2"/><text x="${hx + 13}" y="${hy + 4}" fill="#427c9b" font-size="12" font-weight="700">Y</text></g>`;
  }

  function update() {
    const p = params();
    formatValues(p);
    const projected = hypotheses(centerPatch.u, centerPatch.v, p);
    const peak = drawDistribution(projected, p);
    drawWorld(p, projected);
    const positions = projected.map(item => `(${item.meanX.toFixed(2)}, ${item.meanY.toFixed(2)})`).join(' and ');
    status.textContent = `Center key patch · depth σ near ${p.nearSigma.toFixed(2)}, far ${p.farSigma.toFixed(2)} · projected query positions ${positions}${peak === 0 ? ' · outside the query image' : ''}`;
  }

  Object.values(controls).forEach(input => input.addEventListener('input', update));
  cameraYControl.addEventListener('input', () => {
    camera.y = Number(cameraYControl.value);
    update();
  });
  extentControl.addEventListener('change', update);
  const clamp = (value, minimum, maximum) => Math.max(minimum, Math.min(maximum, value));
  function pointerPosition(event) {
    const rect = worldView.getBoundingClientRect();
    return [(event.clientX - rect.left) * 480 / rect.width, (event.clientY - rect.top) * 300 / rect.height];
  }
  worldView.addEventListener('pointerdown', event => {
    const handle = event.target.closest('[data-drag]');
    if (!handle) return;
    event.preventDefault();
    const [x, y] = pointerPosition(event);
    drag = { id: event.pointerId, kind: event.shiftKey ? 'height' : handle.dataset.drag, x, y, cameraX: camera.x, cameraY: camera.y, cameraZ: camera.z };
    worldView.setPointerCapture(event.pointerId);
  });
  worldView.addEventListener('pointermove', event => {
    if (!drag || drag.id !== event.pointerId) return;
    const [x, y] = pointerPosition(event);
    if (drag.kind === 'height' || drag.kind === 'camera') {
      camera.y = clamp(drag.cameraY - (y - drag.y) / 42, -1.5, 1.5);
      cameraYControl.value = camera.y.toFixed(2);
      if (drag.kind === 'camera') camera.x = clamp(drag.cameraX + (x - drag.x) / 55, -1.5, 1.5);
    } else {
      camera.z = clamp(drag.cameraZ - (y - drag.y) / 32, -0.25, 1.5);
      camera.x = clamp(drag.cameraX + ((x - drag.x) - 15 * (camera.z - drag.cameraZ)) / 55, -1.5, 1.5);
    }
    update();
  });
  function endDrag(event) {
    if (!drag || drag.id !== event.pointerId) return;
    drag = null;
    if (worldView.hasPointerCapture(event.pointerId)) worldView.releasePointerCapture(event.pointerId);
  }
  worldView.addEventListener('pointerup', endDrag);
  worldView.addEventListener('pointercancel', endDrag);
  document.getElementById('demo-reset').addEventListener('click', () => {
    for (const [key, input] of Object.entries(controls)) input.value = DEFAULTS[key];
    camera.x = DEFAULTS.baselineX;
    camera.y = DEFAULTS.baselineY;
    cameraYControl.value = DEFAULTS.baselineY;
    camera.z = DEFAULTS.baselineZ;
    extentControl.checked = DEFAULTS.spatialExtent;
    update();
  });
  update();
})();
