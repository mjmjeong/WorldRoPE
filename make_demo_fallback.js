// Render the default query-view probability density for no-JavaScript fallback.
const fs = require('node:fs');
const path = require('node:path');
const demo = require('./worldrope-demo.js');

const width = 420;
const height = 300;
const step = 4;
const center = demo.keyCoordinate(10, 7);
const components = demo.hypotheses(center.u, center.v, demo.DEFAULTS);
const cells = [];
let peak = 0;
for (let y = 0; y < height; y += step) {
  for (let x = 0; x < width; x += step) {
    const u = -0.9 + (x + step / 2) * 1.8 / width;
    const v = 0.65 - (y + step / 2) * 1.3 / height;
    const density = demo.projectedDensity(u, v, components);
    cells.push({ x, y, density });
    peak = Math.max(peak, density);
  }
}
const mix = (a, b, t) => Math.round(a + (b - a) * t);
const rects = cells.map(({ x, y, density }) => {
  const t = peak ? density / peak : 0;
  const rgb = [mix(250, 91, t), mix(249, 76, t), mix(252, 120, t)];
  return `<rect x="${x}" y="${y}" width="${step}" height="${step}" fill="rgb(${rgb.join(',')})"/>`;
});
const centerPixels = components.map(item => [(item.meanX + 0.9) * width / 1.8, (0.65 - item.meanY) * height / 1.3]);
const [a, b] = centerPixels;
const line = `<path d="M${a[0] - 100 * (b[0] - a[0])} ${a[1] - 100 * (b[1] - a[1])} L${a[0] + 100 * (b[0] - a[0])} ${a[1] + 100 * (b[1] - a[1])}" fill="none" stroke="#dc8658" stroke-width="1.7" stroke-dasharray="7 5"/>`;
const labelX = 350;
const labelY = a[1] + (labelX - a[0]) * (b[1] - a[1]) / (b[0] - a[0]);
const labelAngle = Math.atan2(b[1] - a[1], b[0] - a[0]) * 180 / Math.PI;
const lineLabel = `<text x="${labelX}" y="${labelY - 7}" transform="rotate(${labelAngle} ${labelX} ${labelY})" fill="#b56338" stroke="#fff" stroke-width="3" paint-order="stroke" text-anchor="end" font-family="Arial,sans-serif" font-size="11" font-weight="bold">Epipolar line</text>`;
const markers = components.map((item, index) => {
  const x = (item.meanX + 0.9) * width / 1.8;
  const y = (0.65 - item.meanY) * height / 1.3;
  if (x < 0 || x >= width || y < 0 || y >= height) return '';
  const color = index ? '#80609b' : '#477b9d';
  return `<circle cx="${x}" cy="${y}" r="5" fill="#fff" stroke="${color}" stroke-width="2"/><text x="${Math.min(x + 9, width - 31)}" y="${Math.max(y - 8, 33)}" fill="${color}" font-family="Arial,sans-serif" font-size="12" font-weight="bold">${index ? 'Far' : 'Near'}</text>`;
}).join('');
const svg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 420 300" role="img" aria-label="Default projected probability density and epipolar line in the query view">${rects.join('')}${line}${lineLabel}<path d="M210 0V300M0 150H420" fill="none" stroke="#b5c8ca" stroke-width="1"/><text x="13" y="20" fill="#6a777b" font-family="Arial,sans-serif" font-size="11">QUERY IMAGE  u / v</text>${markers}</svg>\n`;
fs.writeFileSync(path.join(__dirname, 'assets/projected-distribution-default.svg'), svg);
