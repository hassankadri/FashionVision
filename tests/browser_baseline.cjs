// Load the production browser CNN without rendering the React UI.
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const assert = require('node:assert/strict');

const root = path.resolve(__dirname, '..');

function loadBrowser() {
  const assetRoot = path.join(root, 'frontend/react/public/frontend');
  const source = fs.readFileSync(path.join(assetRoot, 'cnn-browser.js'), 'utf8');
  const rawWeights = JSON.parse(fs.readFileSync(path.join(assetRoot, 'model_weights.json'), 'utf8'));
  const exportMarker = 'window.FashionCNN = Object.freeze({ load, predict });';
  const inferenceMarker = 'const one = forward(view, weights);';
  assert.ok(source.includes(exportMarker), 'Review harness if CNN exports change');
  assert.ok(source.includes(inferenceMarker), 'Review harness if CNN prediction loop changes');

  const context = vm.createContext({
    atob: (value) => Buffer.from(value, 'base64').toString('binary'),
    window: {},
    rawWeights,
  });
  // Expose existing private operations for regression checks; preserve test assertions.
  const instrumented = source
    .replace(inferenceMarker, 'const one = window.runInference(view);')
    .replace(exportMarker, [
      'const weights = load(rawWeights);',
      'window.runInference = (pixels) => forward(pixels, weights);',
      'window.runInferenceTTA = (pixels) => predict(pixels, weights);',
      exportMarker,
    ].join('\n'));
  vm.runInContext(instrumented, context);
  return context.window;
}

module.exports = { loadBrowser, root };
