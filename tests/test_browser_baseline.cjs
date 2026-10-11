const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const { spawnSync } = require('node:child_process');

const { loadBrowser, root } = require('./browser_baseline.cjs');
const standalone = require('./browser/inference.js');

const read = (name) => JSON.parse(
  fs.readFileSync(path.join(root, name), 'utf8')
);

const weights = read('frontend/react/public/frontend/model_weights.json');
const ttaCases = read('tests/browser/tta_validation_cases.json');

function close(actual, expected, tolerance) {
  assert.equal(actual.length, 10);
  assert.equal(expected.length, 10);

  actual.forEach((value, i) => {
    assert.ok(Number.isFinite(value), `Nonfinite probability ${i}`);
    assert.ok(
      Math.abs(value - expected[i]) < tolerance,
      `Probability ${i}: ${value} != ${expected[i]}`
    );
  });
}

test('browser single-pass retains all 25 Python reference vectors', () => {
  const browser = loadBrowser();
  const cases = read('tests/browser/validation_cases.json');

  assert.equal(cases.length, 25);

  for (const c of cases) {
    close(
      browser.runInference(new Float32Array(c.pixels)),
      c.expected_probs,
      1e-4
    );
  }
});

test('browser FIVE-view TTA retains its separately frozen baseline vectors', () => {
  const fixture = read('tests/fixtures/browser_five_view_tta.json');
  const sourceHash = crypto.createHash('sha256')
    .update(fs.readFileSync(path.join(root, 'tests/browser/tta_validation_cases.json')))
    .digest('hex');

  assert.equal(fixture.input_fixture_sha256, sourceHash);
  assert.equal(fixture.behavior, 'browser-five-view-no-flip');
  assert.equal(fixture.cases.length, 5);

  const browser = loadBrowser();

  for (const c of fixture.cases) {
    close(
      browser.runInferenceTTA(new Float32Array(ttaCases[c.source_case].pixels)),
      c.expected_probs,
      1e-6
    );
  }
});

test('browser FIVE views are original, left, right, up, down with circular edges and equal averaging', () => {
  const browser = loadBrowser();
  const pixels = Float32Array.from({ length: 784 }, (_, i) => i);
  const seen = [];

  browser.runInference = (view) => {
    seen.push(Array.from(view));
    return Array.from(
      { length: 10 },
      (_, i) => i === seen.length - 1 ? 1 : 0
    );
  };

  const result = browser.runInferenceTTA(pixels);
  assert.equal(seen.length, 5, 'No sixth/flipped view in the browser baseline');

  const offsets = [[0, 0], [0, 1], [0, -1], [1, 0], [-1, 0]];

  offsets.forEach(([dy, dx], view) => {
    const expected = Array.from(
      { length: 784 },
      (_, i) => pixels[
        ((Math.floor(i / 28) + dy + 28) % 28) * 28
        + ((i % 28 + dx + 28) % 28)
      ]
    );

    assert.deepEqual(seen[view], expected);
  });

  close(result, [0.2, 0.2, 0.2, 0.2, 0.2, 0, 0, 0, 0, 0], 1e-7);
});

test('standalone SIX-view TTA retains all existing Python fixtures separately', () => {
  assert.equal(ttaCases.length, 5);

  for (const c of ttaCases) {
    close(
      standalone.runInferenceTTA(new Float32Array(c.pixels), weights),
      c.expected_avg_probs,
      1e-4
    );
  }
});

test('standalone self-test exits nonzero for single-pass, six-view, empty and nonfinite failures', () => {
  // All faults are injected in the child process. No fixture files are changed.
  for (const fault of ['single', 'tta', 'empty', 'nonfinite']) {
    const script = `
      const fs = require('node:fs');
      const original = fs.readFileSync;
      fs.readFileSync = function(file, ...args) {
        const data = original.call(this, file, ...args);
        const name = String(file).replaceAll('\\\\', '/');
        const target = ${JSON.stringify(fault)} === 'tta' ? '/tta_validation_cases.json' : '/validation_cases.json';
        if (!name.endsWith(target)) return data;
        const cases = JSON.parse(data);
        if (${JSON.stringify(fault)} === 'empty') return '[]';
        const key = ${JSON.stringify(fault)} === 'tta' ? 'expected_avg_probs' : 'expected_probs';
        cases[0][key][0] = ${JSON.stringify(fault)} === 'nonfinite' ? 'NaN' : 42;
        return JSON.stringify(cases);
      };
      process.argv = [process.execPath, require('node:path').resolve('tests/browser/inference.js')];
      require('node:module').runMain();
    `;

    const result = spawnSync(process.execPath, ['-e', script], {
      cwd: root,
      encoding: 'utf8',
      timeout: 60000,
    });

    assert.ifError(result.error);
    assert.equal(result.status, 1, `${fault}: ${result.stdout}\n${result.stderr}`);
    assert.match(
      result.stdout,
      /Overall TTA max probability difference/,
      'Must reach regression verdict rather than fail to launch'
    );
  }
});