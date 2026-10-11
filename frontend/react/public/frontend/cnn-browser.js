
/*
 * Browser Fashion-MNIST CNN.
 * Conv(ReLU) -> BatchNorm -> Pool.
 * Uses five-view shift-only test-time augmentation.
 * Does NOT replace the existing Node inference.js.
 */
(() => {
  'use strict';

  function decode(b64) {
    const raw = atob(b64);
    const bytes = new Uint8Array(raw.length);
    for (let i = 0; i < raw.length; i++) {
      bytes[i] = raw.charCodeAt(i);
    }
    return new Float32Array(bytes.buffer);
  }

  function layer(raw) {
    const output = {};
    for (const [key, value] of Object.entries(raw)) {
      output[key] = [
        'kernel', 'bias', 'gamma', 'beta', 'mean', 'var'
      ].includes(key) ? decode(value) : value;
    }
    return output;
  }

  function load(raw) {
    return {
      conv1: layer(raw.conv1),
      bn1: layer(raw.bn1),
      conv2: layer(raw.conv2),
      bn2: layer(raw.bn2),
      dense1: layer(raw.dense1),
      dense2: layer(raw.dense2),
      class_names: raw.class_names
    };
  }

  function conv(input, H, W, Cin, kernel, shape, bias, Cout) {
    const [kh, kw] = shape;
    const ph = Math.floor((kh - 1) / 2);
    const pw = Math.floor((kw - 1) / 2);
    const output = new Float32Array(H * W * Cout);

    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        for (let co = 0; co < Cout; co++) {
          let sum = bias[co];

          for (let ky = 0; ky < kh; ky++) {
            const iy = y + ky - ph;
            if (iy < 0 || iy >= H) continue;

            for (let kx = 0; kx < kw; kx++) {
              const ix = x + kx - pw;
              if (ix < 0 || ix >= W) continue;

              for (let ci = 0; ci < Cin; ci++) {
                sum += input[(iy * W + ix) * Cin + ci] *
                  kernel[((ky * kw + kx) * Cin + ci) * Cout + co];
              }
            }
          }
          output[(y * W + x) * Cout + co] = sum > 0 ? sum : 0;
        }
      }
    }
    return output;
  }

  function bn(input, H, W, C, gamma, beta, mean, variance, epsilon) {
    const output = new Float32Array(H * W * C);
    const scale = new Float32Array(C);
    const shift = new Float32Array(C);

    for (let c = 0; c < C; c++) {
      scale[c] = gamma[c] / Math.sqrt(variance[c] + epsilon);
      shift[c] = beta[c] - mean[c] * scale[c];
    }

    for (let i = 0; i < H * W; i++) {
      for (let c = 0; c < C; c++) {
        const idx = i * C + c;
        output[idx] = input[idx] * scale[c] + shift[c];
      }
    }
    return output;
  }

  function pool(input, H, W, C) {
    const hh = H / 2;
    const ww = W / 2;
    const output = new Float32Array(hh * ww * C);

    for (let y = 0; y < hh; y++) {
      for (let x = 0; x < ww; x++) {
        for (let c = 0; c < C; c++) {
          const a = input[((2 * y) * W + 2 * x) * C + c];
          const b = input[((2 * y) * W + 2 * x + 1) * C + c];
          const d = input[((2 * y + 1) * W + 2 * x) * C + c];
          const e = input[((2 * y + 1) * W + 2 * x + 1) * C + c];
          output[(y * ww + x) * C + c] = Math.max(a, b, d, e);
        }
      }
    }
    return output;
  }

  function dense(input, kernel, shape, bias, relu) {
    const [inDim, outDim] = shape;
    const output = new Float32Array(outDim);

    for (let o = 0; o < outDim; o++) {
      let sum = bias[o];
      for (let i = 0; i < inDim; i++) {
        sum += input[i] * kernel[i * outDim + o];
      }
      output[o] = relu ? Math.max(0, sum) : sum;
    }
    return output;
  }

  function softmax(values) {
    const m = Math.max(...values);
    const exp = Array.from(values, v => Math.exp(v - m));
    const total = exp.reduce((a, b) => a + b, 0);
    return exp.map(v => v / total);
  }

  function forward(pixels, w) {
    const a = conv(pixels, 28, 28, 1,
      w.conv1.kernel, w.conv1.kernel_shape.slice(0, 2),
      w.conv1.bias, 32);

    const b = bn(a, 28, 28, 32,
      w.bn1.gamma, w.bn1.beta, w.bn1.mean,
      w.bn1.var, w.bn1.epsilon);
    const c = pool(b, 28, 28, 32);

    const d = conv(c, 14, 14, 32,
      w.conv2.kernel, w.conv2.kernel_shape.slice(0, 2),
      w.conv2.bias, 64);

    const e = bn(d, 14, 14, 64,
      w.bn2.gamma, w.bn2.beta, w.bn2.mean,
      w.bn2.var, w.bn2.epsilon);
    const f = pool(e, 14, 14, 64);

    const g = dense(f, w.dense1.kernel,
      w.dense1.kernel_shape, w.dense1.bias, true);
    const h = dense(g, w.dense2.kernel,
      w.dense2.kernel_shape, w.dense2.bias, false);

    return softmax(h);
  }

  function roll(input, shift, axis) {
    const output = new Float32Array(784);
    for (let y = 0; y < 28; y++) {
      for (let x = 0; x < 28; x++) {
        const sy = axis === 0
          ? ((y - shift) % 28 + 28) % 28 : y;
        const sx = axis === 1
          ? ((x - shift) % 28 + 28) % 28 : x;
        output[y * 28 + x] = input[sy * 28 + sx];
      }
    }
    return output;
  }

  function predict(pixels, weights) {
    const views = [
      pixels,
      roll(pixels, -1, 1),
      roll(pixels, 1, 1),
      roll(pixels, -1, 0),
      roll(pixels, 1, 0)
    ];

    const probs = new Float32Array(10);
    for (const view of views) {
      const one = forward(view, weights);
      for (let i = 0; i < 10; i++) {
        probs[i] += one[i] / views.length;
      }
    }
    return Array.from(probs);
  }

  window.FashionCNN = Object.freeze({ load, predict });
})();
