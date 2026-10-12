import { test } from 'node:test';
import assert from 'node:assert/strict';
import { registerHooks } from 'node:module';

const sdk = 'data:text/javascript,' + encodeURIComponent([
  'export async function issueSignedToken(options) { return globalThis.blobFake.issue(options); }',
  'export async function presignUrl(...args) { return globalThis.blobFake.presign(...args); }',
  'export async function del(...args) { return globalThis.blobFake.del(...args); }',
  'export async function list(...args) { return globalThis.blobFake.list(...args); }',
].join('\n'));
registerHooks({
  resolve(specifier, context, next) {
    if (specifier === '@vercel/blob' || specifier === '@vercel/blob/client') return { url: sdk, shortCircuit: true };
    return next(specifier, context);
  },
  load(url, context, next) {
    const result = next(url, context);
    if (url.endsWith('/src/imageRequest.js')) {
      return { ...result, source: String(result.source).replace('import.meta.env.DEV', 'globalThis.localDevelopment') };
    }
    return result;
  },
});
const control = await import('../frontend/react/server/uploads.js');
const { default: authorize } = await import('../frontend/react/api/uploads/authorize.js');
const { default: sweep } = await import('../frontend/react/api/uploads/sweep.js');
const { postImage } = await import('../frontend/react/src/imageRequest.js');

function settings() {
  delete process.env.MOBILENET_API_ORIGIN;
  delete process.env.BLOB_READ_WRITE_TOKEN;
  delete process.env.BLOB_PRIVATE_HOST;
  Object.assign(process.env, { PRIVATE_UPLOADS_ENABLED: '1', UPLOAD_AUTH_SECRET: 'c'.repeat(64),
    UPSTASH_REDIS_REST_URL: 'https://example.upstash.io', UPSTASH_REDIS_REST_TOKEN: 'test',
    BLOB_STORE_ID: 'store_examplestore', VERCEL_OIDC_TOKEN: 'runtime-oidc-test-only',
    CRON_SECRET: 'd'.repeat(32), VERCEL: '1' });
}
function response() {
  return { code: 0, data: null, headers: {},
    setHeader(name, value) { this.headers[name] = value; },
    status(code) { this.code = code; return this; }, json(data) { this.data = data; return this; } };
}
function request(body) {
  return { method: 'POST', body, headers: { origin: 'https://fashion-vision.vercel.app',
    'sec-fetch-site': 'same-origin', 'content-type': 'application/json', 'x-vercel-forwarded-for': '192.0.2.1' } };
}

test('OIDC authorization issues one scoped private URL without exposing signing or runtime credentials', async () => {
  settings();
  let constraints, command, delegation;
  globalThis.fetch = async (_url, options) => { command = JSON.parse(options.body); return Response.json({ result: 1 }); };
  const material = { delegationToken: 'scoped-delegation', clientSigningToken: 'server-signing-secret' };
  globalThis.blobFake = {
    issue: async options => { delegation = options; return material; },
    presign: async (token, options) => {
      assert.equal(token, material); constraints = options;
      return { presignedUrl: 'https://vercel.com/api/blob/?scoped-signature' };
    },
  };
  const res = response();
  await authorize(request({ size: control.MAX_BYTES, type: 'image/png', purpose: 'clip' }), res);
  assert.equal(res.code, 200);
  assert.equal(res.data.presignedUrl, 'https://vercel.com/api/blob/?scoped-signature');
  for (const secret of ['server-signing-secret', 'runtime-oidc-test-only']) {
    assert.equal(JSON.stringify(res.data).includes(secret), false);
  }
  assert.equal(res.data.clientToken, undefined);
  assert.deepEqual(delegation.operations, ['put']);
  assert.equal(delegation.pathname, res.data.pathname);
  assert.equal(delegation.maximumSizeInBytes, control.MAX_BYTES);
  assert.deepEqual(delegation.allowedContentTypes, ['image/png']);
  assert.equal(delegation.validUntil, constraints.validUntil);
  assert.equal(delegation.token, undefined); assert.equal(delegation.oidcToken, undefined);
  assert.equal(constraints.access, 'private'); assert.equal(constraints.operation, 'put');
  assert.equal(constraints.maximumSizeInBytes, 15 * 1024 * 1024);
  assert.deepEqual(constraints.allowedContentTypes, ['image/png']);
  assert.equal(constraints.pathname, res.data.pathname);
  assert.equal(constraints.allowOverwrite, false);
  assert.equal(constraints.addRandomSuffix, false);
  assert(constraints.validUntil > Date.now() && constraints.validUntil <= Date.now() + 300000);
  assert.equal(command[0], 'EVAL');
  assert.deepEqual(command.slice(8, 12), control.LIMITS);
});

test('authorization rejects missing settings, foreign origin, oversized files and prototype MIME names', async () => {
  settings();
  for (const bad of [
    { size: control.MAX_BYTES + 1, type: 'image/png', purpose: 'clip' },
    { size: 10, type: '__proto__', purpose: 'clip' },
    { size: 10, type: 'image/png', purpose: 'clip', pathname: '../secret' },
  ]) {
    const res = response();
    await authorize(request(bad), res);
    assert.equal(res.code, 400);
  }
  const foreign = request({ size: 10, type: 'image/png', purpose: 'clip' });
  foreign.headers.origin = 'https://attacker.example';
  const res = response(); await authorize(foreign, res); assert.equal(res.code, 403);
  delete process.env.UPLOAD_AUTH_SECRET;
  const missing = response(); await authorize(request({ size: 10, type: 'image/png', purpose: 'clip' }), missing);
  assert.equal(missing.code, 503);
});

test('distributed quota exhaustion returns 429 and Redis outages fail closed', async () => {
  settings();
  globalThis.fetch = async () => Response.json({ result: 0 });
  const res = response(); await authorize(request({ size: 10, type: 'image/png', purpose: 'clip' }), res);
  assert.equal(res.code, 429);
  globalThis.fetch = async () => { throw new Error('unavailable'); };
  const unavailable = response(); await authorize(request({ size: 10, type: 'image/png', purpose: 'clip' }), unavailable);
  assert.equal(unavailable.code, 503);
});

test('receipts cannot select other paths, capabilities or arbitrary URLs', () => {
  const ticket = { uploadId: 'a'.repeat(32), receipt: 'b'.repeat(64) };
  const record = { mime: 'image/png', size: 100, purpose: 'clip', expires: Math.floor(Date.now()/1000) + 600,
    receiptHash: control.receiptHash(ticket.receipt), pathname: control.PREFIX + ticket.uploadId + '.png' };
  assert.equal(control.checkRecord(record, ticket), record);
  assert.throws(() => control.checkRecord({ ...record, pathname: 'private/other.png' }, ticket));
  assert.throws(() => control.checkRecord({ ...record, expires: 0 }, ticket));
  assert.throws(() => control.checkRecord(record, { ...ticket, receipt: 'f'.repeat(64) }));
  assert.throws(() => control.receiptBody({ ...ticket, url: 'https://attacker.example' }));
  assert.throws(() => control.backendFor('mobilenet'));
});

test('small and local uploads preserve the original raw request and result', async () => {
  const expected = { prediction: 'Shirt', confidence: 0.75, predictions: [] };
  globalThis.localDevelopment = false;
  for (const [size, local] of [[100, false], [15 * 1024 * 1024, true]]) {
    const file = new File([new Uint8Array(size)], 'original.png', { type: 'image/png' });
    globalThis.localDevelopment = local;
    const calls = [];
    globalThis.fetch = async (url, options) => { calls.push([url, options]); return Response.json(expected); };
    assert.deepEqual(await postImage('/api/predict', file), expected);
    assert.equal(calls.length, 1);
    assert.equal(calls[0][0], '/api/predict');
    assert.equal(calls[0][1].body, file);
    assert.deepEqual(calls[0][1].headers, { 'Content-Type': 'image/png' });
  }
});

test('large uploads PUT unchanged bytes to the scoped URL and clean up after inference failure', async () => {
  globalThis.localDevelopment = false;
  const file = new File([new Uint8Array(4_000_001)], 'original.png', { type: 'image/png' });
  const ticket = { uploadId: 'a'.repeat(32), receipt: 'b'.repeat(64) };
  let uploaded;
  const uploadUrl = 'https://vercel.com/api/blob/?scoped-signature';
  const paths = [];
  globalThis.fetch = async (url, options) => {
    paths.push(url);
    if (url.endsWith('authorize')) return Response.json({ ...ticket, pathname: control.PREFIX + ticket.uploadId + '.png', presignedUrl: uploadUrl });
    if (url === uploadUrl) { uploaded = options; return Response.json({}); }
    if (url.endsWith('infer')) {
      assert.deepEqual(JSON.parse(options.body), ticket);
      return Response.json({ detail: 'model unavailable' }, { status: 503 });
    }
    return Response.json({ status: 'deleted' });
  };
  await assert.rejects(postImage('/api/predict', file), /model unavailable/);
  assert.equal(uploaded.body, file); assert.equal(uploaded.method, 'PUT');
  assert.deepEqual(uploaded.headers, { 'Content-Type': 'image/png' });
  assert.equal(uploaded.credentials, 'omit'); assert.equal(uploaded.redirect, 'error');
  assert.equal(uploaded.referrerPolicy, 'no-referrer');
  assert.deepEqual(paths, ['/api/uploads/authorize', uploadUrl, '/api/uploads/infer', '/api/uploads/cleanup']);
});

test('cron is authenticated and deletes only expired files in the application namespace', async () => {
  settings();
  const deleted = [];
  globalThis.blobFake = {
    list: async options => { assert.equal(options.token, undefined); assert.equal(options.oidcToken, undefined); return ({ hasMore: false, blobs: [
      { pathname: control.PREFIX + 'a'.repeat(32) + '.png', uploadedAt: new Date(Date.now() - 7200000) },
      { pathname: control.PREFIX + 'b'.repeat(32) + '.png', uploadedAt: new Date() },
      { pathname: 'unrelated/private.png', uploadedAt: new Date(0) },
    ] }); },
    del: async (paths, options) => {
      assert.equal(options.token, undefined); assert.equal(options.oidcToken, undefined);
      deleted.push(...paths);
    },
  };
  const denied = response(); await sweep({ method: 'GET', headers: {} }, denied); assert.equal(denied.code, 403);
  process.env.PRIVATE_UPLOADS_ENABLED = '0'; // Abandoned-object cleanup stays available.
  const accepted = response();
  await sweep({ method: 'GET', headers: { authorization: 'Bearer ' + process.env.CRON_SECRET } }, accepted);
  assert.equal(accepted.code, 200);
  assert.deepEqual(deleted, [control.PREFIX + 'a'.repeat(32) + '.png']);
});

test('cleanup requires the bound receipt and backend proof while inference is active', async () => {
  settings();
  const { default: cleanup } = await import('../frontend/react/api/uploads/cleanup.js');
  const ticket = { uploadId: 'a'.repeat(32), receipt: 'b'.repeat(64) };
  const record = { mime: 'image/png', size: 100, purpose: 'clip', expires: 0,
    status: 'processing', receiptHash: control.receiptHash(ticket.receipt),
    pathname: control.PREFIX + ticket.uploadId + '.png' };
  let deleted = 0;
  globalThis.blobFake = { del: async (pathname, options) => {
    assert.equal(pathname, record.pathname); assert.equal(options.token, undefined);
    assert.equal(options.oidcToken, undefined); deleted++;
  } };
  globalThis.fetch = async (_url, options) => {
    const command = JSON.parse(options.body);
    if (command[0] === 'GET') return Response.json({ result: JSON.stringify(record) });
    if (command[1].includes("record.status='deleted'")) return Response.json({ result: 1 });
    return Response.json({ result: command.at(-2) === '1' ? record.pathname : 'processing' });
  };
  const browser = response(); await cleanup(request(ticket), browser);
  assert.equal(browser.code, 202); assert.equal(deleted, 0);
  const forged = request(ticket); forged.headers['x-fv-cleanup-proof'] = '0'.repeat(64);
  const refused = response(); await cleanup(forged, refused);
  assert.equal(refused.code, 202); assert.equal(deleted, 0);
  const backend = request(ticket); backend.headers['x-fv-cleanup-proof'] = control.cleanupProof(ticket);
  const completed = response(); await cleanup(backend, completed);
  assert.equal(completed.code, 200); assert.equal(deleted, 1);
  const wrong = response(); await cleanup(request({ ...ticket, receipt: 'f'.repeat(64) }), wrong);
  assert.equal(wrong.code, 403); assert.equal(deleted, 1);
});

test('OIDC configuration and disabled uploads fail closed without falling back to static tokens', async () => {
  settings();
  assert.equal(control.requireSettings().blobToken, undefined);
  assert.equal(control.requireBlobOidc().storeId, 'examplestore');
  process.env.BLOB_READ_WRITE_TOKEN = 'unused-static-token';
  delete process.env.VERCEL_OIDC_TOKEN;
  assert.throws(() => control.requireSettings(), /OIDC/);
  settings(); process.env.BLOB_PRIVATE_HOST = 'otherstore.private.blob.vercel-storage.com';
  assert.throws(() => control.requireSettings(), /OIDC/);
  settings(); process.env.BLOB_STORE_ID = '../another';
  assert.throws(() => control.requireSettings(), /OIDC/);
  settings(); process.env.PRIVATE_UPLOADS_ENABLED = '0';
  const res = response(); await authorize(request({ size: 100, type: 'image/png', purpose: 'clip' }), res);
  assert.equal(res.code, 503);
});

test('OIDC issuance failures return a generic error without leaking SDK credentials', async () => {
  settings(); globalThis.fetch = async () => Response.json({ result: 1 });
  globalThis.blobFake = { issue: async () => { throw new Error('runtime-oidc-test-only'); } };
  const res = response(); await authorize(request({ size: 100, type: 'image/png', purpose: 'clip' }), res);
  assert.equal(res.code, 503); assert.equal(JSON.stringify(res.data).includes('runtime-oidc-test-only'), false);
});

test('inference forwards only the authorized receipt to the exact CLIP route and preserves its response', async () => {
  settings();
  const { default: infer } = await import('../frontend/react/api/uploads/infer.js');
  const ticket = { uploadId: 'a'.repeat(32), receipt: 'b'.repeat(64) };
  const record = { mime: 'image/png', size: 100, purpose: 'clip', status: 'issued',
    expires: Math.floor(Date.now()/1000) + 600, receiptHash: control.receiptHash(ticket.receipt),
    pathname: control.PREFIX + ticket.uploadId + '.png' };
  const prediction = { prediction: 'Shirt', confidence: 0.75, predictions: [] };
  const calls = [];
  globalThis.fetch = async (url, options) => {
    if (url.endsWith('.upstash.io')) return Response.json({ result: JSON.stringify(record) });
    calls.push([url, options]); return Response.json(prediction);
  };
  const res = response(); await infer(request(ticket), res);
  assert.equal(res.code, 200); assert.deepEqual(res.data, prediction);
  assert.equal(calls[0][0], 'https://fashion-vision-clip.vercel.app/api/predict/blob');
  assert.deepEqual(JSON.parse(calls[0][1].body), ticket);
  assert.equal(calls[0][1].redirect, 'error');
  const rejected = response(); await infer(request({ ...ticket, url: 'https://attacker.example' }), rejected);
  assert.equal(rejected.code, 403); assert.equal(calls.length, 1);
});

test('Vercel routing retains exact CLIP rewrites and all four frontend upload functions', async () => {
  const { readFileSync, existsSync } = await import('node:fs');
  const configuration = JSON.parse(readFileSync(new URL('../frontend/react/vercel.json', import.meta.url), 'utf8'));
  assert.deepEqual(configuration.rewrites, [
    { source: '/api/predict', destination: 'https://fashion-vision-clip.vercel.app/api/predict' },
    { source: '/api/health', destination: 'https://fashion-vision-clip.vercel.app/api/health' },
  ]);
  for (const action of ['authorize', 'infer', 'cleanup', 'sweep']) {
    assert(existsSync(new URL('../frontend/react/api/uploads/' + action + '.js', import.meta.url)));
    assert(configuration.functions['api/uploads/' + action + '.js']);
  }
  assert.deepEqual(configuration.crons, [{ path: '/api/uploads/sweep', schedule: '0 3 * * *' }]);
  assert.notEqual(configuration.env?.PRIVATE_UPLOADS_ENABLED, '1');
  const pkg = JSON.parse(readFileSync(new URL('../frontend/react/package.json', import.meta.url), 'utf8'));
  assert.equal(pkg.dependencies['@vercel/blob'], '2.8.1');
});
