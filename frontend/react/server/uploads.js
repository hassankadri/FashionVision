import { createHash, createHmac, randomBytes, timingSafeEqual } from 'node:crypto';
import { isIP } from 'node:net';

export const MAX_BYTES = 15 * 1024 * 1024;
export const PREFIX = 'fashionvision-temp/v1/';
export const MIME_EXT = Object.freeze({
  'image/jpeg': 'jpg', 'image/png': 'png', 'image/webp': 'webp', 'image/bmp': 'bmp',
});
export const RECEIPT_TTL = 10 * 60;
export const RECORD_TTL = 48 * 60 * 60;
export const TOKEN_TTL = 5 * 60;
export const LIMITS = [3, 5, 10, 30]; // Per IP/10min, per IP/day, global/day, global/month.

export class UploadError extends Error {
  constructor(status, message) { super(message); this.status = status; }
}

export function requireBlobOidc() {
  const storeId = (process.env.BLOB_STORE_ID || '').toLowerCase().replace(/^store_/, '');
  const privateHost = storeId + '.private.blob.vercel-storage.com';
  // Check availability only; the SDK obtains and refreshes the actual OIDC token.
  if (process.env.VERCEL !== '1' || !process.env.VERCEL_OIDC_TOKEN ||
      !/^[a-z0-9]+$/.test(storeId) ||
      (process.env.BLOB_PRIVATE_HOST && process.env.BLOB_PRIVATE_HOST !== privateHost)) {
    throw new UploadError(503, 'Private Blob OIDC authentication is not configured.');
  }
  return { storeId, privateHost };
}

export function requireSettings() {
  if (process.env.PRIVATE_UPLOADS_ENABLED !== '1') {
    throw new UploadError(503, 'Private image uploads are not configured.');
  }
  const secret = process.env.UPLOAD_AUTH_SECRET;
  if (!/^[a-f0-9]{64}$/i.test(secret || '')) {
    throw new UploadError(503, 'Private image uploads are not configured.');
  }
  let url;
  try { url = new URL(process.env.UPSTASH_REDIS_REST_URL || ''); } catch {}
  if (!url || url.protocol !== 'https:' || !url.hostname.endsWith('.upstash.io') ||
      (url.port && url.port !== '443') || url.username || url.password || url.search || url.hash || !['', '/'].includes(url.pathname) ||
      !process.env.UPSTASH_REDIS_REST_TOKEN ||
      !/^[A-Za-z0-9_-]{32,}$/.test(process.env.CRON_SECRET || '')) {
    throw new UploadError(503, 'Private image uploads are not configured.');
  }
  requireBlobOidc();
  return { secret, redisUrl: url.origin, redisToken: process.env.UPSTASH_REDIS_REST_TOKEN };
}

export async function redis(command) {
  const settings = requireSettings();
  try {
    const response = await fetch(settings.redisUrl, {
      method: 'POST',
      headers: { Authorization: 'Bearer ' + settings.redisToken, 'Content-Type': 'application/json' },
      body: JSON.stringify(command), signal: AbortSignal.timeout(5000),
    });
    if (!response.ok) throw new Error('Redis request failed');
    const envelope = await response.json();
    if (envelope.error) throw new Error('Redis command failed');
    return envelope.result;
  } catch {
    throw new UploadError(503, 'Upload authorization is temporarily unavailable.');
  }
}

export function sameOrigin(req) {
  if (req.headers.origin !== 'https://fashion-vision.vercel.app' ||
      (req.headers['sec-fetch-site'] && req.headers['sec-fetch-site'] !== 'same-origin')) {
    throw new UploadError(403, 'This upload request is not allowed.');
  }
}

export function clientKey(req) {
  const settings = requireSettings();
  // Vercel overwrites this header; an arbitrary client identifier is not trusted.
  if (process.env.VERCEL !== '1') throw new UploadError(503, 'Private uploads require the configured Vercel environment.');
  const ip = req.headers['x-vercel-forwarded-for'];
  if (typeof ip !== 'string' || !isIP(ip.trim())) throw new UploadError(503, 'Upload rate limiting is unavailable.');
  return createHmac('sha256', settings.secret).update('rate:' + ip.trim()).digest('hex');
}

export async function jsonBody(req) {
  if (req.method !== 'POST') throw new UploadError(405, 'Use POST.');
  if ((req.headers['content-type'] || '').split(';')[0] !== 'application/json') {
    throw new UploadError(415, 'Use application/json.');
  }
  if (Number(req.headers['content-length'] || 0) > 2048) throw new UploadError(413, 'Request is too large.');
  let text;
  if (req.body !== undefined) {
    text = typeof req.body === 'string' ? req.body :
      Buffer.isBuffer(req.body) ? req.body.toString('utf8') : JSON.stringify(req.body);
  } else {
    const chunks = [];
    let size = 0;
    for await (const chunk of req) {
      size += Buffer.byteLength(chunk);
      if (size > 2048) throw new UploadError(413, 'Request is too large.');
      chunks.push(Buffer.from(chunk));
    }
    text = Buffer.concat(chunks).toString('utf8');
  }
  if (Buffer.byteLength(text || '') > 2048) throw new UploadError(413, 'Request is too large.');
  try {
    const body = JSON.parse(text);
    if (!body || typeof body !== 'object' || Array.isArray(body)) throw new Error();
    return body;
  } catch { throw new UploadError(400, 'Invalid upload request.'); }
}

export function receiptBody(body) {
  if (!body || typeof body !== 'object' || Array.isArray(body) ||
      Object.keys(body).sort().join(',') !== 'receipt,uploadId' ||
      typeof body.uploadId !== 'string' || typeof body.receipt !== 'string' ||
      !/^[a-f0-9]{32}$/.test(body.uploadId || '') || !/^[a-f0-9]{64}$/.test(body.receipt || '')) {
    throw new UploadError(403, 'Invalid or expired upload authorization.');
  }
  return body;
}
export const keyFor = id => 'fv:upload:v1:' + id;
export const receiptHash = value => createHash('sha256').update(value).digest('hex');

export function checkRecord(record, ticket, { expired = false } = {}) {
  receiptBody(ticket);
  if (!record || !['clip', 'mobilenet', 'gradcam'].includes(record.purpose) ||
      !Number.isSafeInteger(record.size) || record.size < 1 || record.size > MAX_BYTES ||
      !Number.isSafeInteger(record.expires) || !Object.hasOwn(MIME_EXT, record.mime) ||
      record.pathname !== PREFIX + ticket.uploadId + '.' + MIME_EXT[record.mime] ||
      record.receiptHash !== receiptHash(ticket.receipt) ||
      (!expired && record.expires <= Math.floor(Date.now() / 1000))) {
    throw new UploadError(403, 'Invalid or expired upload authorization.');
  }
  return record;
}
export async function getRecord(ticket, options) {
  receiptBody(ticket);
  const value = await redis(['GET', keyFor(ticket.uploadId)]);
  let record;
  try { record = value ? JSON.parse(value) : null; } catch {}
  return checkRecord(record, ticket, options);
}

export function backendFor(purpose) {
  if (purpose === 'clip') return ['https://fashion-vision-clip.vercel.app', '/api/predict/blob'];
  const origin = process.env.MOBILENET_API_ORIGIN || '';
  if (!/^https:\/\/[a-z0-9-]+\.vercel\.app$/.test(origin) || origin === 'https://fashion-vision-clip.vercel.app') {
    throw new UploadError(503, 'MobileNet private uploads are not configured.');
  }
  return [origin, purpose === 'mobilenet' ? '/api/predict/mobilenet/blob' : '/api/explain/mobilenet/blob'];
}

const ALLOCATE = [
  "for i=1,4 do",
  "  if tonumber(redis.call('GET', KEYS[i]) or '0') >= tonumber(ARGV[i]) then return 0 end",
  "end",
  "if redis.call('EXISTS', KEYS[5]) == 1 then return -1 end",
  "for i=1,4 do",
  "  if redis.call('INCR', KEYS[i]) == 1 then redis.call('EXPIRE', KEYS[i], ARGV[i+4]) end",
  "end",
  "redis.call('SET', KEYS[5], ARGV[9], 'EX', ARGV[10])",
  "return 1",
].join('\n');
export async function allocate(req, body) {
  const settings = requireSettings();
  sameOrigin(req);
  if (Object.keys(body).sort().join(',') !== 'purpose,size,type' ||
      !Number.isSafeInteger(body.size) || body.size < 1 || body.size > MAX_BYTES || !Object.hasOwn(MIME_EXT, body.type) ||
      !['clip', 'mobilenet', 'gradcam'].includes(body.purpose)) {
    throw new UploadError(400, 'Choose a supported image no larger than 15 MiB.');
  }
  backendFor(body.purpose);
  const now = Math.floor(Date.now() / 1000);
  const uploadId = randomBytes(16).toString('hex');
  const receipt = randomBytes(32).toString('hex');
  const pathname = PREFIX + uploadId + '.' + MIME_EXT[body.type];
  const record = { pathname, mime: body.type, size: body.size, purpose: body.purpose,
    expires: now + RECEIPT_TTL, receiptHash: receiptHash(receipt), status: 'issued' };
  const ip = clientKey(req);
  const day = new Date().toISOString().slice(0, 10);
  const month = day.slice(0, 7);
  const keys = ['fv:rate:' + ip + ':' + Math.floor(now / 600), 'fv:rate:' + ip + ':' + day,
    'fv:quota:day:' + day, 'fv:quota:month:' + month, keyFor(uploadId)];
  const result = await redis(['EVAL', ALLOCATE, 5, ...keys, ...LIMITS,
    600, 86400, 86400, 35 * 86400, JSON.stringify(record), RECORD_TTL]);
  if (result !== 1) throw new UploadError(429, 'The upload limit has been reached. Please try again later.');
  return { settings, uploadId, receipt, pathname, record, validUntil: (now + TOKEN_TTL) * 1000 };
}

const START_CLEANUP = [
  "local value=redis.call('GET',KEYS[1])",
  "if not value then return '' end",
  "local record=cjson.decode(value)",
  "if record.receiptHash ~= ARGV[1] then return '' end",
  "if record.status == 'deleted' then return 'deleted' end",
  "if record.status == 'processing' and ARGV[2] ~= '1' then return 'processing' end",
  "if tonumber(record.cleanupAttempts or 0) >= 3 then return 'pending' end",
  "record.cleanupAttempts=tonumber(record.cleanupAttempts or 0)+1",
  "record.status='cleanup'",
  "redis.call('SET',KEYS[1],cjson.encode(record),'EX',ARGV[3])",
  "return record.pathname",
].join('\n');
export function cleanupProof(ticket) {
  return createHmac('sha256', requireSettings().secret)
    .update('cleanup:' + ticket.uploadId + ':' + ticket.receipt).digest('hex');
}
export async function beginCleanup(req, ticket) {
  const record = await getRecord(ticket, { expired: true });
  const supplied = req.headers['x-fv-cleanup-proof'] || '';
  const expected = cleanupProof(ticket);
  const internal = typeof supplied === 'string' && /^[a-f0-9]{64}$/.test(supplied) &&
    timingSafeEqual(Buffer.from(supplied), Buffer.from(expected));
  const result = await redis(['EVAL', START_CLEANUP, 1, keyFor(ticket.uploadId),
    record.receiptHash, internal ? '1' : '0', RECORD_TTL]);
  if (!result) throw new UploadError(403, 'Invalid or expired upload authorization.');
  if (['deleted', 'processing', 'pending'].includes(result)) return { result, record };
  if (result !== record.pathname) throw new UploadError(403, 'Invalid upload authorization.');
  return { result: 'delete', record };
}

export function sendError(res, error) {
  res.setHeader('Cache-Control', 'no-store');
  if (error.status === 405) res.setHeader('Allow', 'POST');
  res.status(error instanceof UploadError ? error.status : 503).json({
    detail: error instanceof UploadError ? error.message : 'Private image uploads are temporarily unavailable.',
  });
}
