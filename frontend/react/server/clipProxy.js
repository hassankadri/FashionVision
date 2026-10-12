import { deployment, backendHeaders } from './deployment.js';
import { MAX_BYTES, MIME_EXT, sameOrigin, UploadError } from './uploads.js';

// Web Request bodies avoid Node helper parsing; original binary bytes stay intact.
export async function proxyClip(request, health = false) {
  const reply = (data, status) => Response.json(data, { status, headers: { 'Cache-Control': 'no-store' } });
  try {
    const config = deployment();
    if (request.method !== (health ? 'GET' : 'POST')) {
      return reply({ detail: health ? 'Use GET.' : 'Use POST.' }, 405);
    }
    let body;
    const headers = backendHeaders();
    if (!health) {
      sameOrigin({ headers: Object.fromEntries(request.headers) });
      const type = request.headers.get('content-type') || '';
      if (!Object.hasOwn(MIME_EXT, type.split(';')[0].toLowerCase())) throw new UploadError(415, 'Unsupported image type.');
      headers['Content-Type'] = type;
      if (Number(request.headers.get('content-length') || 0) > MAX_BYTES) throw new UploadError(413, 'Image exceeds 15 MiB.');
      const chunks = [];
      let size = 0;
      for await (const chunk of request.body || []) {
        size += chunk.length;
        if (size > MAX_BYTES) throw new UploadError(413, 'Image exceeds 15 MiB.');
        chunks.push(Buffer.from(chunk));
      }
      body = Buffer.concat(chunks);
      if (!body.length) throw new UploadError(400, 'Expected original image bytes.');
    }
    const upstream = await fetch(config.clip + (health ? '/api/health' : '/api/predict'), {
      method: health ? 'GET' : 'POST', headers, body, redirect: 'error', signal: AbortSignal.timeout(275000),
    });
    const chunks = [];
    let size = 0;
    for await (const chunk of upstream.body) {
      size += chunk.length;
      if (size > 2 * 1024 * 1024) throw new UploadError(502, 'Inference response is too large.');
      chunks.push(Buffer.from(chunk));
    }
    return reply(JSON.parse(Buffer.concat(chunks).toString('utf8')), upstream.status);
  } catch (error) {
    return reply({ detail: error instanceof UploadError ? error.message : 'Inference routing is temporarily unavailable.' },
      error instanceof UploadError ? error.status : 503);
  }
}
