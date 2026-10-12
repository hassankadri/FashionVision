const DIRECT_UPLOAD_BYTES = 4_000_000;
const MAX_BYTES = 15 * 1024 * 1024;
const PURPOSES = {
  '/api/predict': 'clip',
  '/api/predict/mobilenet': 'mobilenet',
  '/api/explain/mobilenet': 'gradcam',
};

async function responseData(response) {
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(typeof data.detail === 'string' ? data.detail : 'Request failed (HTTP ' + response.status + ').');
  }
  return data;
}

export async function postImage(endpoint, file, signal) {
  if (import.meta.env.DEV || file.size <= DIRECT_UPLOAD_BYTES) {
    return responseData(await fetch(endpoint, {
      method: 'POST', headers: { 'Content-Type': file.type }, body: file, signal,
    }));
  }
  if (file.size > MAX_BYTES) throw new Error('Images must be 15 MB or smaller.');
  if (!PURPOSES[endpoint]) throw new Error('Unsupported inference endpoint.');

  let ticket;
  try {
    const authorization = await responseData(await fetch('/api/uploads/authorize', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ size: file.size, type: file.type, purpose: PURPOSES[endpoint] }), signal,
    }));
    ticket = { uploadId: authorization.uploadId, receipt: authorization.receipt };
    await responseData(await fetch(authorization.presignedUrl, {
      method: 'PUT', headers: { 'Content-Type': file.type }, body: file, signal,
      credentials: 'omit', redirect: 'error', referrerPolicy: 'no-referrer', cache: 'no-store',
    }));
    return await responseData(await fetch('/api/uploads/infer', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(ticket), signal,
    }));
  } finally {
    if (ticket) {
      // Independent cancellation allows cleanup after upload/prediction cancellation.
      await fetch('/api/uploads/cleanup', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(ticket), keepalive: true, signal: AbortSignal.timeout(8000),
      }).catch(() => {});
    }
  }
}
