import { backendFor, getRecord, jsonBody, receiptBody, sameOrigin, sendError, UploadError } from '../../server/uploads.js';

export default async function handler(req, res) {
  res.setHeader('Cache-Control', 'no-store');
  try {
    sameOrigin(req);
    const ticket = receiptBody(await jsonBody(req));
    const record = await getRecord(ticket);
    if (record.status !== 'issued') throw new UploadError(403, 'Invalid or expired upload authorization.');
    const [origin, endpoint] = backendFor(record.purpose);
    const response = await fetch(origin + endpoint, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(ticket), redirect: 'error', signal: AbortSignal.timeout(275000),
    });
    const chunks = [];
    let size = 0;
    for await (const chunk of response.body) {
      size += chunk.length;
      if (size > 2 * 1024 * 1024) throw new UploadError(502, 'The inference response is too large.');
      chunks.push(Buffer.from(chunk));
    }
    const data = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    res.status(response.status).json(data);
  } catch (error) { sendError(res, error); }
}
