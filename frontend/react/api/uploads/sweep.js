import { timingSafeEqual } from 'node:crypto';
import { del, list } from '@vercel/blob';
import { requireBlobOidc } from '../../server/uploads.js';

import { deployment } from '../../server/deployment.js';

export default async function handler(req, res) {
  res.setHeader('Cache-Control', 'no-store');
  const expected = 'Bearer ' + (process.env.CRON_SECRET || '');
  const supplied = req.headers.authorization || '';
  if (req.method !== 'GET' || (process.env.CRON_SECRET || '').length !== 64 || !/^[a-f0-9]{64}$/.test(process.env.CRON_SECRET || '') ||
      typeof supplied !== 'string' || supplied.length !== expected.length ||
      !timingSafeEqual(Buffer.from(supplied), Buffer.from(expected))) {
    return res.status(403).json({ detail: 'Not authorized.' });
  }
  let cursor;
  let deleted = 0;
  const started = Date.now();
  try {
    requireBlobOidc();
    const prefix = deployment().prefix;
    do {
      const page = await list({ prefix, limit: 100, cursor,
        abortSignal: AbortSignal.timeout(10000) });
      const expired = page.blobs.filter(blob =>
        blob.pathname.startsWith(prefix) && /^[a-f0-9]{32}\.(jpg|png|webp|bmp)$/.test(blob.pathname.slice(prefix.length)) &&
        new Date(blob.uploadedAt).getTime() < Date.now() - 3600000);
      if (expired.length) {
        await del(expired.map(blob => blob.pathname), { abortSignal: AbortSignal.timeout(10000) });
        deleted += expired.length;
      }
      cursor = page.hasMore ? page.cursor : undefined;
      if (Date.now() - started > 40000 && cursor) {
        return res.status(503).json({ detail: 'Cleanup incomplete; retry or inspect the store.', deleted });
      }
    } while (cursor);
    return res.status(200).json({ deleted });
  } catch {
    return res.status(503).json({ detail: 'Temporary object cleanup failed; retry or inspect the store.', deleted });
  }
}
