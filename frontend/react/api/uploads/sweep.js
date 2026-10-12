import { timingSafeEqual } from 'node:crypto';
import { del, list } from '@vercel/blob';
import { PREFIX, requireBlobOidc } from '../../server/uploads.js';

export default async function handler(req, res) {
  res.setHeader('Cache-Control', 'no-store');
  const expected = 'Bearer ' + (process.env.CRON_SECRET || '');
  const supplied = req.headers.authorization || '';
  if (req.method !== 'GET' || !/^[A-Za-z0-9_-]{32,}$/.test(process.env.CRON_SECRET || '') ||
      typeof supplied !== 'string' || supplied.length !== expected.length ||
      !timingSafeEqual(Buffer.from(supplied), Buffer.from(expected))) {
    return res.status(403).json({ detail: 'Not authorized.' });
  }
  let cursor;
  let deleted = 0;
  const started = Date.now();
  try {
    requireBlobOidc();
    do {
      const page = await list({ prefix: PREFIX, limit: 100, cursor,
        abortSignal: AbortSignal.timeout(10000) });
      const expired = page.blobs.filter(blob =>
        /^fashionvision-temp\/v1\/[a-f0-9]{32}\.(jpg|png|webp|bmp)$/.test(blob.pathname) &&
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
