import { del } from '@vercel/blob';
import { beginCleanup, jsonBody, keyFor, receiptBody, redis, sendError } from '../../server/uploads.js';

const COMPLETE = [
  "local value=redis.call('GET',KEYS[1])",
  "if not value then return 0 end",
  "local record=cjson.decode(value)",
  "if record.receiptHash ~= ARGV[1] then return 0 end",
  "record.status='deleted'",
  "redis.call('SET',KEYS[1],cjson.encode(record),'EX',ARGV[2])",
  "return 1",
].join('\n');
export default async function handler(req, res) {
  res.setHeader('Cache-Control', 'no-store');
  try {
    const ticket = receiptBody(await jsonBody(req));
    const { result, record } = await beginCleanup(req, ticket);
    if (result === 'delete') {
      await del(record.pathname, { abortSignal: AbortSignal.timeout(10000) });
      await redis(['EVAL', COMPLETE, 1, keyFor(ticket.uploadId), record.receiptHash, 48 * 3600]);
      res.status(200).json({ status: 'deleted' });
    } else {
      res.status(result === 'deleted' ? 200 : 202).json({ status: result });
    }
  } catch (error) { sendError(res, error); }
}
