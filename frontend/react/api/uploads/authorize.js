import { issueSignedToken, presignUrl } from '@vercel/blob';
import { allocate, jsonBody, sendError } from '../../server/uploads.js';

export default async function handler(req, res) {
  res.setHeader('Cache-Control', 'no-store');
  try {
    const authorization = await allocate(req, await jsonBody(req));
    const scope = {
      pathname: authorization.pathname, operations: ['put'],
      allowedContentTypes: [authorization.record.mime],
      maximumSizeInBytes: authorization.record.size,
      validUntil: authorization.validUntil,
    };
    // OIDC is resolved/refreshed by the SDK. Signing material stays server-side.
    const signedToken = await issueSignedToken({ ...scope, abortSignal: AbortSignal.timeout(10000) });
    const { presignedUrl } = await presignUrl(signedToken, {
      operation: 'put', access: 'private', pathname: scope.pathname,
      allowedContentTypes: scope.allowedContentTypes,
      maximumSizeInBytes: scope.maximumSizeInBytes, validUntil: scope.validUntil,
      addRandomSuffix: false, allowOverwrite: false, cacheControlMaxAge: 60,
    });
    res.status(200).json({ uploadId: authorization.uploadId, receipt: authorization.receipt,
      pathname: authorization.pathname, presignedUrl });
  } catch (error) { sendError(res, error); }
}
