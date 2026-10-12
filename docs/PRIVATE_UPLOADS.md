# Private image uploads

The implementation is prepared, not deployed. Keep `PRIVATE_UPLOADS_ENABLED=0`
(or absent) on every project; this audit does not enable it. Large uploads return
503 while disabled, even when OIDC is connected. Existing small-image requests
and local development do not depend on Blob or Redis.

The maintainer reports that `fashion-vision` and `fashion-vision-clip` are connected
to the same private store through OIDC. Cloud permissions and actual operations
have not been exercised in this offline audit.

## Request flow

1. In a production build, files up to **4,000,000 bytes** use the existing
   raw-image endpoint. Vercel's two CLIP rewrites remain unchanged.
2. Larger files, up to **15 MiB**, request authorization from
   `POST /api/uploads/authorize`. Only declared size, image MIME type and an
   enumerated operation are accepted; original filenames are not sent.
3. The Node function allocates a random object name and a 256-bit capability,
   records a SHA-256 hash of the capability in Redis, and calls the supported
   OIDC-compatible `issueSignedToken()` and `presignUrl()` methods. The delegation
   permits only `put` on the exact generated pathname, declared MIME and byte
   ceiling, with a **five-minute** expiry. The presigned URL also disallows
   overwrites/random suffixes and selects a 60-second CDN cache lifetime.
   Neither the runtime OIDC token nor the delegation's client signing key is
   included in the response.
4. The browser uses the documented presigned `PUT` flow with the same File object.
   Bytes travel directly to private Blob storage, without re-encoding, compression,
   multipart or a serverless upload proxy. The request omits browser credentials,
   refuses redirects and sends no referrer. Only the scoped upload URL is exposed.
5. `POST /api/uploads/infer` forwards only the opaque upload ID and capability
   to the fixed backend endpoint selected by the server-side record. The
   browser cannot supply a destination URL or pathname.
6. The Python backend atomically claims the receipt in Redis. The claim checks
   its hash, operation, expiry and unused status. It constructs the private
   object's URL from the connected `BLOB_STORE_ID` and validated generated
   pathname. The official lightweight Python OIDC helper resolves a fresh token
   from the request context (including expiry/refresh handling); the backend uses
   the documented private HTTP Bearer read, refusing redirects. No token is cached
   by application code or forwarded from the frontend project.
   It bounds the streamed bytes, verifies exact authorized size and MIME
   metadata, and checks Pillow's actual format, decoding and dimensions before
   calling the existing inference function.
7. Prediction/Grad-CAM response fields are passed through unchanged. Backend
   finally cleanup and browser finally cleanup attempt deletion even when
   validation, retrieval or inference fails, or the user cancels.

No image bytes pass through the authorization or forwarding functions. Model
files, CLIP prompts/tokenizer/preprocessing, MobileNet preprocessing and the
crop editor are unchanged. Local Vite development always uses the original raw
upload path, including images above 4 MB.

## Required configuration

**Configuration to verify:** A persistent **free-tier Upstash Redis** database
must be shared by the frontend and both backend projects. Blob alone cannot
supply distributed counters or single-use receipt state. Redis configuration has
not been inspected or verified in this audit. Do not upgrade to a paid plan.
Missing, exhausted or unreachable Redis fails closed; there is no in-memory fallback.

Set these server-only environment variables on the frontend, CLIP backend and,
when deployed, MobileNet backend:

| Variable | Value |
| --- | --- |
| `PRIVATE_UPLOADS_ENABLED` | Keep `0` or absent; a future authorized activation requires successful cloud checks |
| `BLOB_STORE_ID` | Vercel-managed connected store ID, either `store_<id>` or `<id>`; it must identify the same private store on every project |
| `VERCEL_OIDC_TOKEN` | Vercel-managed short-lived runtime credential; do not copy, persist or manually set it |
| `BLOB_PRIVATE_HOST` | Optional consistency check against the host derived from `BLOB_STORE_ID`; remove a stale value or correct it |
| `UPSTASH_REDIS_REST_URL` | Free Redis HTTPS REST endpoint ending in `.upstash.io` |
| `UPSTASH_REDIS_REST_TOKEN` | Free Redis REST credential |
| `UPLOAD_AUTH_SECRET` | The same independently generated 32-byte secret, encoded as 64 hex characters |

Additional **frontend-only** settings:

- `CRON_SECRET`: independently generated random secret of at least 32 URL-safe
  characters. Vercel supplies it as the cron Authorization bearer token.
- `MOBILENET_API_ORIGIN`: optional verified MobileNet backend origin such as
  `https://<your-mobilenet-project>.vercel.app`; no path or trailing slash.
  Without this, MobileNet/Grad-CAM large-image authorization fails before an
  upload. CLIP is never substituted for MobileNet.

Never prefix any of these with `VITE_`, put them in source, or log them.
`BLOB_READ_WRITE_TOKEN` is no longer required or passed to Blob operations.
The Node SDK resolves and refreshes OIDC automatically; Python binds the current
request headers to Vercel's official OIDC helper and resolves the token just before
each read. OIDC failures produce a generic 503, without a static-token fallback.
`generateClientTokenFromReadWriteToken`/`handleUpload` still require static tokens;
they are replaced by the officially supported presigned PUT flow.
No webhook callback is used, so `BLOB_WEBHOOK_PUBLIC_KEY` is not required by this
implementation. Give preview deployments no production upload settings by default. Origin checks currently allow only
`https://fashion-vision.vercel.app`; local development does not mint tokens.

Frontend Vercel Root Directory remains `frontend/react`, preset Vite, build
`npm run build`, output `dist`. The frontend now also has four standard Node
Functions under `api/uploads/`. Fluid Compute and durations are configured in
its vercel.json. Its daily cleanup schedule is `0 3 * * *` (03:00 UTC), within
Hobby's once-per-day restriction. Enable source files outside Root Directory
for the Python backend build staging, as before. Both Python staging scripts
now copy the shared Blob transport module alongside their canonical code.

Deploy the backend updates and verify the new endpoints before enabling the
frontend feature in a future authorized release. This work does not deploy,
push, configure a cloud account or enable the feature on the live website.

## Security and abuse controls

- Server-generated, random filenames under `fashionvision-temp/v1/`.
  No user-supplied object paths, arbitrary hosts, public objects, or download
  endpoint is exposed.
- Ten-minute, 256-bit bearer capabilities authorize only their recorded
  operation/object. A Redis Lua claim allows **one inference** across all workers.
  Knowingly sharing a capability grants access to that operation; do not log it.
- Presigned upload URLs expire in five minutes, authorize only `put` at the
  exact generated path, allow only the declared MIME and size, and disallow
  overwriting an existing object. Treat these URLs as temporary bearer secrets;
  do not log them. Signing keys and runtime tokens remain server-side.
- All control requests are bounded to 2 KiB; inference responses are bounded to
  2 MiB. Private reads reject redirects and oversized/mismatched data.
- Actual image checks: JPEG/PNG/WebP/BMP only, positive dimensions, maximum
  8192 per side and 16 million pixels, full decoding and format/MIME agreement.
  These are transport safeguards; they do not alter image bytes or inference.
- Atomic authorization quotas: **3 per IP per ten minutes**, **5 per IP per
  day**, **10 globally per day**, and **30 globally per calendar month**.
  These intentionally conservative portfolio defaults count attempts, including
  failed/abandoned uploads. Counters are never refunded by cleanup.
- IPs come from Vercel's overwritten `x-vercel-forwarded-for` header.
  Redis stores keyed HMAC pseudonyms, not raw IP addresses. Origin checks help
  prevent browser cross-site requests; they are not user authentication and do
  not stop a non-browser attacker spoofing an Origin header.
- Cleanup during active inference needs a backend HMAC proof, preventing
  browser cancellation cleanup from deleting an image still being read.
  Invalid receipts never authorize reads/deletion. Cleanup attempts are bounded.
- Existing small-image endpoints remain public and unchanged. These new
  controls do not make every existing endpoint abuse-proof. A distributed
  attacker can exhaust the conservative quotas and deny large uploads.

## Cleanup and retention

The backend requests deletion in a finally block after a successful claim,
including retrieval, validation and model failures. The browser independently
requests cleanup after completion/cancellation. SDK deletion is idempotent;
confirmed deleted receipts cannot start another inference.

There is no guarantee that finally runs after a hard process kill, or that a
network/delete request succeeds. Failures leave the object for the protected
daily sweep. That job lists only the application namespace and deletes matching
temporary images older than **one hour**, with bounded network calls/runtime.
It works even when Redis or uploads are disabled.

**Abandoned uploads are not deleted after exactly one hour.** With daily Hobby
cron plus scheduling jitter, deletion normally occurs at the next eligible
daily run and may be over a day after upload. Outages, quota exhaustion,
incomplete sweeps or missed jobs can extend retention. Monitor the job and
manually retry its authenticated endpoint or inspect the store if it fails.

A presigned upload URL is not a one-use authorization. An owner may finish late
or recreate the same pathname after immediate deletion while its five-minute URL
is still valid. The one-use inference receipt cannot be reused; the later object
is covered by the sweep. This is why immediate deletion alone is insufficient.

Individual authorization/tombstone records expire after 48 hours from their last
state transition. Per-IP counters last up to ten minutes or 24 hours. Global
monthly aggregate counters expire after 35 days and contain no IP identifiers.
Redis records contain object identifiers, declared MIME/size, operation, expiry,
receipt hashes and processing/cleanup state, never image bytes or predictions.
Deletion from storage does not promise immediate erasure from provider caches,
backups or logs. Blob CDN eviction can take up to a minute; hosting retention
remains provider/configuration dependent.

## Costs and remaining checks

Only Vercel Hobby capabilities and a genuinely free Redis plan are used.
Hobby Blob currently includes 1 GB storage, 10 GB Blob transfer, 10,000 simple
operations and 2,000 advanced operations. Hobby pauses Blob at exceeded limits
rather than charging overage. Redis's free plan also has finite limits. The
monthly authorization cap helps constrain usage but is not an absolute guarantee
against token replay, other projects sharing quotas, bot traffic or existing
small-image inference consuming Vercel compute. No paid WAF rate limiter is used.

Before activation, verify:

- The actual Blob host belongs to the private store and all projects have the
  matching credentials and shared Redis. Confirm free-plan settings.
- Confirm OIDC connections include the target environment on each project; connect
  the future MobileNet project to the same private store before using its Blob routes.
- SDK authorization, direct private upload, backend retrieval and cleanup on
  Vercel; deployed backend changes and Node Function routing have not been tested.
  Local routing remains exact: `/api/uploads/*` maps to four frontend Node
  Functions, `/api/predict` and `/api/health` are the only CLIP rewrites.
  No wildcard rewrite or React build requirement is added to the Python backend.
- Real Redis Lua atomic concurrency, expiry/replay rejection, quota exhaustion,
  and unavailable Redis behavior across instances.
- Cancellation, late uploads, cleanup failure/retry and the authenticated daily
  cron invocation; verify no unrelated store files are touched.
- CLIP/MobileNet Linux bundle size, memory and cold starts with the transport
  module and new lightweight Python dependencies. `vercel-oidc==0.10.0`,
  `vercel-headers==0.7.2` and `httpx2==2.13.1` are declared in both backend
  requirements; no full Vercel platform SDK or TensorFlow dependency is added to CLIP.
  The OIDC packages are not installed in the current local environment, so offline
  helper tests use controlled substitutes. Production dependency installation and
  the final Linux bundle size must be verified by a later authorized build.
- Separate small-image MobileNet routing remains a separate configuration task;
  it is never redirected to CLIP by this implementation.

References:
[OIDC and SDK authentication](https://vercel.com/docs/vercel-blob/using-blob-sdk),
[presigned PUT flow](https://vercel.com/docs/vercel-blob/vercel-signed-urls),
[official Python OIDC helper](https://github.com/vercel/vercel-py/tree/main/src/vercel/oidc),
[private Blob reads](https://vercel.com/docs/vercel-blob/private-storage),
[Blob pricing](https://vercel.com/docs/vercel-blob/usage-and-pricing),
[Hobby cron restrictions](https://vercel.com/docs/cron-jobs/usage-and-pricing),
[trusted request headers](https://vercel.com/docs/headers/request-headers),
[free Redis pricing](https://upstash.com/pricing/redis).
