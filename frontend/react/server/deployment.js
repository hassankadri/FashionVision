// Server-only destinations and storage isolation. Never infer trust from request headers.
const FRONTEND = 'https://fashion-vision.vercel.app';
const CLIP = 'https://fashion-vision-clip.vercel.app';
function invalid() { throw new Error('Deployment routing is not configured.'); }
export function deployment() {
  const environment = process.env.VERCEL_ENV;
  if (!['production', 'preview'].includes(environment)) invalid();
  if (environment === 'production') {
    if ((process.env.UPLOAD_FRONTEND_ORIGIN && process.env.UPLOAD_FRONTEND_ORIGIN !== FRONTEND) ||
        (process.env.CLIP_API_ORIGIN && process.env.CLIP_API_ORIGIN !== CLIP) ||
        process.env.UPLOAD_ALLOWED_ORIGINS || process.env.PRIVATE_UPLOAD_NAMESPACE) invalid();
    return { environment, frontend: FRONTEND, origins: [FRONTEND], clip: CLIP,
      prefix: 'fashionvision-temp/v1/', redisPrefix: 'fv:' };
  }
  const frontend = process.env.UPLOAD_FRONTEND_ORIGIN || '';
  const clip = process.env.CLIP_API_ORIGIN || '';
  const namespace = process.env.PRIVATE_UPLOAD_NAMESPACE || '';
  const validFrontend = value => typeof value === 'string' && value === value.trim() &&
    /^https:\/\/fashion-vision-(?!clip-)[a-z0-9]+(?:-[a-z0-9]+)*\.vercel\.app$/.test(value);
  if (!validFrontend(frontend) || clip !== clip.trim() || namespace !== namespace.trim() ||
      !/^https:\/\/fashion-vision-clip-[a-z0-9]+(?:-[a-z0-9]+)*\.vercel\.app$/.test(clip) ||
      !/^[a-z0-9]+(?:-[a-z0-9]+)*$/.test(namespace) || namespace.length > 64) invalid();
  let extra;
  try { extra = JSON.parse(process.env.UPLOAD_ALLOWED_ORIGINS || '[]'); } catch { invalid(); }
  if (!Array.isArray(extra) || extra.length > 8 || !extra.every(validFrontend)) invalid();
  return { environment, frontend, clip, origins: [...new Set([frontend, ...extra])],
    prefix: 'fashionvision-temp/preview/' + namespace + '/', redisPrefix: 'fv:preview:' + namespace + ':' };
}
export function backendHeaders() {
  const config = deployment();
  const secret = process.env.CLIP_PREVIEW_BYPASS_SECRET;
  return config.environment === 'preview' && secret ? { 'x-vercel-protection-bypass': secret } : {};
}
