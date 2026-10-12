import { proxyClip } from '../server/clipProxy.js';
export default { fetch(request) { return proxyClip(request); } };
