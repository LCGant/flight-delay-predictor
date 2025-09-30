const CACHE_TTL_MS = 10 * 60 * 1000;
const COOLDOWN_MS = 15 * 1000;

const cache = new Map();
const inFlight = new Map();
const lastHit = new Map();

function getCache(key) {
  const item = cache.get(key);
  if (!item) return null;
  if (Date.now() - item.ts > CACHE_TTL_MS) { cache.delete(key); return null; }
  return item.data;
}

function setCache(key, data) {
  cache.set(key, { ts: Date.now(), data });
  if (cache.size > 300) {
    const first = cache.keys().next().value;
    if (first) cache.delete(first);
  }
}

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (!msg || msg.type !== "FDRB_PREDICT") return;

  const url = msg.url;
  const payload = msg.payload || {};
  const key = msg.request_key || JSON.stringify(payload);

  const now = Date.now();
  const last = lastHit.get(key) || 0;
  if (now - last < COOLDOWN_MS) {
    const cached = getCache(key);
    if (cached) { sendResponse({ ok: true, data: cached }); return true; }
    const infl = inFlight.get(key);
    if (infl) {
      infl.then(d => sendResponse({ ok: true, data: d }))
          .catch(e => sendResponse({ ok: false, error: String(e) }));
      return true;
    }
  }
  lastHit.set(key, now);

  const cached = getCache(key);
  if (cached) { sendResponse({ ok: true, data: cached }); return true; }

  if (inFlight.has(key)) {
    inFlight.get(key)
      .then(d => sendResponse({ ok: true, data: d }))
      .catch(e => sendResponse({ ok: false, error: String(e) }));
    return true;
  }

  const p = fetch(url, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(payload),
  })
    .then(async r => {
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const data = await r.json();
      setCache(key, data);
      return data;
    })
    .finally(() => inFlight.delete(key));

  inFlight.set(key, p);

  p.then(d => sendResponse({ ok: true, data: d }))
   .catch(e => sendResponse({ ok: false, error: String(e) }));

  return true;
});
