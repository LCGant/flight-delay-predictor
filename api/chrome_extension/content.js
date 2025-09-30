(() => {
  const API_URL = "http://127.0.0.1:8000/predict";
  const log = (...a) => console.log("[FDRB]", ...a);

  const CARD_DEBOUNCE_MS = 350;
  const API_TTL_MS = 10 * 60 * 1000;
  const KEEPALIVE_MS = 4000;
  const SCAN_THROTTLE_MS = 150;
  const IS_FLIGHTS_PAGE = /\/travel\/flights\//.test(location.pathname);

  const seenKeys = new Set();
  const pillDataCache = new Map();
  const cardDebounce = new WeakMap();
  const cardKey = new WeakMap();
  const pillLastSeen = new WeakMap();
  function markPillSeen(pillEl) { try { pillLastSeen.set(pillEl, performance.now()); } catch {} }

  let modalRoot = null;
  let modalObserver = null;
  let keepAliveTimer = null;
  let scanThrottleTimer = null;

  const IATA_SET = new Set(["GRU","CGH","VCP","GIG","SDU","BSB","CNF","CWB","POA","FLN","SSA","REC","FOR","BEL","MAO","NAT","SLZ","CGB","CGR","IGU","BPS","UDI","JPA","MCZ","THE","GYN","VIX","ARU","JDF","RAO"]);
  const AIRLINE_NAME_TO_IATA = { "GOL":"G3","GOL LINHAS AÉREAS":"G3","GOL LINHAS AEREAS":"G3","LATAM":"LA","LATAM AIRLINES BRASIL":"LA","AZUL":"AD","AZUL LINHAS AÉREAS":"AD","AZUL LINHAS AEREAS":"AD","VOEPASS":"2Z" };

  const MONTHS = { "jan":1,"fev":2,"mar":3,"abr":4,"mai":5,"jun":6,"jul":7,"ago":8,"set":9,"out":10,"nov":11,"dez":12 };
  const pad2 = (n)=>String(n).padStart(2,"0");
  function addDaysUTC(isoDate, days=0) { const [y,m,d] = isoDate.split("-").map(Number); const dt = new Date(Date.UTC(y, m-1, d)); dt.setUTCDate(dt.getUTCDate() + days); return `${dt.getUTCFullYear()}-${pad2(dt.getUTCMonth()+1)}-${pad2(dt.getUTCDate())}`; }
  function parsePtDate(str) { const s=(str||"").toLowerCase(); const m=s.match(/(\d{1,2})\s+de\s+([a-zç]+)/i); if(!m) return null; const day=+m[1]; const mon=MONTHS[(m[2]||"").slice(0,3)]; if(!mon) return null; const today=new Date(); let y=today.getFullYear(); if(mon < (today.getMonth()+1) - 6) y += 1; return `${y}-${pad2(mon)}-${pad2(day)}`; }
  function dateFromHeader() { const scope=document.querySelector('header')||document.body; const els=scope?scope.querySelectorAll('button,div,span,time'):[]; for(const el of els){ const t=(el.textContent||"").trim(); if(/\d{1,2}\s+de\s+[a-z]/i.test(t)){ const d=parsePtDate(t); if(d) return d; } } return null; }
  function dateFromURL() { try { const u = new URL(location.href); const tfs = u.searchParams.get("tfs") || ""; const tokens = tfs.match(/[A-Za-z0-9_\-]{10,}/g) || []; for (let tok of tokens) { try { const b64 = tok.replace(/-/g, "+").replace(/_/g, "/"); const txt = atob(b64); const m = txt.match(/20\d{2}-\d{2}-\d{2}/); if (m) return m[0]; } catch {} } } catch {} return null; }
  function guessOutboundISODate() { return dateFromHeader() || dateFromURL() || new Date().toISOString().slice(0,10); }
  function guessRouteFromHeader() { const header=document.querySelector('header'); if(!header) return null; const text=header.innerText||""; const m=text.match(/\b([A-Z]{3})\b/g); if(!m) return null; const codes=m.filter(c=>IATA_SET.has(c)); const uniq=[]; for(const c of codes) if(!uniq.includes(c)) uniq.push(c); if(uniq.length>=2) return {origin:uniq[0],dest:uniq[1]}; return null; }

  function extractAirlineIata(textUpper) { for (const name of Object.keys(AIRLINE_NAME_TO_IATA)) { if (textUpper.includes(name)) return AIRLINE_NAME_TO_IATA[name]; } return null; }
  function extractOperatingAirlineIata(textUpper) { const m=textUpper.match(/OPERAD[OA]\s+POR\s+([A-Z\u00C0-\u017F ]{2,})/); if(m){ const full=m[1].replace(/\s+/g," ").trim(); for(const k of Object.keys(AIRLINE_NAME_TO_IATA)){ if(full.includes(k)) return AIRLINE_NAME_TO_IATA[k]; } } return null; }
  function extractFlightNumber(text) { const rx=/\b([A-Z0-9]{2})\s?(\d{1,4}[A-Z]?)\b/g; let best=null,m; while((m=rx.exec(text))!==null){ const pfx=m[1].toUpperCase(); if(IATA_SET.has(pfx)) continue; best=`${pfx}${m[2]}`; if(["LA","G3","AD","2Z"].includes(pfx)) break; } return best; }
  function extractDepArrTimes(card) { const text=card.innerText||""; const m=text.match(/(\d{1,2}:\d{2})\s*[–—-]\s*(\d{1,2}:\d{2})(?:\s*\+(\d))?/); if(!m) return null; return { dep:m[1].padStart(5,"0"), arr:m[2].padStart(5,"0"), plusDays:m[3]?parseInt(m[3],10):0 }; }
  function computeArrivalISO(depDateISO, depHHMM, arrHHMM, plusDays) { if(!depDateISO || !depHHMM || !arrHHMM) return null; const [dh,dm]=depHHMM.split(":").map(Number); const [ah,am]=arrHHMM.split(":").map(Number); let add=plusDays||0; if(!plusDays && (ah*60+am)<(dh*60+dm)) add=1; const date=addDaysUTC(depDateISO, add); return `${date}T${arrHHMM}`; }
  function parseDurationMinutes(text) { const s=(text||"").toLowerCase(); const m1=s.match(/(\d+)\s*h(?:\s*(\d+)\s*min)?/); if(m1){ const h=parseInt(m1[1],10); const mm=m1[2]?parseInt(m1[2],10):0; return h*60+mm; } const m2=s.match(/(\d+)\s*min/); if(m2) return parseInt(m2[1],10); return null; }
  function extractStops(text) { const s=(text||"").toLowerCase(); if(/\b(direto|sem\s+escala[s]?|sem\s+parada[s]?)\b/.test(s)) return 0; const m=s.match(/(\d+)\s*(parada|paradas|escala|escalas)/); if(m) return parseInt(m[1],10); return null; }

  function isVisible(el) { if(!el || !el.isConnected) return false; const st=getComputedStyle(el); if(st.display==="none"||st.visibility==="hidden"||parseFloat(st.opacity)<0.01) return false; const r=el.getBoundingClientRect(); if(r.width<4||r.height<4) return false; return true; }
  function hasTimePattern(root) { const txt=(root.innerText||""); return /\b\d{1,2}:\d{2}\s*[–—-]\s*\d{1,2}:\d{2}\b/.test(txt); }

  function createPillColumn(key, inModal=false) {
    const pillCol = document.createElement("div");
    pillCol.className = "fdrb-pill-column" + (inModal ? " in-modal" : "");
    pillCol.dataset.key = key;
    pillCol.style.cssText = `display:flex!important;flex-direction:column!important;justify-content:center!important;align-items:flex-start!important;padding:0 16px!important;min-width:120px!important;font-family:'Google Sans Text','Google Sans',Roboto,Arial,sans-serif!important;${inModal ? "z-index:2!important;" : ""}`;
    const wrap = document.createElement("div");
    wrap.className = "fdrb-pill-wrapper";
    wrap.style.cssText = `display:flex!important;flex-direction:column!important;gap:4px!important;`;
    const label = document.createElement("div");
    label.className = "fdrb-pill-label";
    label.textContent = "Risco de atraso";
    label.style.cssText = `font-size:11px!important;color:#70757a!important;font-weight:400!important;line-height:1.2!important;margin:0!important;`;
    const value = document.createElement("div");
    value.className = "fdrb-pill-value skeleton";
    value.textContent = "Calculando...";
    value.style.cssText = `font-size:14px!important;font-weight:500!important;color:#3c4043!important;line-height:1.2!important;margin:0!important;`;
    wrap.appendChild(label); wrap.appendChild(value); pillCol.appendChild(wrap);
    return { column: pillCol, value };
  }
  function updatePillValue(valueEl, p, thr) {
    const pctStr = (p*100).toLocaleString("pt-BR", { minimumFractionDigits: 1, maximumFractionDigits: 1 }) + "%";
    valueEl.classList.remove("skeleton","error");
    valueEl.textContent = pctStr;
    const base = `font-size:14px!important;font-weight:500!important;line-height:1.2!important;margin:0!important;`;
    if (p >= thr) valueEl.style.cssText = base + `color:#ea4335!important;`;
    else if (p >= thr*0.7) valueEl.style.cssText = base + `color:#fbbc04!important;`;
    else valueEl.style.cssText = base + `color:#34a853!important;`;
  }

  const BASE_CARD_SELECTORS = ['.mxvQLc.uj4xv','.pIav2d','.yR1fYc','[role="listitem"]','[data-flt-ve="result"]'];
  const MODAL_ROOT_SELECTORS = ['[role="dialog"][aria-modal="true"]','.Rk10dc[role="dialog"]','c-wiz[role="dialog"]','.VfPpkd-ww9zNc'];
  const MODAL_CARD_SELECTORS = ['.OgQvJf.nKlB3b','.mxvQLc','[class*="flight-card"]','[data-flt-ve="result"]','[role="listitem"]'];

  function inModalNode(node) { return !!modalRoot && node && modalRoot.contains(node); }
  function findInsertionPoint(card, inModal) {
    const selectors = inModal ? MODAL_CARD_SELECTORS : ['.OgQvJf.nKlB3b','.mxvQLc','[class*="flight-card"]','[role="listitem"]','[data-flt-ve="result"]'];
    for (const selector of selectors) {
      const container = card.querySelector(selector) || (card.matches(selector) ? card : null);
      if (!container) continue;
      const columnsContainer = container.querySelector('.Ak5kof, .BbR8Ec')?.parentElement;
      if (columnsContainer) return columnsContainer;
      const children = Array.from(container.children);
      if (children.length >= 2) {
        const targetIndex = Math.min(2, children.length - 1);
        return { parent: container, afterElement: children[targetIndex] };
      }
      return container;
    }
    return null;
  }
  function ensurePillInserted(card, uiKey, inModal=false) {
    let el = card.querySelector(`.fdrb-pill-column[data-key="${uiKey}"]`);
    if (el) { markPillSeen(el); return el; }
    card.querySelectorAll('.fdrb-pill-column').forEach(x => x.remove());
    const { column } = createPillColumn(uiKey, inModal);
    const ip = findInsertionPoint(card, inModal);
    if (!ip) return null;
    if (ip.afterElement) ip.afterElement.after(column);
    else if (ip.appendChild) ip.appendChild(column);
    else card.appendChild(column);
    cardKey.set(card, uiKey);
    markPillSeen(column);
    return column;
  }

  function queryCards(root=null) {
    const scope = root || document;
    const selectors = BASE_CARD_SELECTORS.join(',');
    const out = [];
    const seen = new Set();
    scope.querySelectorAll(selectors).forEach(el => {
      if (seen.has(el)) return;
      if (!isVisible(el)) return;
      if (!hasTimePattern(el)) return;
      seen.add(el); out.push(el);
    });
    return out;
  }

  function callAPI(payload, requestKey) {
    return new Promise((resolve) => {
      chrome.runtime.sendMessage(
        { type: "FDRB_PREDICT", url: API_URL, payload, request_key: requestKey },
        (resp) => {
          if (!resp || !resp.ok) return resolve({ ok:false, error: resp ? resp.error : "no response" });
          resolve({ ok:true, data: resp.data });
        }
      );
    });
  }

  function scheduleAnnotate(card, fn) {
    clearTimeout(cardDebounce.get(card));
    const t = setTimeout(fn, CARD_DEBOUNCE_MS);
    cardDebounce.set(card, t);
  }

  function extractRoute(card, globalRoute) {
    const t = (card.innerText||"");
    const mR = t.match(/\b([A-Z]{3})\s*[–—-]\s*([A-Z]{3})\b/);
    if (mR && IATA_SET.has(mR[1]) && IATA_SET.has(mR[2])) return { origin: mR[1], dest: mR[2] };
    return globalRoute || null;
  }

  async function annotateCard(card, outboundISO, globalRoute) {
    if (!isVisible(card)) return;
    const depArr = extractDepArrTimes(card);
    if (!depArr) return;
    const { dep, arr, plusDays } = depArr;
    const route = extractRoute(card, globalRoute);
    if (!route) return;

    const text = card.innerText || "";
    const txtU = text.toUpperCase();
    const marketingIata = extractAirlineIata(txtU) || undefined;
    const operatingIata = extractOperatingAirlineIata(txtU) || marketingIata || undefined;
    const flightNumber = extractFlightNumber(text) || null;

    const depIso = `${outboundISO}T${dep}`;
    const arrIso = computeArrivalISO(outboundISO, dep, arr, plusDays);
    const durationMin = parseDurationMinutes(text);
    const stops = extractStops(text);

    const isInModal = inModalNode(card);
    const baseKey = `${route.origin}-${route.dest}-${depIso}-${operatingIata||""}-${flightNumber||""}`;
    const uiKey = isInModal ? `${baseKey}-MODAL` : baseKey;

    const pillHost = ensurePillInserted(card, uiKey, isInModal);
    if (!pillHost) { setTimeout(() => scheduleAnnotate(card, () => annotateCard(card, outboundISO, globalRoute)), 250); return; }

    const cached = pillDataCache.get(baseKey);
    const now = Date.now();
    if (cached && (now - cached.ts) < API_TTL_MS) {
      const v = pillHost.querySelector('.fdrb-pill-value');
      if (v) { updatePillValue(v, cached.p, cached.thr); markPillSeen(pillHost); }
      return;
    }

    const payload = {
      origin_iata: route.origin,
      dest_iata: route.dest,
      departure_iso: depIso,
      arrival_iso: arrIso,
      duration_min: durationMin,
      stops: stops,
      marketing_airline_iata: marketingIata || null,
      operating_airline_iata: operatingIata || null,
      airline_iata: marketingIata || null,
      flight_number: flightNumber,
      scenario: "climo"
    };

    log("features", { baseKey, uiKey, features: payload });

    try {
      const resp = await callAPI(payload, baseKey);
      if (!resp.ok) throw new Error(resp.error || "api error");
      const { probability: p, threshold: thr } = resp.data;
      pillDataCache.set(baseKey, { p, thr, ts: Date.now() });
      const v = pillHost.querySelector('.fdrb-pill-value');
      if (v) { updatePillValue(v, p, thr); markPillSeen(pillHost); }
      seenKeys.add(baseKey);
      log("ok", baseKey, (p*100).toFixed(1), isInModal ? "[modal]" : "[lista]");
    } catch (e) {
      const v = pillHost.querySelector('.fdrb-pill-value');
      if (v) { v.classList.add("error"); v.textContent = "Erro"; v.style.cssText = `font-size:14px!important;font-weight:500!important;line-height:1.2!important;margin:0!important;color:#ea4335!important;`; }
      log("erro API", e);
    }
  }

  function observeCard(card) {
    if (card._fdrbObserved) return;
    const obs = new MutationObserver(() => {
      const outboundISO = guessOutboundISODate();
      const globalRoute = guessRouteFromHeader();
      scheduleAnnotate(card, () => annotateCard(card, outboundISO, globalRoute));
    });
    obs.observe(card, { childList: true, subtree: true });
    card._fdrbObserved = true;
  }

  const io = new IntersectionObserver((entries) => {
    const outboundISO = guessOutboundISODate();
    const globalRoute = guessRouteFromHeader();
    for (const e of entries) {
      if (!e.isIntersecting) continue;
      scheduleAnnotate(e.target, () => annotateCard(e.target, outboundISO, globalRoute));
    }
  }, { root: null, threshold: 0.1 });

  function scheduleScan(fn, ms=SCAN_THROTTLE_MS) {
    if (scanThrottleTimer) return;
    scanThrottleTimer = setTimeout(() => { scanThrottleTimer = null; fn(); }, ms);
  }

  function scanContext(root=null) {
    const cards = queryCards(root);
    cards.forEach(c => { io.observe(c); observeCard(c); });
    const scope = root || document;
    scope.querySelectorAll('.fdrb-pill-column').forEach(pill => {
      if (!pill.isConnected) { pill.remove(); return; }
      if (IS_FLIGHTS_PAGE) {
        const last = pillLastSeen.get(pill) || 0;
        if (performance.now() - last > 2500) { pill.remove(); }
        return;
      }
      const host = pill.closest(BASE_CARD_SELECTORS.join(','));
      if (!host || !host.isConnected) pill.remove();
    });
  }

  function scanAll() {
    if (modalRoot && document.contains(modalRoot)) { scanContext(modalRoot); }
    else { scanContext(document); }
  }

  function startKeepAlive() {
    clearInterval(keepAliveTimer);
    keepAliveTimer = setInterval(() => {
      const outboundISO = guessOutboundISODate();
      const globalRoute = guessRouteFromHeader();
      const scope = (modalRoot && document.contains(modalRoot)) ? modalRoot : document;
      queryCards(scope).forEach(card => {
        if (!isVisible(card)) return;
        const depArr = extractDepArrTimes(card);
        if (!depArr) return;
        const { dep } = depArr;
        const route = extractRoute(card, globalRoute);
        if (!route) return;
        const depIso = `${outboundISO}T${dep}`;
        const txtU = (card.innerText||"").toUpperCase();
        const marketingIata = extractAirlineIata(txtU) || undefined;
        const operatingIata = extractOperatingAirlineIata(txtU) || marketingIata || undefined;
        const flightNumber = extractFlightNumber(card.innerText||"") || null;
        const baseKey = `${route.origin}-${route.dest}-${depIso}-${operatingIata||""}-${flightNumber||""}`;
        const isInModal = inModalNode(card);
        const uiKey = isInModal ? `${baseKey}-MODAL` : baseKey;
        if (!card.querySelector(`.fdrb-pill-column[data-key="${uiKey}"]`)) {
          const host = ensurePillInserted(card, uiKey, isInModal);
          if (!host) return;
          const cached = pillDataCache.get(baseKey);
          if (cached) {
            const v = host.querySelector('.fdrb-pill-value');
            if (v) { updatePillValue(v, cached.p, cached.thr); markPillSeen(host); }
          }
        } else {
          const host = card.querySelector(`.fdrb-pill-column[data-key="${uiKey}"]`);
          if (host) markPillSeen(host);
        }
      });
    }, KEEPALIVE_MS);
  }

  function detectModalRoot() {
    for (const sel of MODAL_ROOT_SELECTORS) {
      const el = document.querySelector(sel);
      if (el && isVisible(el)) return el;
    }
    return null;
  }

  function attachModalObserver(root) {
    detachModalObserver();
    if (!root) return;
    modalRoot = root;
    modalObserver = new MutationObserver(() => { scheduleScan(() => scanContext(modalRoot)); });
    modalObserver.observe(modalRoot, { childList: true, subtree: true });
    log("Modal detectado → observando...");
    scheduleScan(() => scanContext(modalRoot), 0);
  }

  function detachModalObserver() { if (modalObserver) { modalObserver.disconnect(); modalObserver = null; } modalRoot = null; }

  const globalMO = new MutationObserver(() => {
    const dlg = detectModalRoot();
    if (dlg && dlg !== modalRoot) { attachModalObserver(dlg); }
    else if (!dlg && modalRoot) { log("Modal fechado."); detachModalObserver(); setTimeout(() => { scheduleScan(scanAll, 0); }, 250); }
    else { scheduleScan(scanAll); }
  });

  function init() {
    log("Inicializando extensão…");
    globalMO.observe(document.documentElement, { childList: true, subtree: true });
    scheduleScan(scanAll, 0);
    startKeepAlive();
    window.addEventListener("load", () => scheduleScan(scanAll));
    window.addEventListener("popstate", () => scheduleScan(scanAll));
    window.addEventListener("hashchange", () => scheduleScan(scanAll));
    document.addEventListener('visibilitychange', () => { if (!document.hidden) scheduleScan(scanAll); });
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();

  log("Extensão Flight Delay Risk (Flights-safe) pronta.");
})();
