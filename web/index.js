// ==========================
// Partículas
// ==========================
(function createParticles() {
  const c = document.getElementById("particles");
  for (let i = 0; i < 50; i++) {
    const p = document.createElement("div");
    p.className = "particle";
    p.style.left = Math.random() * 100 + "%";
    p.style.animationDelay = Math.random() * 10 + "s";
    p.style.animationDuration = 10 + Math.random() * 10 + "s";
    c.appendChild(p);
  }
})();

// ==========================
// API URL
// ==========================
let API_URL = "http://127.0.0.1:8000/predict";
const apiUrlInput = document.getElementById("apiUrlInput");
if (apiUrlInput) {
  apiUrlInput.value = API_URL;
  apiUrlInput.addEventListener("change", () => {
    API_URL = apiUrlInput.value.trim() || API_URL;
  });
}

// ==========================
// Modo Básico/Avançado
// ==========================
const modeBtns = document.querySelectorAll(".mode-btn");
const advancedSection = document.getElementById("advancedSection");
let currentMode = "basic";

modeBtns.forEach((btn) => {
  btn.addEventListener("click", () => {
    modeBtns.forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");
    currentMode = btn.dataset.mode;
    advancedSection.classList.toggle("show", currentMode === "advanced");
  });
});

// ==========================
// Aeroportos BR
// ==========================
const AIRPORTS = {
  GRU: { name: "São Paulo/Guarulhos", lat: -23.4356, lon: -46.4731 },
  CGH: { name: "São Paulo/Congonhas", lat: -23.6261, lon: -46.6566 },
  VCP: { name: "Campinas/Viracopos", lat: -23.0074, lon: -47.1344 },
  GIG: { name: "Rio de Janeiro/Galeão", lat: -22.8089, lon: -43.2436 },
  SDU: { name: "Rio de Janeiro/Santos Dumont", lat: -22.9105, lon: -43.1631 },
  BSB: { name: "Brasília", lat: -15.8711, lon: -47.9186 },
  // ... restante da lista original ...
  BVB: { name: "Boa Vista", lat: 2.8461, lon: -60.6901 },
};

const origemSel = document.getElementById("origem");
const destinoSel = document.getElementById("destino");

function fillAirportSelects() {
  const opts = ['<option value="">Selecione o aeroporto</option>'];
  Object.entries(AIRPORTS).forEach(([iata, info]) =>
    opts.push(`<option value="${iata}">${iata} - ${info.name}</option>`)
  );
  origemSel.innerHTML = opts.join("");
  destinoSel.innerHTML = opts.join("");
}
fillAirportSelects();
// ==========================
// Mapa
// ==========================
let map, markers = {}, activePick = "origin";

function initMap() {
  // limites máximos do mundo
  const worldBounds = [
    [-90, -180], // canto sudoeste
    [90, 180]    // canto nordeste
  ];

  map = L.map("map", {
    zoomControl: true,
    scrollWheelZoom: true,   // mantém o scroll se você quiser
    minZoom: 2,              // impede de afastar demais
    maxBounds: worldBounds,  // trava dentro do globo
    maxBoundsViscosity: 1.0  // 1.0 = totalmente rígido
  }).setView([-14.2350, -51.9253], 4);

  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    minZoom: 2,
    maxZoom: 19,
    attribution: "&copy; OpenStreetMap",
    noWrap: true             // impede repetição horizontal
  }).addTo(map);

  Object.entries(AIRPORTS).forEach(([iata, info]) => {
    const m = L.marker([info.lat, info.lon]).addTo(map);
    m.bindTooltip(`${iata} - ${info.name}`);
    m.on("click", () => {
      if (activePick === "origin") origemSel.value = iata;
      else destinoSel.value = iata;
      highlightSelection();
    });
    markers[iata] = m;
  });
}
initMap();

function highlightSelection() {
  const o = origemSel.value, d = destinoSel.value;
  Object.entries(markers).forEach(([iata, m]) =>
    m.setOpacity(iata === o || iata === d ? 1 : 0.5)
  );

  if (window.routeLine) {
    map.removeLayer(window.routeLine);
    window.routeLine = null;
  }
  if (o && d && AIRPORTS[o] && AIRPORTS[d]) {
    const latlngs = [
      [AIRPORTS[o].lat, AIRPORTS[o].lon],
      [AIRPORTS[d].lat, AIRPORTS[d].lon],
    ];
    window.routeLine = L.polyline(latlngs, { color: "#7c3aed", weight: 3 }).addTo(map);
    map.fitBounds(window.routeLine.getBounds().pad(0.25));
  }
}

// Seleção origem/destino
const selOriginChip = document.getElementById("selOrigin");
const selDestChip   = document.getElementById("selDest");
const clearSelChip  = document.getElementById("clearSel");

selOriginChip.addEventListener("click", () => {
  activePick = "origin";
  selOriginChip.classList.add("active");
  selDestChip.classList.remove("active");
});
selDestChip.addEventListener("click", () => {
  activePick = "dest";
  selDestChip.classList.add("active");
  selOriginChip.classList.remove("active");
});
clearSelChip.addEventListener("click", () => {
  origemSel.value = "";
  destinoSel.value = "";
  if (window.routeLine) {
    map.removeLayer(window.routeLine);
    window.routeLine = null;
  }
  Object.values(markers).forEach((m) => m.setOpacity(1));
});

origemSel.addEventListener("change", highlightSelection);
destinoSel.addEventListener("change", highlightSelection);

// ==========================
// Helpers
// ==========================
function haversineKm(lat1, lon1, lat2, lon2) {
  const R = 6371, toRad = x => x * Math.PI / 180;
  const dLat = toRad(lat2 - lat1);
  const dLon = toRad(lon2 - lon1);
  const a = Math.sin(dLat / 2) ** 2 +
            Math.cos(toRad(lat1)) * Math.cos(toRad(lat2)) *
            Math.sin(dLon / 2) ** 2;
  return 2 * R * Math.asin(Math.sqrt(a));
}

// ==========================
// Cenário Avançado
// ==========================
const scenarioSelect = document.getElementById("scenarioSelect");
const manualExtras = document.getElementById("manualExtras");
scenarioSelect.addEventListener("change", () => {
  manualExtras.style.display = scenarioSelect.value === "manual" ? "block" : "none";
  if (scenarioSelect.value !== "manual") {
    [
      "ov_chuva_dep", "ov_bad_dep", "ov_chuva_arr", "ov_bad_arr",
      "ov_hist_empresa", "ov_hist_rota", "ov_hist_num_voo",
      "ov_hist_origem_hora", "ov_hist_std_rota"
    ].forEach(id => { const el = document.getElementById(id); if (el) el.value = ""; });
  }
});

const CLIMA_MAP = {
  0: { p_rain: 0.00, p_bad: 0.00 },
  1: { p_rain: 0.08, p_bad: 0.10 },
  2: { p_rain: 0.15, p_bad: 0.20 },
  3: { p_rain: 0.35, p_bad: 0.45 },
  4: { p_rain: 0.65, p_bad: 0.80 }
};

// ==========================
// UI e Análise
// ==========================
const analyzeBtn = document.getElementById("analyzeBtn");
const btnText = document.getElementById("btnText");
const resultsContainer = document.getElementById("resultsContainer");

function showNotification(message, type = "success") {
  const el = document.getElementById("notification");
  el.textContent = message;
  const colors = {
    error: "linear-gradient(135deg,#ef4444,#dc2626)",
    warning: "linear-gradient(135deg,#f59e0b,#d97706)",
    success: "linear-gradient(135deg,#10b981,#059669)",
  };
  el.style.background = colors[type] || colors.success;
  el.classList.add("show");
  setTimeout(() => el.classList.remove("show"), 3000);
}

function displayResults(data, origem, destino, partida, airline) {
  const probability = typeof data.probability === "number" ? data.probability : 0;
  const percentage = Math.round(probability * 100);

  resultsContainer.classList.add("show");
  const riskMeterFill = document.getElementById("riskMeterFill");
  const riskPercentage = document.getElementById("riskPercentage");
  const riskLabel = document.getElementById("riskLabel");

  let color, label;
  if (percentage < 30) { color = "#10b981"; label = "Baixo Risco"; }
  else if (percentage < 70) { color = "#f59e0b"; label = "Risco Moderado"; }
  else { color = "#ef4444"; label = "Alto Risco"; }
  riskMeterFill.style.stroke = color;

  let current = 0, inc = Math.max(1, Math.round(percentage / 50));
  const timer = setInterval(() => {
    current += inc;
    if (current >= percentage) { current = percentage; clearInterval(timer); }
    riskPercentage.textContent = current + "%";
    const offset = 754 - (754 * current / 100);
    riskMeterFill.style.strokeDashoffset = offset;
  }, 20);

  document.getElementById("routeStat").textContent = `${origem} → ${destino}`;
  const d = new Date(partida);
  document.getElementById("timeStat").textContent =
    d.toLocaleString("pt-BR", { hour: "2-digit", minute: "2-digit", day: "2-digit", month: "2-digit" });

  const airlineNames = { G3: "GOL", LA: "LATAM", AD: "Azul", "2Z": "Voepass" };
  document.getElementById("airlineStat").textContent = airlineNames[airline] || airline;
  document.getElementById("recommendationStat").textContent =
    percentage < 30 ? "✅ Voo pontual" :
    percentage < 70 ? "⚠️ Chegue cedo" : "🔴 Considere alternativas";

  resultsContainer.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

analyzeBtn.addEventListener("click", async () => {
  const origem = origemSel.value;
  const destino = destinoSel.value;
  const partida = document.getElementById("partida").value;
  const airline = document.getElementById("airline").value;

  if (!origem || !destino || !partida || !airline) {
    showNotification("⚠️ Preencha origem, destino, data/hora e companhia.", "warning");
    return;
  }

  analyzeBtn.classList.add("loading");
  btnText.innerHTML = 'Analisando com IA... <span class="loading-ring" style="display:inline-block;margin-left:10px;"></span>';

  const departureISO = new Date(partida).toISOString();
  const duracao = document.getElementById("duracao").value;
  let arrivalISO = null;
  if (duracao) {
    const t = new Date(partida);
    t.setMinutes(t.getMinutes() + parseInt(duracao, 10));
    arrivalISO = t.toISOString();
  }

  const scenario = currentMode === "basic" ? "climo" : (scenarioSelect.value || "climo");
  const payload = {
    origin_iata: origem,
    dest_iata: destino,
    departure_iso: departureISO,
    arrival_iso: arrivalISO,
    duration_min: duracao ? parseFloat(duracao) : null,
    stops: parseInt(document.getElementById("paradas").value, 10),
    operating_airline_iata: airline,
    scenario
  };

  // Distância aproximada
  if (AIRPORTS[origem] && AIRPORTS[destino]) {
    const dk = haversineKm(
      AIRPORTS[origem].lat, AIRPORTS[origem].lon,
      AIRPORTS[destino].lat, AIRPORTS[destino].lon
    );
    const ovDist = document.getElementById("ov_dist_km");
    if (ovDist && !ovDist.value) ovDist.value = dk.toFixed(1);
  }

  try {
    const response = await fetch(API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const data = await response.json();
    displayResults(data, origem, destino, partida, airline);
    showNotification("✨ Análise concluída com sucesso!", "success");
  } catch (err) {
    console.error("[FD] API error:", err);
    showNotification("❌ Erro ao conectar com a API. Verifique se está rodando e a porta/CORS.", "error");
  } finally {
    analyzeBtn.classList.remove("loading");
    btnText.innerHTML = "🚀 Analisar com IA";
  }
});

// ==========================
// Inicialização
// ==========================
window.addEventListener("load", () => {
  const t = new Date();
  t.setDate(t.getDate() + 1);
  t.setHours(10, 0, 0, 0);
  document.getElementById("partida").value =
    new Date(t.getTime() - t.getTimezoneOffset() * 60000).toISOString().slice(0, 16);

  document.querySelector(".scroll-indicator").addEventListener("click", () => {
    document.querySelector(".container").scrollIntoView({ behavior: "smooth" });
  });
});
