const labelMeta = {
  insufficient_effect_evidence: { name: "Insufficient evidence", color: "#b8c4d1" },
  undefined_geometry: { name: "Undefined geometry", color: "#64748b" },
  directed_ray: { name: "Directed ray", color: "#1677b8" },
  axis_or_antipodal: { name: "Axis / antipodal", color: "#ed9a2e" },
  oneD_diffuse: { name: "1D diffuse", color: "#c9a227" },
  multi_mode_directional_geometry: { name: "Multi-mode directional", color: "#2f9e69" },
  global_2D_directional_subspace: { name: "2D directional subspace", color: "#d55e5e" },
  global_kD_directional_subspace: { name: "kD directional subspace", color: "#8757b5" },
  residual_lowD_k: { name: "Low-D residual", color: "#22a4aa" },
  unresolved_high_dimensional_or_diffuse: { name: "Unresolved high-D / diffuse", color: "#9a6952" },
};

const summaryGroups = [
  { name: "Insufficient evidence", labels: ["insufficient_effect_evidence"], color: "#b8c4d1" },
  { name: "Undefined", labels: ["undefined_geometry"], color: "#64748b" },
  { name: "1D families", labels: ["directed_ray", "axis_or_antipodal", "oneD_diffuse"], color: "#1677b8" },
  { name: "Structured low-D", labels: ["multi_mode_directional_geometry", "global_2D_directional_subspace", "global_kD_directional_subspace", "residual_lowD_k"], color: "#8757b5" },
  { name: "Unresolved high-D / diffuse", labels: ["unresolved_high_dimensional_or_diffuse"], color: "#9a6952" },
];

const displayNumber = value => value == null ? "—" : Number(value).toLocaleString(undefined, { maximumFractionDigits: 3 });
const percent = (value, total) => total ? `${(value / total * 100).toFixed(1)}%` : "0%";
const siteRoot = new URL("../../", new URL(document.body.dataset.atlasUrl, window.location.href));
const state = { data: null, architecture: "topk", label: "all", screenPoints: [], selected: null };

const themeStorageKey = "fega-theme";
const darkThemeQuery = window.matchMedia("(prefers-color-scheme: dark)");
function storedTheme() { try { return window.localStorage.getItem(themeStorageKey); } catch { return null; } }
function applyTheme(theme, persist) {
  document.documentElement.dataset.theme = theme;
  document.documentElement.style.colorScheme = theme;
  document.querySelectorAll("[data-theme-toggle]").forEach(toggle => {
    const label = theme === "dark" ? "Switch to light mode" : "Switch to dark mode";
    toggle.setAttribute("aria-label", label); toggle.setAttribute("title", label); toggle.setAttribute("aria-pressed", String(theme === "dark"));
  });
  if (persist) { try { window.localStorage.setItem(themeStorageKey, theme); } catch {} }
  window.dispatchEvent(new CustomEvent("fega-themechange", { detail: { theme } }));
}
applyTheme(storedTheme() || (darkThemeQuery.matches ? "dark" : "light"), false);
document.querySelectorAll("[data-theme-toggle]").forEach(toggle => toggle.addEventListener("click", () => applyTheme(document.documentElement.dataset.theme === "dark" ? "light" : "dark", true)));
darkThemeQuery.addEventListener("change", event => { if (!storedTheme()) applyTheme(event.matches ? "dark" : "light", false); });

const canvas = document.getElementById("atlas-canvas");
const context = canvas.getContext("2d");
const tooltip = document.getElementById("atlas-tooltip");
const chartRoot = document.getElementById("architecture-summary");
const architectureControls = document.getElementById("architecture-controls");
const labelControls = document.getElementById("label-controls");
const featureTitle = document.getElementById("feature-title");
const featureDescription = document.getElementById("feature-description");
const metricGrid = document.getElementById("metric-grid");
const featureCard = document.getElementById("feature-card");
const featureEmpty = document.getElementById("feature-empty");

function currentArchitecture() {
  return state.data.architectures.find(item => item.id === state.architecture);
}

function friendlyLabel(label) {
  return labelMeta[label]?.name ?? label.replaceAll("_", " ");
}

function labelColor(label) {
  return labelMeta[label]?.color ?? "#91a4b7";
}

function visibleFeatures() {
  const architecture = currentArchitecture();
  return architecture.features.filter(feature => state.label === "all" || feature.label === state.label);
}

function renderSummary() {
  chartRoot.replaceChildren();
  state.data.architectures.forEach(architecture => {
    const card = document.createElement("button");
    card.type = "button";
    card.className = `architecture-card ${architecture.id === state.architecture ? "is-selected" : ""}`;
    card.setAttribute("aria-pressed", String(architecture.id === state.architecture));
    const groups = summaryGroups.map(group => ({ ...group, count: group.labels.reduce((sum, label) => sum + (architecture.labels[label] ?? 0), 0) }));
    card.innerHTML = `<h3>${architecture.name}</h3><p>${displayNumber(architecture.total)} RAVEL City–Country features</p><div class="stacked-bar" aria-label="Geometry label distribution">${groups.map(group => `<span style="width:${group.count / architecture.total * 100}%;background:${group.color}" title="${group.name}: ${group.count}"></span>`).join("")}</div><div class="legend-list">${groups.map(group => `<span class="legend-item"><i class="legend-dot" style="background:${group.color}"></i>${group.name} ${percent(group.count, architecture.total)}</span>`).join("")}</div>`;
    card.addEventListener("click", () => {
      state.architecture = architecture.id;
      state.label = "all";
      state.selected = null;
      renderAll();
      document.getElementById("atlas").scrollIntoView({ behavior: "smooth", block: "start" });
    });
    chartRoot.append(card);
  });
}

  function renderHeroCards() {
    const stream = document.getElementById("hero-card-stream");
    const cardSeeds = [
      ["topk", "directed_ray"],
      ["matryoshka", "global_2D_directional_subspace"],
      ["relu", "residual_lowD_k"],
      ["topk", "multi_mode_directional_geometry"],
      ["relu", "oneD_diffuse"],
      ["matryoshka", "axis_or_antipodal"],
      ["relu", "global_kD_directional_subspace"],
      ["topk", "unresolved_high_dimensional_or_diffuse"],
    ];
    const cardCopy = {
      directed_ray: ["Stable directed ray", "Most directions repeat one effect"],
      global_2D_directional_subspace: ["Directions fill a plane", "Second component is meaningfully used"],
      global_kD_directional_subspace: ["A low-dimensional effect space", "Several directions remain active"],
      residual_lowD_k: ["Stable average direction", "Residual disagreement is low-D"],
      multi_mode_directional_geometry: ["Several directed modes", "Modes explain a weak global ray"],
      oneD_diffuse: ["Continuous one-axis spread", "No clean ray or two-mode split"],
      axis_or_antipodal: ["Shared axis, opposing effects", "Directions repeat along one line"],
      unresolved_high_dimensional_or_diffuse: ["Diffuse effect geometry", "No compact pattern is resolved"],
    };
    const cards = cardSeeds.map(([architectureId, label]) => {
      const architecture = state.data.architectures.find(item => item.id === architectureId);
      return { architecture, label, record: architecture?.featured[label] };
    }).filter(item => item.record?.assets?.sphere && item.record?.assets?.projection);
    const items = cards.map(item => {
      const metrics = item.record.metrics;
      const heroArchitectureName = item.architecture.id === "matryoshka" ? "Matryoshka" : item.architecture.name;
      const [headline, detail] = cardCopy[item.label] ?? [friendlyLabel(item.label), "Representative FEGA result"];
      const metricValues = [
        ["Ray", metrics.c_ray],
        ["Span-2", metrics.span_2],
        ["Residual", metrics.residual_energy],
      ].filter(([, value]) => value !== null && value !== undefined).slice(0, 2);
      return `<figure class="hero-card hero-poster" style="--card-accent:${labelColor(item.label)}">
        <div class="hero-poster-tab">Feature ${metrics.id} <span>|</span> ${heroArchitectureName}</div>
        <header class="hero-poster-copy"><strong>${headline}</strong><span>${detail}</span></header>
        <div class="hero-poster-views">
          <section><span>Normalized directions</span><img src="${new URL(item.record.assets.sphere, siteRoot).href}" alt="${item.architecture.name} feature ${metrics.id} normalized directions"></section>
          <section><span>Ray + residual view</span><img src="${new URL(item.record.assets.projection, siteRoot).href}" alt="${item.architecture.name} feature ${metrics.id} projection view"></section>
        </div>
        <footer class="hero-poster-metrics">${metricValues.map(([name, value]) => `<span class="hero-poster-metric"><small>${name}</small><strong>${displayNumber(value)}</strong></span>`).join("")}<span class="hero-poster-metric"><small>Contexts</small><strong>${displayNumber(metrics.n)}</strong></span></footer>
        <figcaption>${friendlyLabel(item.label)} · ${item.architecture.name} · feature ${metrics.id}</figcaption>
      </figure>`;
    });
    const lanes = [
      items,
      [...items.slice(3), ...items.slice(0, 3)],
      [...items.slice(6), ...items.slice(0, 6)],
    ];
    stream.innerHTML = items.length ? `<div class="hero-card-lanes">${lanes.map((lane, index) => `<div class="hero-card-lane hero-card-lane-${index + 1}">${[...lane, ...lane].join("")}</div>`).join("")}</div>` : "<span>Rendered feature cards are unavailable for this export.</span>";
  }

function renderControls() {
  architectureControls.replaceChildren();
  state.data.architectures.forEach(architecture => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `control-button ${architecture.id === state.architecture ? "is-selected" : ""}`;
    button.textContent = architecture.name;
    button.addEventListener("click", () => { state.architecture = architecture.id; state.label = "all"; state.selected = null; renderAll(); });
    architectureControls.append(button);
  });
  labelControls.replaceChildren();
  const architecture = currentArchitecture();
  const options = ["all", ...Object.keys(labelMeta).filter(label => architecture.labels[label])];
  options.forEach(label => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `control-button label-button ${label === state.label ? "is-selected" : ""}`;
    const count = label === "all" ? architecture.total : architecture.labels[label];
    button.innerHTML = `<span class="label-name">${label === "all" ? "All labels" : `<i class="label-swatch" style="background:${labelColor(label)}"></i>${friendlyLabel(label)}`}</span><small>${displayNumber(count)}</small>`;
    button.addEventListener("click", () => { state.label = label; state.selected = null; renderAll(); });
    labelControls.append(button);
  });
}

function renderMetricGrid(metrics) {
  const values = [
    ["Feature", metrics.id], ["Valid contexts", metrics.n], ["c_ray", metrics.c_ray],
    ["Span-2", metrics.span_2], ["Residual energy", metrics.residual_energy], ["Evidence", metrics.evidence?.replaceAll("_", " ")],
  ];
  metricGrid.innerHTML = values.map(([name, value]) => `<div class="metric"><span>${name}</span><strong>${typeof value === "number" ? displayNumber(value) : value ?? "—"}</strong></div>`).join("");
}

function showFeatured() {
  const architecture = currentArchitecture();
  const preferred = state.label === "all" ? (architecture.featured.directed_ray ?? Object.values(architecture.featured)[0]) : architecture.featured[state.label];
  if (!preferred) {
    featureTitle.textContent = state.label === "all" ? "Choose a label" : friendlyLabel(state.label);
    featureDescription.textContent = "This selected category has no curated representative card in the exported page assets. Atlas points still expose their recorded metadata.";
    metricGrid.innerHTML = "";
    featureCard.hidden = true;
    featureEmpty.hidden = false;
    return;
  }
  const { metrics, assets } = preferred;
  featureTitle.textContent = `${friendlyLabel(metrics.label)} · feature ${metrics.id}`;
  featureDescription.textContent = "Representative rendered candidate from this result run. Metrics are copied from its FEGA feature record.";
  renderMetricGrid(metrics);
  if (assets.card) {
    featureCard.src = new URL(assets.card, siteRoot).href;
    featureCard.hidden = false;
    featureEmpty.hidden = true;
  } else {
    featureCard.hidden = true;
    featureEmpty.hidden = false;
  }
}

function showSelected(feature) {
  state.selected = feature;
  featureTitle.textContent = `${friendlyLabel(feature.label)} · feature ${feature.id}`;
    featureDescription.textContent = `Selected directly from the ${currentArchitecture().name} UMAP map. This point has recorded result metadata; a rendered card is available only for curated representatives.`;
  renderMetricGrid({ id: feature.id, n: feature.n, c_ray: null, span_2: null, residual_energy: null, evidence: feature.evidence });
  featureCard.hidden = true;
  featureEmpty.hidden = false;
}

function resizeCanvas() {
  const rect = canvas.getBoundingClientRect();
  const scale = window.devicePixelRatio || 1;
  canvas.width = Math.round(rect.width * scale);
  canvas.height = Math.round(rect.height * scale);
  context.setTransform(scale, 0, 0, scale, 0, 0);
  return rect;
}

function drawAtlas() {
  const rect = resizeCanvas();
  const features = visibleFeatures().filter(feature => Number.isFinite(feature.x) && Number.isFinite(feature.y));
  const width = rect.width;
  const height = rect.height;
  const dark = document.documentElement.dataset.theme === "dark";
  context.clearRect(0, 0, width, height);
  context.fillStyle = dark ? "#102037" : "#fbfdff";
  context.fillRect(0, 0, width, height);
  const padding = { left: 38, right: 16, top: 18, bottom: 30 };
  const xs = features.map(feature => feature.x);
  const ys = features.map(feature => feature.y);
  const xMin = Math.min(...xs), xMax = Math.max(...xs), yMin = Math.min(...ys), yMax = Math.max(...ys);
  const xSpan = xMax - xMin || 1, ySpan = yMax - yMin || 1;
  context.strokeStyle = dark ? "#29445f" : "#e4ebf2";
  context.lineWidth = 1;
  for (let index = 0; index < 5; index += 1) {
    const x = padding.left + index / 4 * (width - padding.left - padding.right);
    const y = padding.top + index / 4 * (height - padding.top - padding.bottom);
    context.beginPath(); context.moveTo(x, padding.top); context.lineTo(x, height - padding.bottom); context.stroke();
    context.beginPath(); context.moveTo(padding.left, y); context.lineTo(width - padding.right, y); context.stroke();
  }
  context.fillStyle = dark ? "#8ea7bf" : "#91a1b2";
  context.font = "10px Inter";
  context.fillText("UMAP 2", 5, 14);
  context.fillText("UMAP 1", width - 48, height - 9);
  state.screenPoints = features.map(feature => ({
    feature,
    x: padding.left + ((feature.x - xMin) / xSpan) * (width - padding.left - padding.right),
    y: height - padding.bottom - ((feature.y - yMin) / ySpan) * (height - padding.top - padding.bottom),
  }));
  state.screenPoints.forEach(point => {
    context.beginPath();
    context.fillStyle = labelColor(point.feature.label);
    context.globalAlpha = state.label === "all" ? 0.58 : 0.83;
    context.arc(point.x, point.y, state.selected?.id === point.feature.id ? 4.5 : 2.3, 0, Math.PI * 2);
    context.fill();
  });
  context.globalAlpha = 1;
  if (state.selected) {
    const active = state.screenPoints.find(point => point.feature.id === state.selected.id);
    if (active) { context.beginPath(); context.strokeStyle = "#0f2138"; context.lineWidth = 1.5; context.arc(active.x, active.y, 6.5, 0, Math.PI * 2); context.stroke(); }
  }
  document.getElementById("atlas-count").textContent = displayNumber(features.length);
  document.getElementById("atlas-count-label").textContent = state.label === "all" ? "features shown" : `${friendlyLabel(state.label).toLowerCase()} features shown`;
  document.getElementById("plot-title").textContent = `${currentArchitecture().name} · ${state.label === "all" ? "all geometry labels" : friendlyLabel(state.label)}`;
}

function nearestFeature(event) {
  const rect = canvas.getBoundingClientRect();
  const x = event.clientX - rect.left, y = event.clientY - rect.top;
  let nearest = null, distance = 12;
  state.screenPoints.forEach(point => { const currentDistance = Math.hypot(point.x - x, point.y - y); if (currentDistance < distance) { nearest = point; distance = currentDistance; } });
  return nearest;
}

canvas.addEventListener("mousemove", event => {
  const nearest = nearestFeature(event);
  if (!nearest) { tooltip.hidden = true; return; }
  const container = canvas.parentElement.getBoundingClientRect();
  tooltip.hidden = false;
  tooltip.style.left = `${Math.min(event.clientX - container.left + 12, container.width - 180)}px`;
  tooltip.style.top = `${Math.max(event.clientY - container.top - 60, 26)}px`;
  tooltip.innerHTML = `<strong>feature ${nearest.feature.id}</strong><br>${friendlyLabel(nearest.feature.label)}<br>n = ${displayNumber(nearest.feature.n)}`;
});
canvas.addEventListener("mouseleave", () => { tooltip.hidden = true; });
canvas.addEventListener("click", event => { const nearest = nearestFeature(event); if (nearest) { showSelected(nearest.feature); drawAtlas(); } });

function renderAll() {
  renderHeroCards();
  renderSummary();
  renderControls();
  showFeatured();
  drawAtlas();
}

async function initializeAtlas() {
  try {
    const response = await fetch(document.body.dataset.atlasUrl);
    if (!response.ok) throw new Error(`Could not load results (${response.status})`);
    state.data = await response.json();
    renderAll();
  } catch (error) {
    chartRoot.innerHTML = `<p class="loading-message">The exported RAVEL data could not be loaded: ${error.message}</p>`;
    document.getElementById("plot-title").textContent = "Results unavailable";
  }
}

document.getElementById("copy-bibtex").addEventListener("click", async event => {
  const button = event.currentTarget;
  try { await navigator.clipboard.writeText(document.getElementById("bibtex").textContent); button.innerHTML = '<i class="fas fa-check"></i><span>Copied</span>'; setTimeout(() => button.innerHTML = '<i class="far fa-copy"></i><span>Copy</span>', 1600); }
  catch { button.querySelector("span").textContent = "Select text"; }
});
const scrollTop = document.getElementById("scroll-top");
window.addEventListener("scroll", () => scrollTop.classList.toggle("visible", window.scrollY > 400));
window.addEventListener("resize", () => { if (state.data) drawAtlas(); });
window.addEventListener("fega-themechange", () => { if (state.data) drawAtlas(); });
scrollTop.addEventListener("click", () => window.scrollTo({ top: 0, behavior: "smooth" }));
initializeAtlas();
