const labelMeta = {
  directed_ray: {
    name: "Directed ray",
    color: "#1f77b4",
    description: "The normalized removal effects repeatedly point in one shared direction.",
  },
  axis_or_antipodal: {
    name: "Axis / antipodal",
    color: "#ff7f0e",
    description: "The effects share one unsigned axis, with retained contexts on both sides.",
  },
  multi_mode_directional_geometry: {
    name: "Multi-mode directional",
    color: "#2ca02c",
    description: "The effect cloud separates into several coherent directional modes.",
  },
  global_2D_directional_subspace: {
    name: "2D directional subspace",
    color: "#d62728",
    description: "The normalized effects occupy one shared, meaningfully used plane.",
  },
  global_kD_directional_subspace: {
    name: "kD directional subspace",
    color: "#9467bd",
    description: "The normalized effects occupy one shared low-dimensional subspace.",
  },
  residual_lowD_k: {
    name: "Low-D residual",
    color: "#17becf",
    description: "A mean direction remains, with context-dependent deviations in a small subspace.",
  },
  oneD_diffuse: {
    name: "1D diffuse",
    color: "#bcbd22",
    description: "One-dimensional span evidence remains, without enough signed agreement for a ray or balanced sign split for an axis.",
  },
  unresolved_high_dimensional_or_diffuse: {
    name: "Unresolved high-D / diffuse",
    color: "#8c564b",
    description: "No strict family is supported; the cloud retains high-dimensional, diffuse, or long-tail evidence.",
  },
};

const displayNumber = value => value == null ? "—" : Number(value).toLocaleString(undefined, { maximumFractionDigits: 3 });
function normalizedPercentages(groups, total) {
  if (!total) return groups.map(() => "0.0%");
  const exactTenths = groups.map(group => group.count / total * 1000);
  const tenths = exactTenths.map(Math.floor);
  let remaining = 1000 - tenths.reduce((sum, value) => sum + value, 0);
  const priority = exactTenths
    .map((value, index) => ({ index, remainder: value - tenths[index] }))
    .sort((left, right) => right.remainder - left.remainder);
  for (let index = 0; index < remaining; index += 1) tenths[priority[index % priority.length].index] += 1;
  return tenths.map(value => `${(value / 10).toFixed(1)}%`);
}
const siteRoot = new URL("../../", new URL(document.body.dataset.valueAtlasUrl, window.location.href));
const state = {
  data: null,
  datasets: { value: null, pointer: null },
  mode: "value",
  task: "prontoqa",
  architecture: "topk",
  label: "all",
  screenPoints: [],
  selected: null,
};

const themeStorageKey = "fega-theme";
const atlasReducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
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
applyTheme(storedTheme() || "light", false);
document.querySelectorAll("[data-theme-toggle]").forEach(toggle => toggle.addEventListener("click", () => applyTheme(document.documentElement.dataset.theme === "dark" ? "light" : "dark", true)));

const canvas = document.getElementById("atlas-canvas");
const context = canvas.getContext("2d");
const tooltip = document.getElementById("atlas-tooltip");
const chartRoot = document.getElementById("architecture-summary");
const architectureControls = document.getElementById("architecture-controls");
const labelControls = document.getElementById("label-controls");
const featureTitle = document.getElementById("feature-title");
const featureDescription = document.getElementById("feature-description");
const metricGrid = document.getElementById("metric-grid");
const featureVisuals = document.getElementById("feature-visuals");
const featureSphere = document.getElementById("feature-sphere");
const featureProjection = document.getElementById("feature-projection");
const featureEmpty = document.getElementById("feature-empty");
const taskFilter = document.getElementById("task-filter");
const taskControls = document.getElementById("task-controls");
const featureModeButtons = [...document.querySelectorAll("[data-feature-mode]")];
const findingsEyebrow = document.getElementById("findings-eyebrow");
const findingsTitle = document.getElementById("findings-title");
const findingsLede = document.getElementById("findings-lede");
const atlasEyebrow = document.getElementById("atlas-eyebrow");
const atlasLede = document.getElementById("atlas-lede");

function currentTask() {
  return state.datasets.pointer?.tasks.find(task => task.id === state.task);
}

function selectActiveData() {
  state.data = state.mode === "value" ? state.datasets.value : currentTask();
  if (!state.data?.architectures.some(item => item.id === state.architecture)) {
    state.architecture = state.data?.architectures[0]?.id ?? "topk";
  }
}

function currentArchitecture() {
  return state.data.architectures.find(item => item.id === state.architecture);
}

function displayedLabelGroups(architecture) {
  return Object.entries(labelMeta)
    .map(([label, meta]) => ({ label, ...meta, count: architecture.labels[label] ?? 0 }))
    .filter(group => group.count > 0);
}

function displayedTotal(architecture) {
  return displayedLabelGroups(architecture).reduce((sum, group) => sum + group.count, 0);
}

function friendlyLabel(label) {
  return labelMeta[label]?.name ?? label.replaceAll("_", " ");
}

function labelColor(label) {
  return labelMeta[label]?.color ?? "#91a4b7";
}

function labelDescription(label) {
  return labelMeta[label]?.description ?? "Recorded FEGA geometry for this feature.";
}

function renderedRecords(architecture = currentArchitecture()) {
  if (architecture?.rendered) return Object.values(architecture.rendered).flat();
  return Object.values(architecture?.featured ?? {});
}

function visibleFeatures() {
  const architecture = currentArchitecture();
  return architecture.features.filter(feature => labelMeta[feature.label] && (state.label === "all" || feature.label === state.label));
}

function renderSummary() {
  chartRoot.replaceChildren();
  state.data.architectures.forEach(architecture => {
    const card = document.createElement("button");
    card.type = "button";
    card.className = `architecture-card ${architecture.id === state.architecture ? "is-selected" : ""}`;
    card.setAttribute("aria-pressed", String(architecture.id === state.architecture));
    const groups = displayedLabelGroups(architecture);
    const total = displayedTotal(architecture);
    const percentages = normalizedPercentages(groups, total);
    const distribution = groups.length
      ? `<div class="stacked-bar" aria-label="Geometry label distribution">${groups.map(group => `<span style="width:${group.count / total * 100}%;background:${group.color}" title="${group.name}: ${group.count}"></span>`).join("")}</div><div class="legend-list">${groups.map((group, index) => `<span class="legend-item"><i class="legend-dot" style="background:${group.color}"></i><span class="legend-name">${group.name}</span><span class="legend-value"><b>${displayNumber(group.count)}</b><small>${percentages[index]}</small></span></span>`).join("")}</div>`
      : `<p class="architecture-empty">No reported geometry families in this slice.</p>`;
    card.innerHTML = `<h3>${architecture.name}</h3><p>${displayNumber(total)} mapped ${architecture.dataset} features</p>${distribution}`;
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
      directed_ray: ["Directed ray", "One shared removal direction"],
      global_2D_directional_subspace: ["2D directional subspace", "One shared, meaningfully used plane"],
      global_kD_directional_subspace: ["kD directional subspace", "One shared low-dimensional span"],
      residual_lowD_k: ["Low-D residual", "Low-dimensional variation around the mean"],
      multi_mode_directional_geometry: ["Multi-mode directional", "Several coherent directional modes"],
      oneD_diffuse: ["1D diffuse evidence", "One-dimensional span without a ray or axis"],
      axis_or_antipodal: ["Axis / antipodal", "One unsigned axis with both signs"],
      unresolved_high_dimensional_or_diffuse: ["Unresolved high-D / diffuse", "High-dimensional, diffuse, or long-tail evidence"],
    };
    const cards = cardSeeds.map(([architectureId, label]) => {
      const architecture = state.datasets.value.architectures.find(item => item.id === architectureId);
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
          <section><span>2D effect view</span><img src="${new URL(item.record.assets.projection, siteRoot).href}" alt="${item.architecture.name} feature ${metrics.id} projection view"></section>
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
  taskControls.replaceChildren();
  taskFilter.hidden = state.mode !== "pointer";
  if (state.mode === "pointer") {
    state.datasets.pointer.tasks.forEach(task => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = `control-button ${task.id === state.task ? "is-selected" : ""}`;
      button.textContent = task.name;
      button.addEventListener("click", () => {
        state.task = task.id;
        state.label = "all";
        state.selected = null;
        selectActiveData();
        updateDatasetCopy();
        renderAll();
      });
      taskControls.append(button);
    });
  }
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
    const count = label === "all" ? displayedTotal(architecture) : architecture.labels[label];
    button.innerHTML = `<span class="label-name">${label === "all" ? "All labels" : `<i class="label-swatch" style="background:${labelColor(label)}"></i>${friendlyLabel(label)}`}</span><small>${displayNumber(count)}</small>`;
    button.addEventListener("click", () => { state.label = label; state.selected = null; renderAll(); });
    labelControls.append(button);
  });
}

function renderMetricGrid(metrics) {
  const values = [
    ["Valid contexts", metrics.n],
        ["Median magnitude", metrics.magnitude],
    ["R²", metrics.r2],
    ["C_ray", metrics.c_ray],
    ["Span-2", metrics.span_2],
    ["Residual energy", metrics.residual_energy],
    ["Selected k", metrics.selected_k],
  ].filter(([, value]) => value !== null && value !== undefined);
  metricGrid.innerHTML = values.map(([name, value]) => `<div class="metric"><span>${name}</span><strong>${typeof value === "number" ? displayNumber(value) : value ?? "—"}</strong></div>`).join("");
}

function setFeatureHeading(label, featureId) {
  featureTitle.innerHTML = `<span class="feature-label-line">${friendlyLabel(label)}</span><span class="feature-id-line">feature ${featureId}</span>`;
}

function representativeForFeature(feature) {
  return renderedRecords().find(record => record.metrics?.id === feature.id);
}

function showFeatureVisuals(record) {
  const spherePath = record?.assets?.sphere;
  const projectionPath = record?.assets?.projection;
  if (spherePath || projectionPath) {
    featureSphere.hidden = !spherePath;
    featureProjection.hidden = !projectionPath;
    if (spherePath) featureSphere.src = new URL(spherePath, siteRoot).href;
    if (projectionPath) featureProjection.src = new URL(projectionPath, siteRoot).href;
    featureVisuals.hidden = false;
    featureEmpty.hidden = true;
  } else {
    featureVisuals.hidden = true;
    featureSphere.hidden = true;
    featureProjection.hidden = true;
    featureEmpty.textContent = "Visual preview not included for this feature; the complete recorded metrics are shown above.";
    featureEmpty.hidden = false;
  }
}

function showFeatured() {
  const architecture = currentArchitecture();
  const renderedForLabel = label => architecture.rendered?.[label] ?? (architecture.featured?.[label] ? [architecture.featured[label]] : []);
  const preferred = state.label === "all"
    ? (renderedForLabel("directed_ray")[0] ?? renderedRecords(architecture)[0])
    : renderedForLabel(state.label)[0];
  if (!preferred) {
    featureTitle.textContent = state.label === "all" ? "Choose a label" : friendlyLabel(state.label);
    featureDescription.textContent = state.label === "all"
      ? "Select any point to inspect its recorded geometry metrics."
      : labelDescription(state.label);
    metricGrid.innerHTML = "";
    featureVisuals.hidden = true;
    featureEmpty.textContent = "Select any point to inspect its recorded geometry metrics.";
    featureEmpty.hidden = false;
    return;
  }
  const { metrics, assets } = preferred;
  setFeatureHeading(metrics.label, metrics.id);
  featureDescription.textContent = labelDescription(metrics.label);
  renderMetricGrid(metrics);
  showFeatureVisuals({ assets });
}

function showSelected(feature) {
  state.selected = feature;
  setFeatureHeading(feature.label, feature.id);
  featureDescription.textContent = labelDescription(feature.label);
  renderMetricGrid(feature);
  showFeatureVisuals(representativeForFeature(feature));
}

function resizeCanvas() {
  const rect = canvas.getBoundingClientRect();
  const scale = window.devicePixelRatio || 1;
  const targetWidth = Math.round(rect.width * scale);
  const targetHeight = Math.round(rect.height * scale);
  if (canvas.width !== targetWidth || canvas.height !== targetHeight) {
    canvas.width = targetWidth;
    canvas.height = targetHeight;
  }
  context.setTransform(scale, 0, 0, scale, 0, 0);
  return rect;
}

function drawAtlas(timestamp = performance.now()) {
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
  const pointRadius = state.label !== "all" || features.length <= 40 ? 5 : 2.3;
  state.screenPoints.forEach(point => {
    context.beginPath();
    context.fillStyle = labelColor(point.feature.label);
    context.globalAlpha = state.label === "all" ? 0.58 : 0.83;
    context.arc(point.x, point.y, state.selected?.id === point.feature.id ? Math.max(6, pointRadius + 1) : pointRadius, 0, Math.PI * 2);
    context.fill();
  });
  context.globalAlpha = 1;
  const representativeIds = new Set(
    renderedRecords()
      .filter(record => record.assets?.sphere || record.assets?.projection)
      .map(record => record.metrics?.id),
  );
  const pulse = atlasReducedMotion.matches
    ? 0.35
    : 0.5 - 0.5 * Math.cos((timestamp % 2400) / 2400 * Math.PI * 2);
  state.screenPoints
    .filter(point => representativeIds.has(point.feature.id))
    .forEach(point => {
      context.save();
      context.beginPath();
      context.globalAlpha = 0.24 + 0.38 * pulse;
      context.strokeStyle = "#ff5965";
      context.lineWidth = 1 + 1.1 * pulse;
      context.shadowColor = "rgb(239 51 64 / 32%)";
      context.shadowBlur = 3 + 5 * pulse;
      context.arc(point.x, point.y, pointRadius + 3.5 + 1.8 * pulse, 0, Math.PI * 2);
      context.stroke();
      context.restore();
    });
  context.globalAlpha = 1;
  if (state.selected) {
    const active = state.screenPoints.find(point => point.feature.id === state.selected.id);
    if (active) { context.beginPath(); context.strokeStyle = dark ? "#e6f0fa" : "#0f2138"; context.lineWidth = 1.5; context.arc(active.x, active.y, 6.5, 0, Math.PI * 2); context.stroke(); }
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
  renderSummary();
  renderControls();
  showFeatured();
  drawAtlas();
}

function updateDatasetCopy() {
  const pointer = state.mode === "pointer";
  const taskName = currentTask()?.name ?? "Pointer-like task";
  const pointerTaskCopy = {
    lsc: "Literal Sequence Copying asks the model to find a previous pattern occurrence and copy what came next.",
    wc: "Word Content asks the model to identify prompt-local evidence and emit its associated label.",
    prontoqa: "PrOntoQA requires retrieving a demonstrated source-to-target predicate binding from context.",
    tt: "Token Translation combines prompt-local schema following with value-like lexical retrieval.",
  };
  featureModeButtons.forEach(button => button.setAttribute("aria-selected", String(button.dataset.featureMode === state.mode)));
  findingsEyebrow.textContent = pointer ? `Pointer-like features · ${taskName}` : "Value-like features · RAVEL results";
  findingsTitle.textContent = pointer ? "Mapped pointer-like effects are overwhelmingly diffuse." : "Value-like effects more often show low-dimensional organization.";
  findingsLede.textContent = pointer
    ? "Across the four ICL tasks, among the 80 mapped cases, 74 are classified as unresolved high-dimensional or diffuse. Only three form directed rays, and three occupy global low-dimensional subspaces. Thus, when pointer-like candidates have sufficient evidence for geometric analysis, stable low-dimensional structures are rare."
    : "Value-like candidates exhibit low-dimensional structure more often than pointer-like candidates. Their structured effects usually span multiple directions, while directed rays remain rare.";
  atlasEyebrow.textContent = pointer ? `Interactive pointer-like atlas · ${taskName}` : "Interactive value-like geometry atlas";
  atlasLede.textContent = pointer
    ? pointerTaskCopy[state.task] ?? "Select a point to inspect its recorded geometry label and metrics."
    : "Select an SAE architecture or geometry label, then hover or select a point to inspect its recorded label and metrics.";
}

function setFeatureMode(mode) {
  state.mode = mode === "pointer" ? "pointer" : "value";
  state.label = "all";
  state.selected = null;
  selectActiveData();
  updateDatasetCopy();
  renderAll();
}

async function initializeAtlas() {
  try {
    const [valueResponse, pointerResponse] = await Promise.all([
      fetch(document.body.dataset.valueAtlasUrl, { cache: "no-store" }),
      fetch(document.body.dataset.pointerAtlasUrl, { cache: "no-store" }),
    ]);
    if (!valueResponse.ok || !pointerResponse.ok) throw new Error(`Could not load results (${valueResponse.status}/${pointerResponse.status})`);
    state.datasets.value = await valueResponse.json();
    state.datasets.pointer = await pointerResponse.json();
    selectActiveData();
    updateDatasetCopy();
    renderHeroCards();
    renderAll();
  } catch (error) {
    chartRoot.innerHTML = `<p class="loading-message">The exported RAVEL data could not be loaded: ${error.message}</p>`;
    document.getElementById("plot-title").textContent = "Results unavailable";
  }
}

featureModeButtons.forEach(button => {
  button.addEventListener("click", () => setFeatureMode(button.dataset.featureMode));
  button.addEventListener("keydown", event => {
    if (!["ArrowLeft", "ArrowRight"].includes(event.key)) return;
    event.preventDefault();
    const groupButtons = [...button.closest('[role="tablist"]').querySelectorAll("[data-feature-mode]")];
    const index = groupButtons.indexOf(button);
    const offset = event.key === "ArrowRight" ? 1 : -1;
    const nextButton = groupButtons[(index + offset + groupButtons.length) % groupButtons.length];
    nextButton.focus();
    nextButton.click();
  });
});

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

const methodStory = document.querySelector("[data-method-story]");
if (methodStory) {
  const methodFigure = methodStory.querySelector("[data-method-figure]");
  const methodDiagram = methodStory.querySelector("[data-method-diagram]");
  const methodPhaseLabel = methodStory.querySelector("[data-method-phase-label]");
  const methodButtons = [...methodStory.querySelectorAll("[data-method-phase]")];
  const methodSteps = [...methodStory.querySelectorAll("[data-method-step]")];
  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
  const compactMethod = window.matchMedia("(max-width: 1100px)");
  const narrowMethod = window.matchMedia("(max-width: 760px)");
  const phaseNames = { 1: "Choose contexts", 2: "Shared SAE feature", 3: "Paired intervention", 4: "Compare effects" };
  const mobileViewBoxes = {
    1: "0 25 650 590",
    2: "360 90 540 500",
    3: "500 80 720 500",
    4: "430 600 580 215"
  };
  let activeMethodPhase = 1;

  const updateMethodViewport = () => {
    if (!methodDiagram) return;
    methodDiagram.setAttribute("viewBox", narrowMethod.matches ? mobileViewBoxes[activeMethodPhase] : "0 0 1440 820");
  };

  const setMethodPhase = phase => {
    const nextPhase = Math.max(1, Math.min(4, Number(phase)));
    activeMethodPhase = nextPhase;
    methodFigure.dataset.phase = String(nextPhase);
    methodPhaseLabel.textContent = `${String(nextPhase).padStart(2, "0")} / ${phaseNames[nextPhase]}`;
    methodButtons.forEach(button => button.setAttribute("aria-selected", String(Number(button.dataset.methodPhase) === nextPhase)));
    methodSteps.forEach(step => step.classList.toggle("is-active", Number(step.dataset.methodStep) === nextPhase));
    updateMethodViewport();
  };

  methodButtons.forEach(button => button.addEventListener("click", () => {
    const phase = Number(button.dataset.methodPhase);
    setMethodPhase(phase);
    if (!compactMethod.matches) methodSteps[phase - 1].scrollIntoView({ behavior: reducedMotion.matches ? "auto" : "smooth", block: "center" });
  }));
  methodButtons.forEach((button, index) => button.addEventListener("keydown", event => {
    if (!["ArrowLeft", "ArrowRight"].includes(event.key)) return;
    event.preventDefault();
    const offset = event.key === "ArrowRight" ? 1 : -1;
    const nextButton = methodButtons[(index + offset + methodButtons.length) % methodButtons.length];
    nextButton.focus();
    nextButton.click();
  }));

  let methodFrame;
  const updateMethodFromScroll = () => {
    methodFrame = undefined;
    if (compactMethod.matches) return;
    const midpoint = window.innerHeight * .5;
    const nearest = methodSteps.reduce((best, step) => {
      const distance = Math.abs(step.getBoundingClientRect().top + step.getBoundingClientRect().height / 2 - midpoint);
      return !best || distance < best.distance ? { step, distance } : best;
    }, null);
    if (nearest) setMethodPhase(nearest.step.dataset.methodStep);
  };
  window.addEventListener("scroll", () => { if (!methodFrame) methodFrame = requestAnimationFrame(updateMethodFromScroll); }, { passive: true });
  compactMethod.addEventListener("change", updateMethodFromScroll);
  narrowMethod.addEventListener("change", updateMethodViewport);
  updateMethodFromScroll();
  updateMethodViewport();

}
let atlasPulseFrame = null;
let atlasLastPaint = 0;
function animateAtlasPulse(timestamp) {
  if (atlasPulseFrame === null) return;
  const hasRenderedRepresentative = state.data
    ? renderedRecords().some(record => record.assets?.sphere || record.assets?.projection)
    : false;
  if (!document.hidden && state.data && hasRenderedRepresentative && timestamp - atlasLastPaint >= 40) {
    atlasLastPaint = timestamp;
    drawAtlas(timestamp);
  }
  atlasPulseFrame = requestAnimationFrame(animateAtlasPulse);
}
function startAtlasPulse() {
  if (atlasPulseFrame !== null || atlasReducedMotion.matches) return;
  atlasPulseFrame = requestAnimationFrame(animateAtlasPulse);
}
function stopAtlasPulse() {
  if (atlasPulseFrame === null) return;
  cancelAnimationFrame(atlasPulseFrame);
  atlasPulseFrame = null;
}
atlasReducedMotion.addEventListener("change", () => {
  if (atlasReducedMotion.matches) stopAtlasPulse();
  else startAtlasPulse();
  if (state.data) drawAtlas();
});
initializeAtlas().then(startAtlasPulse);
