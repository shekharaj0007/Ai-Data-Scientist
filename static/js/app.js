/**
 * AI Data Scientist — Web Application
 */
const API = window.location.origin;

const state = {
  filename: null,
  columns: [],
  profile: null,
  analysis: null,
  deployed: false,
  useCaseId: null,
  useCase: null,
  lastPredictResult: null,
  modelInfo: null,
  charts: {},
};

const PAGES = {
  overview: { title: "Overview", subtitle: "Platform dashboard and pipeline status", step: "upload" },
  studio: { title: "Data Studio", subtitle: "Upload data and configure your ML pipeline", step: "analyze" },
  models: { title: "Model Lab", subtitle: "Compare models, review SHAP explainability", step: "review" },
  deploy: { title: "Deployment", subtitle: "Human review gate before production", step: "deploy" },
  predict: { title: "Inference", subtitle: "Batch predictions with drift monitoring", step: "predict" },
};

const STEPS = [
  { id: "upload", num: "01", label: "Upload" },
  { id: "analyze", num: "02", label: "Analyze" },
  { id: "review", num: "03", label: "Review" },
  { id: "deploy", num: "04", label: "Deploy" },
  { id: "predict", num: "05", label: "Predict" },
];

function $(sel) { return document.querySelector(sel); }
function $$(sel) { return document.querySelectorAll(sel); }

function toast(msg, type = "info") {
  const el = document.createElement("div");
  el.className = `toast toast-${type}`;
  el.textContent = msg;
  $("#toasts").appendChild(el);
  setTimeout(() => el.remove(), 4500);
}

function showLoading(text = "Processing…") {
  $("#loading-text").textContent = text;
  $("#loading").classList.add("show");
}

function hideLoading() {
  $("#loading").classList.remove("show");
}

function isLikelyIdColumn(name, stats) {
  const n = (name || "").toLowerCase();
  if (["id", "uuid", "index", "key"].includes(n) || n.endsWith("_id") || n.endsWith("id")) return true;
  if (stats && stats.unique_count === state.profile?.n_rows && stats.unique_count > 20) return true;
  return false;
}

function populateTargetSelect(columns, selected = "") {
  const sel = $("#target-col");
  const cols = state.profile?.columns || {};
  let html = `<option value="">— Unsupervised (clustering) —</option>`;
  for (const c of columns) {
    const stats = cols[c];
    const bad = isLikelyIdColumn(c, stats);
    const label = bad ? `${c} (ID — not a target)` : c;
    html += `<option value="${c}"${c === selected ? " selected" : ""}${bad ? ' data-bad="1"' : ""}>${label}</option>`;
  }
  sel.innerHTML = html;
  updateTargetWarning();
}

function updateTargetWarning() {
  const sel = $("#target-col");
  const warn = $("#target-warning");
  if (!sel || !warn) return;
  const opt = sel.selectedOptions[0];
  if (opt?.dataset.bad === "1") {
    warn.style.display = "block";
    warn.innerHTML = `<div class="alert alert-warning">"${sel.value}" is an ID column — every row is unique. Pick an outcome column (e.g. Churn) or leave blank for clustering.</div>`;
  } else {
    warn.style.display = "none";
    warn.innerHTML = "";
  }
}

$("#target-col")?.addEventListener("change", updateTargetWarning);

function titleCase(s) {
  return (s || "").replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

function renderPipeline(containerId, currentStep) {
  const idx = STEPS.findIndex((s) => s.id === currentStep);
  const el = $(containerId);
  if (!el) return;
  el.innerHTML = STEPS.map((step, i) => {
    let cls = "step";
    if (step.id === currentStep) cls += " active";
    else if (i < idx) cls += " done";
    return `<div class="${cls}"><div class="step-num">${step.num}</div><div class="step-label">${step.label}</div></div>`;
  }).join("");
}

async function api(path, options = {}) {
  const res = await fetch(`${API}${path}`, options);
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    let detail = err.detail || res.statusText;
    if (Array.isArray(detail)) detail = detail.map((d) => d.msg || d).join("; ");
    throw new Error(detail);
  }
  return res.json();
}

function destroyChart(key) {
  if (state.charts[key]) {
    state.charts[key].destroy();
    delete state.charts[key];
  }
}

function navigate(page) {
  $$(".nav-item").forEach((n) => n.classList.toggle("active", n.dataset.page === page));
  $$(".page").forEach((p) => p.classList.remove("active"));
  $(`#page-${page}`)?.classList.add("active");

  const meta = PAGES[page];
  $("#topbar-title").textContent = meta.title;
  $("#topbar-subtitle").textContent = meta.subtitle;
  renderPipeline(`#pipeline-${page}`, meta.step);

  if (page === "overview") renderOverview();
  if (page === "models") renderModels();
  if (page === "deploy") renderDeploy();
  if (page === "predict") renderPredictAlert();
}

$$(".nav-item").forEach((btn) => {
  btn.addEventListener("click", () => navigate(btn.dataset.page));
});

async function checkHealth() {
  const dot = $("#api-status-dot");
  const text = $("#api-status-text");
  try {
    const h = await api("/health");
    dot.classList.add("online");
    const llm = h.llm_ready === true ? " · LLM on" : h.llm_ready === false ? " · LLM off (check .env)" : " · restart server";
    const pt = h.pytorch?.available ? ` · PyTorch (${h.pytorch.device})` : "";
    text.textContent = `API online${llm}${pt}`;
    state.deployed = h.production_model_ready;

    if (!state.analysis && h.analysis_saved) {
      try {
        state.analysis = await api("/analysis/latest");
        if (state.analysis.filename) state.filename = state.analysis.filename;
        updateRunBadge();
      } catch {
        /* no saved analysis */
      }
    }
  } catch {
    dot.classList.remove("online");
    text.textContent = "API offline — run start.bat";
  }
}

const uploadZone = $("#upload-zone");
const fileInput = $("#file-input");

uploadZone.addEventListener("click", () => fileInput.click());
uploadZone.addEventListener("dragover", (e) => { e.preventDefault(); uploadZone.classList.add("dragover"); });
uploadZone.addEventListener("dragleave", () => uploadZone.classList.remove("dragover"));
uploadZone.addEventListener("drop", (e) => {
  e.preventDefault();
  uploadZone.classList.remove("dragover");
  if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
});
fileInput.addEventListener("change", () => {
  if (fileInput.files.length) handleFile(fileInput.files[0]);
});

async function handleFile(file) {
  if (!file.name.endsWith(".csv")) {
    toast("Please upload a CSV file", "error");
    return;
  }

  showLoading("Profiling dataset…");
  try {
    const form = new FormData();
    form.append("file", file);
    const data = await api("/upload", { method: "POST", body: form });

    state.filename = data.filename;
    state.columns = data.columns;
    state.profile = data.profile;
    state.useCaseId = null;
    state.useCase = null;

    uploadZone.classList.add("has-file");
    $("#upload-filename").textContent = `${file.name} · ${data.profile.n_rows.toLocaleString()} rows`;
    $("#run-pipeline").disabled = false;

    populateTargetSelect(data.columns);

    renderProfile(data.profile);
    renderUseCaseBanner();
    toast("Dataset uploaded and profiled", "success");
  } catch (err) {
    toast(err.message, "error");
  } finally {
    hideLoading();
  }
}

function renderProfile(profile) {
  const cols = profile.columns || {};
  const missingCount = Object.values(cols).filter((c) => c.missing_pct > 0).length;
  const highCard = Object.values(cols).filter((c) => c.unique_count > 50).length;

  let html = `
    <div class="grid grid-4" style="margin-bottom:1rem">
      <div class="stat-card card"><div class="stat-label">Rows</div><div class="stat-value">${profile.n_rows.toLocaleString()}</div></div>
      <div class="stat-card card"><div class="stat-label">Columns</div><div class="stat-value">${profile.n_cols}</div></div>
      <div class="stat-card card"><div class="stat-label">Missing cols</div><div class="stat-value">${missingCount}</div></div>
      <div class="stat-card card"><div class="stat-label">High cardinality</div><div class="stat-value">${highCard}</div></div>
    </div>
    <div class="table-wrap"><table>
      <thead><tr><th>Column</th><th>Type</th><th>Missing %</th><th>Unique</th></tr></thead>
      <tbody>`;

  for (const [name, stats] of Object.entries(cols)) {
    html += `<tr><td>${name}</td><td>${stats.dtype}</td><td>${stats.missing_pct}%</td><td>${stats.unique_count}</td></tr>`;
  }
  html += "</tbody></table></div>";
  $("#profile-content").innerHTML = html;
}

$("#run-pipeline").addEventListener("click", async () => {
  if (!state.filename) return;

  const targetCol = $("#target-col").value || null;
  const opt = $("#target-col").selectedOptions[0];
  if (opt?.dataset.bad === "1") {
    toast(`"${targetCol}" is an ID column, not a valid target. Choose Churn/Converted or leave blank.`, "error");
    return;
  }

  showLoading("Training models and generating explainability…");
  try {
    const payload = {
      filename: state.filename,
      target_col: $("#target-col").value || null,
      use_llm: $("#use-llm").checked,
      use_case_id: state.useCaseId,
    };
    state.analysis = await api("/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    state.analysis.filename = state.filename;
    state.deployed = false;
    updateRunBadge();
    toast("Pipeline complete — review results in Model Lab", "success");
    navigate("models");
  } catch (err) {
    toast(err.message, "error");
  } finally {
    hideLoading();
  }
});

function updateRunBadge() {
  const badge = $("#run-badge");
  if (!state.analysis) { badge.style.display = "none"; return; }
  badge.style.display = "inline-flex";
  badge.className = `badge ${state.deployed ? "badge-deployed" : "badge-pending"}`;
  badge.textContent = titleCase(state.deployed ? "deployed" : state.analysis.status);
}

function renderUseCaseCards(cases) {
  return `<div class="grid grid-2" style="margin-bottom:1.25rem">${cases.map((uc) => `
    <div class="card">
      <div class="card-header">
        <div>
          <div class="card-title">${uc.name}</div>
          <div class="card-desc">${uc.description}</div>
        </div>
        <button class="btn btn-primary" data-demo="${uc.id}">Load demo</button>
      </div>
      <div class="card-desc" style="margin-top:0.75rem">${uc.industry} · Target: <strong>${uc.target_col}</strong></div>
    </div>`).join("")}</div>`;
}

function bindDemoButtons(container) {
  container?.querySelectorAll("[data-demo]").forEach((btn) => {
    btn.addEventListener("click", () => loadDemo(btn.dataset.demo));
  });
}

const FALLBACK_USE_CASES = [
  { id: "customer_churn", name: "Customer Churn Prediction", description: "Predict cancellations and trigger retention outreach.", industry: "Telecom / SaaS", target_col: "Churn" },
  { id: "lead_scoring", name: "Sales Lead Scoring", description: "Rank leads by conversion likelihood.", industry: "B2B SaaS", target_col: "Converted" },
  { id: "fraud_detection", name: "Payment Fraud Detection", description: "Flag suspicious transactions before they settle.", industry: "Banking / Fintech", target_col: "IsFraud" },
  { id: "loan_default", name: "Loan Default Risk", description: "Predict loan defaults for underwriting decisions.", industry: "Banking / Lending", target_col: "Default" },
];

async function renderOverview() {
  const el = $("#overview-stats");
  let cases = [];
  let apiStale = false;
  try {
    const data = await api("/use-cases");
    cases = data.use_cases || [];
  } catch {
    cases = FALLBACK_USE_CASES;
    apiStale = true;
  }

  const staleBanner = apiStale
    ? `<div class="alert alert-warning" style="margin-bottom:1rem"><strong>Server needs restart.</strong> Stop the old server (Ctrl+C) and run <strong>start.bat</strong> again for LLM + full features. Demo buttons below still work.</div>`
    : "";

  const useCaseCards = renderUseCaseCards(cases);

  if (!state.analysis) {
    el.innerHTML = staleBanner + useCaseCards + `<div class="card"><div class="empty"><div class="empty-icon">⚡</div><div class="empty-title">Ready to start</div><div class="empty-desc">Pick a production use case above or upload your own CSV in Data Studio</div></div></div>`;
    bindDemoButtons(el);
    return;
  }

  const a = state.analysis;
  const useCaseBlock = a.use_case
    ? `<div class="alert alert-info" style="margin-bottom:1rem"><strong>${a.use_case.name}</strong> — ${a.use_case.business_goal}</div>`
    : "";
  el.innerHTML = staleBanner + useCaseCards + useCaseBlock + `
    <div class="card">
      <div class="card-header"><div class="card-title">Latest run</div></div>
      <div class="grid grid-4">
        <div class="stat-card"><div class="stat-label">Problem</div><div class="stat-value sm">${titleCase(a.problem_type)}</div></div>
        <div class="stat-card"><div class="stat-label">Best model</div><div class="stat-value sm">${a.best_model_name}</div></div>
        <div class="stat-card"><div class="stat-label">Dataset</div><div class="stat-value sm">${a.filename || "—"}</div></div>
        <div class="stat-card"><div class="stat-label">Status</div><div class="stat-value sm">${titleCase(state.deployed ? "deployed" : a.status)}</div></div>
      </div>
    </div>`;
  bindDemoButtons(el);
}

async function loadDemo(useCaseId) {
  const labels = {
    customer_churn: "Customer Churn",
    lead_scoring: "Lead Scoring",
    fraud_detection: "Fraud Detection",
    loan_default: "Loan Default",
  };
  showLoading(`Loading ${labels[useCaseId] || useCaseId} demo…`);
  try {
    const data = await api(`/demo/${useCaseId}`, { method: "POST" });
    state.filename = data.filename;
    state.columns = data.columns;
    state.profile = data.profile;
    state.useCaseId = data.use_case_id;
    state.useCase = data.use_case;
    state.analysis = null;
    state.deployed = false;

    uploadZone.classList.add("has-file");
    $("#upload-filename").textContent = `${data.filename} · ${data.profile.n_rows.toLocaleString()} rows · ${data.use_case.name}`;
    $("#run-pipeline").disabled = false;

    populateTargetSelect(data.columns, data.target_col);

    renderProfile(data.profile);
    renderUseCaseBanner();
    toast(`${data.use_case.name} loaded — run pipeline with target ${data.target_col}`, "success");
    navigate("studio");
  } catch (err) {
    toast(err.message, "error");
  } finally {
    hideLoading();
  }
}

function renderUseCaseBanner() {
  const banner = $("#use-case-banner");
  if (!banner) return;
  if (!state.useCase) {
    banner.innerHTML = "";
    banner.style.display = "none";
    return;
  }
  banner.style.display = "block";
  banner.innerHTML = `<div class="alert alert-info"><strong>${state.useCase.name}</strong> — ${state.useCase.description}</div>`;
}

function renderModels() {
  const el = $("#models-content");
  if (!state.analysis) {
    el.innerHTML = `<div class="empty"><div class="empty-icon">🧪</div><div class="empty-title">No analysis yet</div><div class="empty-desc">Run the pipeline in Data Studio to see model results</div></div>`;
    return;
  }

  const a = state.analysis;
  let warnings = "";
  if (a.leakage_warnings?.length) {
    warnings = `<div class="alert alert-warning">Possible data leakage: ${a.leakage_warnings.join(", ")}</div>`;
  }

  el.innerHTML = `
    ${warnings}
    <div class="grid grid-4" style="margin-bottom:1.25rem">
      <div class="stat-card card"><div class="stat-label">Problem</div><div class="stat-value sm">${titleCase(a.problem_type)}</div></div>
      <div class="stat-card card"><div class="stat-label">Best model</div><div class="stat-value sm">${a.best_model_name}</div></div>
      <div class="stat-card card"><div class="stat-label">Models trained</div><div class="stat-value sm">${a.models_trained || a.leaderboard?.length || 0}</div></div>
      <div class="stat-card card"><div class="stat-label">Modality</div><div class="stat-value sm">${titleCase(a.modality_info?.primary_modality || "tabular")}</div></div>
    </div>

    <div class="tabs" id="model-tabs">
      <button class="tab active" data-tab="leaderboard">Leaderboard</button>
      <button class="tab" data-tab="architecture">Architecture Guide</button>
      <button class="tab" data-tab="explain">Explainability</button>
      <button class="tab" data-tab="suggestions">Suggestions</button>
    </div>

    <div class="tab-panel active" id="tab-leaderboard">
      <div class="split">
        <div class="card"><div class="chart-container lg"><canvas id="chart-leaderboard"></canvas></div></div>
        <div class="card"><div id="leaderboard-table"></div></div>
      </div>
    </div>

    <div class="tab-panel" id="tab-architecture">
      <div id="architecture-content"></div>
    </div>

    <div class="tab-panel" id="tab-explain">
      <div id="explain-content"></div>
    </div>

    <div class="tab-panel" id="tab-suggestions">
      <div id="suggestions-content"></div>
    </div>`;

  renderLeaderboard(a.leaderboard);
  renderArchitecture(a);
  renderExplainability(a);
  renderSuggestions(a);

  $$("#model-tabs .tab").forEach((tab) => {
    tab.addEventListener("click", () => {
      $$("#model-tabs .tab").forEach((t) => t.classList.remove("active"));
      tab.classList.add("active");
      $$(".tab-panel").forEach((p) => p.classList.remove("active"));
      $(`#tab-${tab.dataset.tab}`).classList.add("active");
    });
  });
}

function renderLeaderboard(rows) {
  if (!rows?.length) return;

  const scoreKey = rows[0].test_score !== undefined ? "test_score" : Object.keys(rows[0]).pop();
  const sorted = [...rows].sort((a, b) => (b[scoreKey] || 0) - (a[scoreKey] || 0));

  let table = `<div class="table-wrap"><table><thead><tr>`;
  for (const key of Object.keys(sorted[0])) table += `<th>${titleCase(key)}</th>`;
  table += `</tr></thead><tbody>`;
  for (const row of sorted) {
    table += "<tr>";
    for (const val of Object.values(row)) table += `<td>${val}</td>`;
    table += "</tr>";
  }
  table += "</tbody></table></div>";
  $("#leaderboard-table").innerHTML = table;

  destroyChart("leaderboard");
  const ctx = $("#chart-leaderboard");
  if (!ctx) return;

  state.charts.leaderboard = new Chart(ctx, {
    type: "bar",
    data: {
      labels: sorted.map((r) => r.model || r.k || "model"),
      datasets: [{
        data: sorted.map((r) => r[scoreKey]),
        backgroundColor: sorted.map((_, i) => `rgba(99, 102, 241, ${0.4 + i * 0.15})`),
        borderColor: "#818cf8",
        borderWidth: 1,
        borderRadius: 6,
      }],
    },
    options: {
      indexAxis: "y",
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false }, title: { display: true, text: "Model comparison", color: "#fafafa" } },
      scales: {
        x: { grid: { color: "#27272a" }, ticks: { color: "#71717a" } },
        y: { grid: { display: false }, ticks: { color: "#d4d4d8" } },
      },
    },
  });
}

function renderArchitecture(a) {
  const el = $("#architecture-content");
  const plan = a.architecture_plan;
  if (!plan) {
    el.innerHTML = `<div class="empty"><div class="empty-title">No recommendations</div></div>`;
    return;
  }

  const top = plan.top_recommendation;
  let html = "";

  if (plan.decision_summary) {
    html += `<div class="narrative" style="margin-bottom:1rem">${plan.decision_summary.replace(/\*\*/g, "")}</div>`;
  }

  if (top) {
    html += `
      <div class="card" style="margin-bottom:1rem;border-color:rgba(99,102,241,0.4)">
        <div class="card-title">Top recommendation: ${top.architecture}</div>
        <div class="card-desc" style="margin-top:0.5rem"><strong>Family:</strong> ${top.family.toUpperCase()} ·
          <span class="badge ${top.implemented_in_pipeline ? "badge-deployed" : "badge-pending"}">${top.implemented_in_pipeline ? "Auto-trained here" : "External / GPU needed"}</span>
        </div>
        <p class="card-desc" style="margin-top:0.75rem">${top.reason}</p>
        <p class="card-desc" style="margin-top:0.35rem"><em>When to use:</em> ${top.when_to_use}</p>
      </div>`;
  }

  html += `<div class="card-title" style="margin-bottom:0.75rem">Full architecture catalog for your data</div>`;
  html += `<ul class="suggestion-list">`;

  for (const rec of plan.recommendations || []) {
    const tag = rec.implemented_in_pipeline ? "✅ Trained" : "📋 Recommended";
    html += `<li>
      <strong>${rec.architecture}</strong> <span class="badge badge-info">${rec.family}</span><br>
      <span style="color:var(--muted);font-size:0.82rem">${rec.reason}</span><br>
      <span style="font-size:0.78rem">${tag}</span>
    </li>`;
  }
  html += `</ul>`;

  if (plan.by_family) {
    html += `<div class="card" style="margin-top:1rem"><div class="card-title">By family</div><div class="card-desc" style="margin-top:0.5rem">`;
    for (const [family, items] of Object.entries(plan.by_family)) {
      html += `<div style="margin-bottom:0.35rem"><strong>${family}:</strong> ${items.slice(0, 2).join(", ")}</div>`;
    }
    html += `</div></div>`;
  }

  el.innerHTML = html;
}

function getImportanceValue(row) {
  return row.importance ?? row.mean_abs_shap ?? 0;
}

function renderConfusionMatrix(cm, labels) {
  if (!cm?.length) return "";
  let html = `<div class="card" style="margin-bottom:1rem"><div class="card-title" style="margin-bottom:0.75rem">Confusion matrix</div><div class="table-wrap"><table><thead><tr><th>Actual \\ Pred</th>`;
  labels.forEach((l) => { html += `<th>${l}</th>`; });
  html += `</tr></thead><tbody>`;
  cm.forEach((row, i) => {
    html += `<tr><th>${labels[i] ?? i}</th>`;
    row.forEach((v) => { html += `<td>${v}</td>`; });
    html += `</tr>`;
  });
  html += `</tbody></table></div></div>`;
  return html;
}

function renderExplainability(a) {
  const el = $("#explain-content");
  const exp = a.explanation || {};
  const method = exp.method || "unknown";
  const methodLabels = {
    shap: "SHAP (Shapley values)",
    permutation: "Permutation importance",
    native: "Model-native importance",
    cluster_profile: "Cluster profiles",
    forecast_analysis: "Forecast analysis",
    text_saliency: "Text token saliency",
  };

  let html = `<div style="margin-bottom:1rem"><span class="badge badge-info">${methodLabels[method] || method}</span></div>`;

  if (exp.metrics && Object.keys(exp.metrics).length) {
    html += `<div class="grid grid-4" style="margin-bottom:1rem">`;
    for (const [k, v] of Object.entries(exp.metrics)) {
      if (typeof v === "object") continue;
      html += `<div class="stat-card card"><div class="stat-label">${titleCase(k)}</div><div class="stat-value sm">${v}</div></div>`;
    }
    html += `</div>`;
  }

  if (exp.confusion_matrix) {
    html += renderConfusionMatrix(exp.confusion_matrix, exp.class_labels || []);
  }

  if (exp.cluster_profiles?.length) {
    html += `<div class="card" style="margin-bottom:1rem"><div class="card-title">Cluster profiles</div>`;
    exp.cluster_profiles.forEach((c) => {
      html += `<div style="margin-top:0.75rem"><strong>Cluster ${c.cluster}</strong> (${c.size} points)<ul class="suggestion-list">`;
      c.top_features.forEach((f) => {
        html += `<li>${f.feature}: ${f.mean_value}</li>`;
      });
      html += `</ul></div>`;
    });
    html += `</div>`;
  }

  if (exp.forecast_summary) {
    const fs = exp.forecast_summary;
    html += `<div class="card" style="margin-bottom:1rem"><div class="card-title">Forecast performance</div>
      <div class="card-desc">Test MAE: ${fs.test_mae ?? "—"} · RMSE: ${fs.test_rmse ?? "—"} · Test size: ${fs.test_size ?? "—"}</div></div>`;
  }

  const importance = a.feature_importance || exp.global_importance || [];
  if (importance.length) {
    html += `<div class="card" style="margin-bottom:1rem"><div class="chart-container lg"><canvas id="chart-shap"></canvas></div></div>`;
    html += `<div class="table-wrap"><table><thead><tr><th>Feature</th><th>Importance</th><th>Method</th></tr></thead><tbody>`;
    importance.forEach((r) => {
      html += `<tr><td>${r.feature || r.token}</td><td>${getImportanceValue(r).toFixed?.(4) ?? getImportanceValue(r)}</td><td>${r.method_detail || method}</td></tr>`;
    });
    html += `</tbody></table></div>`;
  }

  if (a.narrative || exp.narrative) {
    html += `<div class="card"><div class="card-title" style="margin-bottom:0.75rem">Executive summary</div><div class="narrative">${a.narrative || exp.narrative}</div></div>`;
  }

  if (!importance.length && !exp.cluster_profiles && !exp.forecast_summary && !a.narrative) {
    html += `<div class="empty"><div class="empty-title">Limited explainability</div><div class="empty-desc">${(exp.warnings || []).join(" ") || "Run with LLM enabled for narrative insights."}</div></div>`;
  }

  el.innerHTML = html;

  if (importance.length) {
    const imp = [...importance].sort((x, y) => getImportanceValue(x) - getImportanceValue(y));
    destroyChart("shap");
    const ctx = $("#chart-shap");
    if (ctx) {
      state.charts.shap = new Chart(ctx, {
        type: "bar",
        data: {
          labels: imp.map((r) => r.feature || r.token),
          datasets: [{
            data: imp.map(getImportanceValue),
            backgroundColor: "rgba(56, 189, 248, 0.5)",
            borderColor: "#38bdf8",
            borderWidth: 1,
            borderRadius: 6,
          }],
        },
        options: {
          indexAxis: "y",
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: { display: false },
            title: { display: true, text: "Global feature importance", color: "#fafafa" },
          },
          scales: {
            x: { grid: { color: "#27272a" }, ticks: { color: "#71717a" } },
            y: { grid: { display: false }, ticks: { color: "#d4d4d8" } },
          },
        },
      });
    }
  }
}

function renderSuggestions(a) {
  const el = $("#suggestions-content");
  let items = (a.feature_suggestions || []).filter(
    (s) => !String(s).includes("LLM narrative unavailable")
  );
  if (!items.length && a.llm_enabled === false) {
    el.innerHTML = `<div class="alert alert-warning">LLM is off — add ANTHROPIC_API_KEY to .env, restart start.bat, then <strong>run the pipeline again</strong>.</div>`;
    return;
  }
  if (!items.length) {
    el.innerHTML = `<div class="empty"><div class="empty-title">No suggestions yet</div><div class="empty-desc">This run was saved before LLM was enabled. Go to Data Studio and click <strong>Run full pipeline</strong> again.</div></div>`;
    return;
  }
  el.innerHTML = `<ul class="suggestion-list">${items.map((s) => `<li>${s}</li>`).join("")}</ul>`;
}

function renderDeploy() {
  const el = $("#deploy-content");
  if (!state.analysis) {
    el.innerHTML = `<div class="empty"><div class="empty-icon">🚀</div><div class="empty-title">Nothing staged</div><div class="empty-desc">Complete an analysis run before deploying</div></div>`;
    return;
  }

  const a = state.analysis;
  const hasLeakage = a.leakage_warnings?.length > 0;
  const hasExplain = !!(a.feature_importance?.length || a.explanation || a.narrative);

  el.innerHTML = `
    <div class="split">
      <div class="card">
        <div class="card-title" style="margin-bottom:1rem">Deployment gate</div>
        <div class="grid grid-2" style="margin-bottom:1.25rem">
          <div class="stat-card"><div class="stat-label">Candidate</div><div class="stat-value sm">${a.best_model_name}</div></div>
          <div class="stat-card"><div class="stat-label">Status</div><div class="stat-value sm"><span class="badge ${state.deployed ? "badge-deployed" : "badge-pending"}">${state.deployed ? "Deployed" : "Pending review"}</span></div></div>
        </div>

        <div class="card-title" style="margin-bottom:0.5rem;font-size:0.85rem">Pre-flight checklist</div>
        <ul class="checklist">
          <li><span class="check-icon">✅</span> Leaderboard reviewed</li>
          <li><span class="check-icon">${hasLeakage ? "⚠️" : "✅"}</span> Leakage warnings ${hasLeakage ? "present — review required" : "clear"}</li>
          <li><span class="check-icon">${hasExplain ? "✅" : "⚠️"}</span> Explainability ${hasExplain ? "reviewed" : "not available"}</li>
        </ul>

        <label class="checkbox-row">
          <input type="checkbox" id="deploy-confirm" ${state.deployed ? "checked disabled" : ""} />
          I confirm this model is ready for production
        </label>

        <button class="btn btn-primary btn-block" id="approve-btn" ${state.deployed ? "disabled" : ""}>
          ${state.deployed ? "Already deployed" : "Approve & deploy"}
        </button>
      </div>

      <div class="card">
        <div class="card-title" style="margin-bottom:0.75rem">Production artifact</div>
        <div class="card-desc" style="margin-bottom:1rem">Download the serialized model after approval.</div>
        <button class="btn btn-secondary" id="download-btn" ${state.deployed ? "" : "disabled"}>Download model (.pkl)</button>
      </div>
    </div>`;

  if (!state.deployed) {
    $("#approve-btn").addEventListener("click", async () => {
      if (!$("#deploy-confirm").checked) {
        toast("Please confirm the checklist first", "error");
        return;
      }
      showLoading("Deploying model…");
      try {
        await api("/approve", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ confirmed: true }),
        });
        state.deployed = true;
        updateRunBadge();
        toast("Model deployed to production", "success");
        renderDeploy();
        loadPredictSchema();
      } catch (err) {
        toast(err.message, "error");
      } finally {
        hideLoading();
      }
    });
  }

  $("#download-btn")?.addEventListener("click", () => {
    window.open(`${API}/model/download`, "_blank");
  });
}

function renderPredictAlert() {
  const el = $("#predict-alert");
  if (!state.deployed) {
    el.innerHTML = `<div class="alert alert-info">Deploy a model first to enable live inference.</div>`;
    return;
  }
  el.innerHTML = `<div class="alert alert-success">Production model is live and ready for predictions.</div>`;
  loadPredictSchema();
}

async function loadPredictSchema() {
  try {
    const info = await api("/model/info");
    state.modelInfo = info;
    const sample = info.sample_payload?.length ? info.sample_payload : [{}];
    $("#predict-json").value = JSON.stringify(sample, null, 2);
    let hint = info.required_columns?.length
      ? `Required columns (raw business fields): ${info.required_columns.join(", ")}`
      : "See sample payload below";
    if (info.use_case?.name) {
      hint = `${info.use_case.name} — ${hint}`;
    }
    if (info.accepts_raw_columns) {
      hint += " · Preprocessing is applied automatically at inference.";
    }
    const hintEl = $("#predict-schema-hint");
    if (hintEl) hintEl.textContent = hint;
  } catch {
    /* model info unavailable */
  }
}

async function loadSampleBatch() {
  const useCaseId = state.modelInfo?.use_case_id;
  if (!useCaseId) {
    toast("Deploy a use-case model to load a sample batch", "error");
    return;
  }
  try {
    const data = await api(`/demo/${useCaseId}/sample-batch`);
    $("#predict-json").value = JSON.stringify(data.records, null, 2);
    toast(`Loaded ${data.records.length} sample records`, "success");
  } catch (err) {
    toast(err.message, "error");
  }
}

async function exportCsvReport() {
  let records;
  try {
    records = JSON.parse($("#predict-json").value);
    if (!Array.isArray(records)) throw new Error("Payload must be a JSON array");
  } catch (err) {
    toast(err.message, "error");
    return;
  }

  showLoading("Generating CSV report…");
  try {
    const res = await fetch(`${API}/reports/export`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ records }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || res.statusText);
    }
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = res.headers.get("Content-Disposition")?.match(/filename=\"(.+)\"/)?.[1]
      || "prediction_report.csv";
    a.click();
    URL.revokeObjectURL(url);
    toast("CSV report downloaded", "success");
  } catch (err) {
    toast(err.message, "error");
  } finally {
    hideLoading();
  }
}

function riskBadgeClass(tier) {
  if (tier === "high") return "badge-pending";
  if (tier === "medium") return "badge-info";
  return "badge-deployed";
}

$("#run-predict").addEventListener("click", async () => {
  let records;
  try {
    records = JSON.parse($("#predict-json").value);
    if (!Array.isArray(records)) throw new Error("Payload must be a JSON array");
  } catch (err) {
    toast(err.message, "error");
    return;
  }

  showLoading("Running inference…");
  try {
    const result = await api("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ records }),
    });

    state.lastPredictResult = result;

    const scoreLabel = result.high_risk_label || "Score";
    const hasBusiness = result.probability_scores || result.probabilities || result.risk_tier;
    let html = `<div class="table-wrap"><table><thead><tr>`;
    html += `<th>#</th><th>Prediction</th>`;
    if (hasBusiness) {
      html += `<th>${scoreLabel}</th><th>Priority</th><th>Recommended action</th>`;
    }
    html += `</tr></thead><tbody>`;

    result.predictions.forEach((p, i) => {
      html += `<tr><td>${i + 1}</td><td>${p}</td>`;
      if (hasBusiness) {
        const prob = result.probability_scores?.[i] ?? result.probabilities?.[i] ?? result.churn_probability?.[i];
        const tier = result.risk_tier?.[i] || "—";
        const action = result.recommended_action?.[i] || "—";
        html += `<td>${prob != null ? `${(prob * 100).toFixed(1)}%` : "—"}</td>`;
        html += `<td><span class="badge ${riskBadgeClass(tier)}">${titleCase(tier)}</span></td>`;
        html += `<td>${action}</td>`;
      }
      html += `</tr>`;
    });
    html += "</tbody></table></div>";

    if (result.drift_warnings?.length) {
      html = `<div class="alert alert-warning">${result.drift_warnings.map((w) => `• ${w}`).join("<br>")}</div>` + html;
    } else {
      html = `<div class="alert alert-success">No drift detected against training baseline.</div>` + html;
    }

    $("#predict-results").innerHTML = html;
    toast("Predictions complete", "success");
  } catch (err) {
    toast(err.message, "error");
  } finally {
    hideLoading();
  }
});

$("#load-sample-batch")?.addEventListener("click", loadSampleBatch);
$("#export-report")?.addEventListener("click", exportCsvReport);

document.addEventListener("DOMContentLoaded", () => {
  renderPipeline("#pipeline-overview", "upload");
  checkHealth();
  setInterval(checkHealth, 30000);
  renderOverview();
});
