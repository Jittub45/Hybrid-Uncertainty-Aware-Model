const SAMPLES = {
  rice:   { N: 60, P: 49, K: 44, temperature: 20.78, humidity: 84.5, ph: 6.24, rainfall: 240.08 },
  mango:  { N: 36, P: 26, K: 26, temperature: 30.17, humidity: 51.08, ph: 6.81, rainfall: 95.23 },
  cotton: { N: 140, P: 38, K: 15, temperature: 24.15, humidity: 75.88, ph: 6.02, rainfall: 69.92 },
  coffee: { N: 109, P: 31, K: 27, temperature: 23.06, humidity: 50.41, ph: 6.97, rainfall: 164.5 },
  apple:  { N: 0, P: 133, K: 200, temperature: 23.67, humidity: 90.49, ph: 5.71, rainfall: 104.23 },
};

const fields = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"];
const submitBtn = document.getElementById("submitBtn");
const resetBtn = document.getElementById("resetBtn");
const errorMsg = document.getElementById("errorMsg");
const resultSection = document.getElementById("resultSection");
const i18nNode = document.getElementById("predictI18n");
const I18N = i18nNode ? JSON.parse(i18nNode.textContent || "{}") : {};

function setError(msg = "") {
  if (!msg) {
    errorMsg.hidden = true;
    errorMsg.textContent = "";
    return;
  }
  errorMsg.hidden = false;
  errorMsg.textContent = msg;
}

function formData() {
  const data = {};
  for (const f of fields) {
    const val = parseFloat(document.getElementById(f).value);
    if (Number.isNaN(val)) {
      throw new Error(`${I18N.validValuePrefix || "Please enter a valid value for"} ${f}.`);
    }
    data[f] = val;
  }
  return data;
}

function resetForm() {
  document.getElementById("cropForm")?.reset();
  setError();
  resultSection.hidden = true;
}

function fillSample(name) {
  const sample = SAMPLES[name];
  if (!sample) return;
  fields.forEach((f) => {
    document.getElementById(f).value = sample[f];
  });
}

function renderResult(data) {
  resultSection.hidden = false;
  document.getElementById("resultEmoji").textContent = data.emoji || "🌱";
  document.getElementById("resultCrop").textContent = data.crop || "-";
  document.getElementById("resultDesc").textContent = data.desc || "-";

  const banner = document.getElementById("resultBanner");
  banner.style.borderColor = data.color || "#2e6b3c";

  const bars = document.getElementById("probBars");
  bars.innerHTML = "";
  (data.top5 || []).forEach((item) => {
    const row = document.createElement("div");
    row.className = "progress-row";
    row.innerHTML = `
      <div class="progress-meta"><strong>${item.crop}</strong><span>${Number(item.prob).toFixed(2)}%</span></div>
      <div class="progress-track"><div class="progress-fill" style="width:${item.prob}%"></div></div>
    `;
    bars.appendChild(row);
  });
}

async function predict() {
  setError();
  submitBtn.disabled = true;
  submitBtn.textContent = I18N.analyzing || "Analyzing...";

  try {
    const payload = formData();
    const resp = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const result = await resp.json();
    if (!result.success) {
      throw new Error(result.error || I18N.predictFailed || "Prediction failed.");
    }
    renderResult(result);
  } catch (err) {
    setError(err.message || I18N.predictError || "Could not predict crop.");
  } finally {
    submitBtn.disabled = false;
    submitBtn.textContent = I18N.getRecommendation || "Get Recommendation";
  }
}

submitBtn?.addEventListener("click", predict);
resetBtn?.addEventListener("click", resetForm);
document.querySelectorAll(".sample-pill").forEach((btn) => {
  btn.addEventListener("click", () => fillSample(btn.dataset.sample));
});
