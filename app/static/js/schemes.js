(function () {
  const i18nNode = document.getElementById("schemesI18n");
  const i18n = i18nNode ? JSON.parse(i18nNode.textContent || "{}") : {};

  const keywordEl = document.getElementById("schemeKeyword");
  const stateEl = document.getElementById("schemeState");
  const categoryEl = document.getElementById("schemeCategory");
  const farmerTypeEl = document.getElementById("schemeFarmerType");
  const incomeLevelEl = document.getElementById("schemeIncomeLevel");
  const landSizeEl = document.getElementById("schemeLandSize");
  const searchBtn = document.getElementById("schemeSearchBtn");
  const resetBtn = document.getElementById("schemeResetBtn");
  const summaryEl = document.getElementById("schemeSummary");
  const errorEl = document.getElementById("schemeError");
  const resultsEl = document.getElementById("schemesResults");

  if (!searchBtn || !resultsEl) {
    console.error("[Schemes] Missing critical elements, aborting");
    return;
  }

  function setError(message) {
    if (!errorEl) return;
    if (!message) {
      errorEl.hidden = true;
      errorEl.textContent = "";
      return;
    }
    errorEl.hidden = false;
    errorEl.textContent = message;
  }

  function option(text, value) {
    const el = document.createElement("option");
    el.textContent = text;
    el.value = value;
    return el;
  }

  function setOptions(selectEl, values, keepFirst) {
    if (!selectEl) return;
    const first = selectEl.options[0];
    selectEl.innerHTML = "";
    if (keepFirst && first) {
      selectEl.appendChild(first);
    }
    (values || []).forEach((item) => {
      selectEl.appendChild(option(item, item));
    });
  }

  function card(scheme) {
    const article = document.createElement("article");
    article.className = "panel-card";

    const categories = (scheme.categories || []).slice(0, 4).join(", ");
    const states = (scheme.states || []).join(", ");

    article.innerHTML = `
      <h3>${scheme.scheme_name || "-"}</h3>
      <p><strong>${i18n.desc || "Description"}:</strong> ${scheme.short_description || scheme.description || "-"}</p>
      <p><strong>${i18n.benefits || "Benefits"}:</strong> ${scheme.benefits || "-"}</p>
      <p><strong>${i18n.eligibility || "Eligibility"}:</strong> ${scheme.eligibility || "-"}</p>
      <p><strong>${i18n.categories || "Categories"}:</strong> ${categories || "-"}</p>
      <p><strong>${i18n.states || "States"}:</strong> ${states || "-"}</p>
      ${scheme.url ? `<p><a class="btn btn-ghost" target="_blank" rel="noopener noreferrer" href="${scheme.url}">${i18n.viewDetails || "View Details"}</a></p>` : ""}
    `;
    return article;
  }

  function setLoading(isLoading) {
    searchBtn.disabled = isLoading;
    searchBtn.textContent = isLoading ? (i18n.searching || "Searching...") : (searchBtn.dataset.defaultText || searchBtn.textContent);
  }

  async function loadOptions() {
    try {
      const resp = await fetch("/api/schemes/options");
      const data = await resp.json();
      if (!resp.ok || !data.success) {
        throw new Error(data.error || "Failed to load filter options.");
      }
      setOptions(stateEl, data.states || [], true);
      setOptions(categoryEl, data.categories || [], true);
      setOptions(farmerTypeEl, data.farmer_types || [], true);
      setOptions(incomeLevelEl, data.income_levels || [], true);
      setOptions(landSizeEl, data.land_sizes || [], true);
    } catch (error) {
      setError(error.message || "Failed to load filter options.");
    }
  }

  async function searchSchemes() {
    setError("");
    setLoading(true);

    try {
      const params = new URLSearchParams({
        language: window.APP_LANG || "en",
        keyword: keywordEl?.value?.trim() || "",
        state: stateEl?.value || "",
        category: categoryEl?.value || "",
        farmer_type: farmerTypeEl?.value || "",
        income_level: incomeLevelEl?.value || "",
        land_size: landSizeEl?.value || "",
        per_page: "300",
      });

      const resp = await fetch(`/api/schemes?${params.toString()}`);
      const data = await resp.json();
      if (!resp.ok || !data.success) {
        throw new Error(data.error || "Failed to fetch schemes.");
      }

      const rows = data.results || [];
      if (summaryEl) {
        summaryEl.textContent = `${i18n.total || "Total schemes found"}: ${data.total || rows.length}`;
      }

      resultsEl.innerHTML = "";
      if (rows.length === 0) {
        const empty = document.createElement("article");
        empty.className = "panel-card";
        empty.innerHTML = `<p>${i18n.noResults || "No schemes found for the selected filters."}</p>`;
        resultsEl.appendChild(empty);
        return;
      }

      rows.forEach((scheme) => {
        resultsEl.appendChild(card(scheme));
      });
    } catch (error) {
      setError(error.message || "Failed to fetch schemes.");
    } finally {
      setLoading(false);
    }
  }

  function resetFilters() {
    if (keywordEl) keywordEl.value = "";
    if (stateEl) stateEl.value = "";
    if (categoryEl) categoryEl.value = "";
    if (farmerTypeEl) farmerTypeEl.value = "";
    if (incomeLevelEl) incomeLevelEl.value = "";
    if (landSizeEl) landSizeEl.value = "";
    searchSchemes();
  }

  searchBtn.dataset.defaultText = searchBtn.textContent;
  searchBtn.addEventListener("click", searchSchemes);
  resetBtn?.addEventListener("click", resetFilters);

  loadOptions().then(searchSchemes);
})();
