/**
 * Netlify Serverless Function — Dataset Search Proxy
 * Runs on Netlify's Node.js runtime (no CORS issues from server side).
 *
 * GET /.netlify/functions/datasets-search?q=QUERY&sources=huggingface,kaggle,openml,zenodo&limit=20
 */

const TIMEOUT_MS = 10_000;

// ── Helpers ──────────────────────────────────────────────────────────────────

function stripHtml(str) {
  return str ? str.replace(/<[^>]+>/g, "").trim() : "";
}

function truncate(str, n = 200) {
  if (!str) return "";
  const s = stripHtml(str);
  return s.length > n ? s.slice(0, n) + "…" : s;
}

async function fetchWithTimeout(url, opts = {}) {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), TIMEOUT_MS);
  try {
    const res = await fetch(url, { ...opts, signal: ctrl.signal });
    return res;
  } finally {
    clearTimeout(timer);
  }
}

// ── Source Fetchers ──────────────────────────────────────────────────────────

async function searchHuggingFace(query, limit) {
  try {
    const url = `https://huggingface.co/api/datasets?search=${encodeURIComponent(query)}&limit=${limit}&sort=downloads`;
    const res = await fetchWithTimeout(url);
    if (!res.ok) return [];
    const data = await res.json();
    if (!Array.isArray(data)) return [];
    return data.map((d) => ({
      id: `hf-${d.id || d.modelId}`,
      source: "huggingface",
      name: d.id || d.modelId || "HuggingFace Dataset",
      description: truncate(d.description || d.cardData?.description || ""),
      tags: (d.tags || []).filter((t) => !t.startsWith("language:") && !t.startsWith("license:")).slice(0, 6),
      downloads: d.downloads ?? null,
      likes: d.likes ?? null,
      url: `https://huggingface.co/datasets/${d.id || d.modelId}`,
    }));
  } catch {
    return [];
  }
}

async function searchKaggle(query, limit) {
  try {
    // Kaggle public API v1 — no auth required for listing (no `limit` param)
    const url = `https://www.kaggle.com/api/v1/datasets/list?search=${encodeURIComponent(query)}&page=1&sortBy=relevance`;
    const res = await fetchWithTimeout(url);
    if (!res.ok) return [];
    const data = await res.json();
    if (!Array.isArray(data)) return [];
    return data.slice(0, limit).map((d) => ({
      id: `kg-${d.ref || d.id}`,
      source: "kaggle",
      name: d.title || d.ref || "Kaggle Dataset",
      description: truncate(d.subtitle || d.description || ""),
      tags: (d.tags || []).map((t) => (typeof t === "object" ? t.name : String(t))).slice(0, 5),
      downloads: d.downloadCount ?? null,
      likes: d.voteCount ?? null,
      url: d.url || `https://www.kaggle.com/datasets/${d.ref}`,
    }));
  } catch {
    return [];
  }
}

async function searchZenodo(query, limit) {
  try {
    const url = `https://zenodo.org/api/records?q=${encodeURIComponent(query)}&type=dataset&size=${limit}&sort=mostrecent`;
    const res = await fetchWithTimeout(url);
    if (!res.ok) return [];
    const data = await res.json();
    const hits = data.hits?.hits ?? [];
    return hits.map((r) => ({
      id: `zen-${r.id}`,
      source: "zenodo",
      name: r.metadata?.title || r.title || `Zenodo #${r.id}`,
      description: truncate(r.metadata?.description || ""),
      tags: [
        ...(r.metadata?.keywords?.slice(0, 4) ?? []),
        ...(r.metadata?.subjects?.map((s) => s.term).slice(0, 2) ?? []),
      ].slice(0, 6),
      downloads: r.stats?.downloads ?? null,
      likes: null,
      url: r.links?.html || `https://zenodo.org/record/${r.id}`,
    }));
  } catch {
    return [];
  }
}

async function searchOpenML(query, limit) {
  try {
    const firstWord = query.trim().split(/\s+/)[0]?.replace(/[^a-zA-Z0-9_]/g, "") || query;
    if (!firstWord) return [];
    const url = `https://www.openml.org/api/v1/json/data/list/data_name/${encodeURIComponent(firstWord)}/limit/${limit}?api_key=`;
    const res = await fetchWithTimeout(url);
    if (!res.ok) return [];
    const data = await res.json();
    const datasets = data.data?.dataset ?? [];
    return datasets.map((d) => ({
      id: `oml-${d.did}`,
      source: "openml",
      name: d.name || `OpenML #${d.did}`,
      description: d.description
        ? truncate(d.description)
        : `${d.NumberOfInstances ?? "?"} baris · ${d.NumberOfFeatures ?? "?"} fitur${d.format ? ` · ${d.format}` : ""}`,
      tags: d.format ? [d.format] : [],
      downloads: null,
      likes: null,
      url: `https://www.openml.org/d/${d.did}`,
    }));
  } catch {
    return [];
  }
}

// ── Handler ──────────────────────────────────────────────────────────────────

export async function handler(event) {
  // CORS headers
  const headers = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type",
    "Content-Type": "application/json",
  };

  if (event.httpMethod === "OPTIONS") {
    return { statusCode: 204, headers, body: "" };
  }

  const params = event.queryStringParameters || {};
  const query = (params.q || "").trim();
  const limitRaw = parseInt(params.limit || "20", 10);
  const limit = Math.min(Math.max(limitRaw, 1), 50);
  const sourcesParam = params.sources || "huggingface,kaggle,openml,zenodo";
  const sources = new Set(sourcesParam.split(",").map((s) => s.trim()));

  if (!query) {
    return {
      statusCode: 400,
      headers,
      body: JSON.stringify({ error: "Parameter 'q' is required" }),
    };
  }

  // Parallel fetch from all requested sources
  const tasks = [];
  if (sources.has("huggingface")) tasks.push(searchHuggingFace(query, limit));
  if (sources.has("kaggle")) tasks.push(searchKaggle(query, limit));
  if (sources.has("zenodo")) tasks.push(searchZenodo(query, limit));
  if (sources.has("openml")) tasks.push(searchOpenML(query, limit));

  const settled = await Promise.allSettled(tasks);
  const results = settled
    .filter((r) => r.status === "fulfilled")
    .flatMap((r) => r.value)
    .sort((a, b) => (b.downloads ?? 0) - (a.downloads ?? 0))
    .slice(0, limit);

  return {
    statusCode: 200,
    headers,
    body: JSON.stringify({ query, count: results.length, results }),
  };
}
