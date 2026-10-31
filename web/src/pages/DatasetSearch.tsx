import { useState, useRef, useCallback, useEffect } from "react";
import type { FormEvent, KeyboardEvent } from "react";
import { Icon } from "../components/Icons";
import { PageHeader } from "../components/PageHeader";
import { useScrollReveal } from "../lib/motion";

/* ── Types ─────────────────────────────────────────────────── */
type Source = "huggingface" | "kaggle" | "openml" | "zenodo";

interface KaggleDataset {
  id?: number | string;
  ref?: string;
  title?: string;
  subtitle?: string;
  description?: string;
  url?: string;
  downloadCount?: number;
  voteCount?: number;
  tags?: { name: string }[] | string[];
}

interface ZenodoRecord {
  id: number;
  title?: string;
  metadata?: {
    title?: string;
    description?: string;
    keywords?: string[];
    subjects?: { term: string }[];
  };
  stats?: { downloads?: number; views?: number };
  links?: { html?: string };
}

interface OpenMLDataset {
  did: number;
  name?: string;
  description?: string;
  NumberOfFeatures?: number;
  NumberOfInstances?: number;
  format?: string;
}

interface ResultCard {
  id: string;
  source: Source;
  name: string;
  description: string;
  tags: string[];
  downloads: number | null;
  likes: number | null;
  url: string;
}

type SearchState = "idle" | "loading" | "results" | "error";

// In production (Netlify), use relative URL so the Netlify Function handles it.
// Locally, use localhost:8000 (FastAPI backend).
const apiBase = (() => {
  if (import.meta.env.VITE_API_URL) return import.meta.env.VITE_API_URL.replace(/\/$/, "");
  if (typeof window !== "undefined") {
    const host = window.location.hostname;
    if (host === "localhost" || host === "127.0.0.1") return "http://localhost:8000";
    // Running on Netlify or any deployed domain → use relative path (Netlify redirect handles it)
    return "";
  }
  return "http://localhost:8000";
})();

/* ── Source config & suggestions ────────────────────────────── */
const SOURCE_CONFIG: { id: Source; label: string; color: string }[] = [
  { id: "huggingface", label: "Hugging Face", color: "#ffcc4d" },
  { id: "kaggle", label: "Kaggle", color: "#20beff" },
  { id: "openml", label: "OpenML", color: "#1f9d6b" },
  { id: "zenodo", label: "Zenodo", color: "#7c5cbf" },
];

const SUGGESTED_TOPICS = [
  "Heart Disease",
  "Customer Churn",
  "Climate Change",
  "Sentiment Analysis",
  "Student Performance",
  "Housing Prices",
  "Credit Risk",
  "Traffic Prediction",
];

/* ── Fallback Fetchers (Browser Direct with CORS) ───────────── */
interface HFDatasetItem {
  id?: string;
  _id?: string;
  description?: string;
  tags?: string[];
  downloads?: number;
  likes?: number;
}

async function fetchHuggingFaceDirect(query: string, limit = 15): Promise<ResultCard[]> {
  try {
    const url = `https://huggingface.co/api/datasets?search=${encodeURIComponent(query)}&limit=${limit}&sort=downloads`;
    const res = await fetch(url);
    if (!res.ok) return [];
    const data: HFDatasetItem[] = await res.json();
    if (!Array.isArray(data)) return [];
    return data.map((d) => ({
      id: `hf-${d.id || d._id}`,
      source: "huggingface",
      name: d.id || "Hugging Face Dataset",
      description: (d.description || "").replace(/<[^>]+>/g, "").replace(/\s+/g, " ").trim().slice(0, 160),
      tags: (d.tags || []).filter((t) => !t.startsWith("language:") && !t.startsWith("license:") && !t.startsWith("region:")).slice(0, 5),
      downloads: d.downloads ?? null,
      likes: d.likes ?? null,
      url: `https://huggingface.co/datasets/${d.id}`,
    }));
  } catch {
    return [];
  }
}

async function fetchKaggleDirect(query: string, limit = 15): Promise<ResultCard[]> {
  try {
    const url = `https://www.kaggle.com/api/v1/datasets/list?search=${encodeURIComponent(query)}&page=1&sortBy=relevance`;
    const res = await fetch(url);
    if (!res.ok) return [];
    const data = await res.json();
    if (!Array.isArray(data)) return [];
    return data.slice(0, limit).map((d: KaggleDataset) => ({
      id: `kg-${d.ref || d.id}`,
      source: "kaggle",
      name: d.title || d.ref || "Kaggle Dataset",
      description: d.subtitle || d.description || "",
      tags: (d.tags || []).map((t) => (typeof t === "object" ? t.name : String(t))).slice(0, 5),
      downloads: d.downloadCount ?? null,
      likes: d.voteCount ?? null,
      url: d.url || `https://www.kaggle.com/datasets/${d.ref}`,
    }));
  } catch {
    return [];
  }
}

async function fetchZenodoDirect(query: string, limit = 10): Promise<ResultCard[]> {
  try {
    const url = `https://zenodo.org/api/records?q=${encodeURIComponent(query)}&type=dataset&size=${limit}&sort=mostrecent`;
    const res = await fetch(url);
    if (!res.ok) return [];
    const data: { hits?: { hits?: ZenodoRecord[] } } = await res.json();
    const hits = data.hits?.hits ?? [];
    return hits.map((r) => ({
      id: `zen-${r.id}`,
      source: "zenodo",
      name: r.metadata?.title || r.title || `Zenodo #${r.id}`,
      description: r.metadata?.description
        ? r.metadata.description.replace(/<[^>]+>/g, "").slice(0, 160)
        : "",
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

async function fetchOpenMLDirect(query: string, limit = 10): Promise<ResultCard[]> {
  try {
    const firstWord = query.trim().split(/\s+/)[0]?.replace(/[^a-zA-Z0-9_]/g, "") || query;
    if (!firstWord) return [];
    const url = `https://www.openml.org/api/v1/json/data/list/data_name/${encodeURIComponent(firstWord)}/limit/${limit}?api_key=`;
    const res = await fetch(url);
    if (!res.ok) return [];
    const data: { data?: { dataset?: OpenMLDataset[] } } = await res.json();
    const datasets = data.data?.dataset ?? [];
    return datasets.map((d) => ({
      id: `oml-${d.did}`,
      source: "openml",
      name: d.name || `OpenML #${d.did}`,
      description: d.description
        ? d.description.replace(/<[^>]+>/g, "").slice(0, 160)
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

/* ── Helpers ────────────────────────────────────────────────── */
function formatNum(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}K`;
  return String(n);
}

function sourceBadge(s: Source) {
  const cfg = SOURCE_CONFIG.find((x) => x.id === s) || { label: s, color: "#4f8cff" };
  return { label: cfg.label, color: cfg.color };
}

/* ── Skeleton ───────────────────────────────────────────────── */
function SkeletonCard() {
  return (
    <div className="dsearch-result-card is-skeleton" aria-hidden>
      <div className="drc-top">
        <span className="drc-badge skeleton-block" style={{ width: 80 }} />
        <span className="drc-name skeleton-block" style={{ width: "55%" }} />
      </div>
      <span className="drc-desc skeleton-block" style={{ width: "90%" }} />
      <span className="drc-desc skeleton-block" style={{ width: "70%" }} />
    </div>
  );
}

/* ── Copy Button ────────────────────────────────────────────── */
function CopyButton({ url }: { url: string }) {
  const [copied, setCopied] = useState(false);

  function copy() {
    navigator.clipboard.writeText(url);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  }

  return (
    <button
      type="button"
      className={`drc-copy ${copied ? "is-copied" : ""}`}
      onClick={copy}
      title="Salin tautan dataset"
      aria-label="Salin tautan dataset"
    >
      <Icon name={copied ? "check" : "editor"} size={13} />
      <span>{copied ? "Tersalin" : "Salin Link"}</span>
    </button>
  );
}

/* ── Main Component ─────────────────────────────────────────── */
export function DatasetSearchPage({ onGoToUpload }: { onGoToUpload?: () => void }) {
  useScrollReveal(null);

  const [query, setQuery] = useState("");
  const [activeSources, setActiveSources] = useState<Set<Source>>(
    new Set(["huggingface", "kaggle", "openml", "zenodo"])
  );
  const [state, setState] = useState<SearchState>("idle");
  const [results, setResults] = useState<ResultCard[]>([]);
  const [errorMsg, setErrorMsg] = useState("");
  const [lastQuery, setLastQuery] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);

  const toggleSource = useCallback((id: Source) => {
    setActiveSources((prev) => {
      const next = new Set(prev);
      if (next.has(id) && next.size > 1) next.delete(id);
      else next.add(id);
      return next;
    });
  }, []);

  const doSearch = useCallback(async (q: string, sources: Set<Source>) => {
    const clean = q.trim();
    if (!clean) return;
    setState("loading");
    setResults([]);
    setErrorMsg("");
    setLastQuery(clean);

    let collected: ResultCard[] = [];

    // 1. Try backend search API first (has no CORS limitations and searches all sources)
    try {
      const sourcesParam = Array.from(sources).join(",");
      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), 8000);
      try {
        const res = await fetch(
          `${apiBase}/api/datasets/search?q=${encodeURIComponent(clean)}&sources=${sourcesParam}&limit=20`,
          { signal: controller.signal }
        );
        if (res.ok) {
          const data = await res.json();
          if (Array.isArray(data.results) && data.results.length > 0) {
            collected = data.results;
          }
        }
      } finally {
        clearTimeout(timer);
      }
    } catch {
      // Backend unavailable or network error, proceed to direct fetch
    }

    // 2. If backend yielded nothing or was unreachable, fallback to direct browser fetch
    if (collected.length === 0) {
      const fetchers: Promise<ResultCard[]>[] = [];
      if (sources.has("huggingface")) fetchers.push(fetchHuggingFaceDirect(clean, 15));
      if (sources.has("kaggle")) fetchers.push(fetchKaggleDirect(clean, 15));
      if (sources.has("zenodo")) fetchers.push(fetchZenodoDirect(clean, 10));
      if (sources.has("openml")) fetchers.push(fetchOpenMLDirect(clean, 10));

      const settled = await Promise.allSettled(fetchers);
      for (const r of settled) {
        if (r.status === "fulfilled") {
          collected.push(...r.value);
        }
      }
    }

    if (collected.length > 0) {
      collected.sort((a, b) => (b.downloads ?? 0) - (a.downloads ?? 0));
      setResults(collected);
      setState("results");
    } else {
      setResults([]);
      setState("results");
    }
  }, []);

  const searchedRef = useRef(false);
  useEffect(() => {
    if (searchedRef.current) return;
    const qParam = new URLSearchParams(window.location.search).get("q");
    if (qParam && qParam.trim()) {
      searchedRef.current = true;
      const clean = qParam.trim();
      setQuery(clean);
      doSearch(clean, activeSources);
    }
  }, [doSearch, activeSources]);

  function handleSubmit(e: FormEvent) {
    e.preventDefault();
    doSearch(query, activeSources);
  }

  function handleKeyDown(e: KeyboardEvent<HTMLInputElement>) {
    if (e.key === "Enter") {
      e.preventDefault();
      doSearch(query, activeSources);
    }
  }

  function handleTopicClick(topic: string) {
    setQuery(topic);
    doSearch(topic, activeSources);
    inputRef.current?.focus();
  }

  return (
    <div className="page">
      <PageHeader
        overline="DATASET SEARCH"
        title="Cari Dataset"
        lead="Telusuri dataset publik langsung dari Hugging Face, Kaggle, OpenML, dan Zenodo."
        actions={
          onGoToUpload ? (
            <button className="primary-button" type="button" onClick={onGoToUpload}>
              <Icon name="upload" size={15} /> Unggah CSV
            </button>
          ) : undefined
        }
      />

      {/* ── Search bar section ──────────────────────────────── */}
      <section className="dsearch-bar-section" data-reveal>
        <form className="dsearch-bar" onSubmit={handleSubmit}>
          <label className="dsearch-input">
            <Icon name="search" size={17} />
            <input
              ref={inputRef}
              value={query}
              onChange={(e) => setQuery(e.currentTarget.value)}
              onKeyDown={handleKeyDown}
              placeholder="Cari dataset: iris, climate, customer churn, medical, sentiment…"
              aria-label="Kata kunci dataset"
              maxLength={200}
              autoFocus
            />
            {query && (
              <button
                type="button"
                className="dsearch-clear"
                onClick={() => { setQuery(""); inputRef.current?.focus(); }}
                aria-label="Hapus kata kunci"
              >
                ×
              </button>
            )}
          </label>
          <button
            type="submit"
            className="primary-button dsearch-submit"
            disabled={!query.trim() || state === "loading"}
          >
            {state === "loading" ? "Mencari…" : "Cari"}
          </button>
        </form>

        {/* Source pill toggles */}
        <div className="dsearch-pills" role="group" aria-label="Sumber dataset">
          {SOURCE_CONFIG.map((src) => (
            <button
              key={src.id}
              type="button"
              className={`dsearch-pill ${activeSources.has(src.id) ? "is-active" : ""}`}
              style={{ "--pill-color": src.color } as React.CSSProperties}
              onClick={() => toggleSource(src.id)}
              aria-pressed={activeSources.has(src.id)}
            >
              <span
                className="dsearch-pill-dot"
                style={{ background: src.color }}
              />
              {src.label}
            </button>
          ))}
        </div>
      </section>

      {/* ── Results area ───────────────────────────────────── */}
      {state === "loading" && (
        <div className="dsearch-results-grid" data-reveal>
          {Array.from({ length: 6 }).map((_, i) => <SkeletonCard key={i} />)}
        </div>
      )}

      {state === "results" && results.length === 0 && (
        <div className="dsearch-empty" data-reveal>
          <Icon name="dataset" size={36} />
          <p>Tidak ada hasil untuk <b>"{lastQuery}"</b>. Coba kata kunci lain atau aktifkan sumber lainnya.</p>
        </div>
      )}

      {state === "results" && results.length > 0 && (
        <>
          <p className="dsearch-count" data-reveal>
            <b>{results.length}</b> dataset ditemukan untuk "<b>{lastQuery}</b>"
          </p>
          <div className="dsearch-results-grid" data-reveal>
            {results.map((card, i) => (
              <DatasetResultCard
                key={card.id}
                card={card}
                delay={i * 30}
              />
            ))}
          </div>
        </>
      )}

      {state === "error" && (
        <div className="dsearch-error" data-reveal>
          <Icon name="shield" size={20} />
          <p>{errorMsg}</p>
          <button
            type="button"
            className="ghost-button"
            onClick={() => doSearch(lastQuery, activeSources)}
          >
            Coba Lagi
          </button>
        </div>
      )}

      {state === "idle" && (
        <div className="dsearch-idle" data-reveal>
          <div className="dsearch-idle-topics">
            <span className="dsearch-idle-heading">Topik Populer:</span>
            <div className="dsearch-topic-pills">
              {SUGGESTED_TOPICS.map((topic) => (
                <button
                  key={topic}
                  type="button"
                  className="dsearch-topic-pill"
                  onClick={() => handleTopicClick(topic)}
                >
                  <Icon name="search" size={12} />
                  {topic}
                </button>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

/* ── Result Card ────────────────────────────────────────────── */
function DatasetResultCard({ card, delay }: { card: ResultCard; delay: number }) {
  const badge = sourceBadge(card.source);
  const shortDesc = card.description.length > 140
    ? card.description.slice(0, 137) + "…"
    : card.description;

  return (
    <article
      className="dsearch-result-card"
      style={{ "--reveal-delay": `${delay}ms` } as React.CSSProperties}
    >
      <div className="drc-top">
        <span
          className="drc-badge"
          style={{ "--badge-color": badge.color } as React.CSSProperties}
        >
          {badge.label}
        </span>
        {card.downloads !== null && (
          <span className="drc-stat">
            ↓ {formatNum(card.downloads)}
          </span>
        )}
        {card.likes !== null && (
          <span className="drc-stat">♥ {formatNum(card.likes)}</span>
        )}
      </div>

      <h3 className="drc-name">{card.name}</h3>

      {shortDesc && <p className="drc-desc">{shortDesc}</p>}

      {card.tags.length > 0 && (
        <div className="drc-tags">
          {card.tags.slice(0, 5).map((tag) => (
            <span key={tag} className="drc-tag">{tag}</span>
          ))}
        </div>
      )}

      <div className="drc-actions">
        <a
          href={card.url}
          target="_blank"
          rel="noopener noreferrer"
          className="drc-link"
          aria-label={`Buka ${card.name} di ${badge.label}`}
        >
          Buka di {badge.label} <Icon name="arrow" size={12} />
        </a>
        <CopyButton url={card.url} />
      </div>
    </article>
  );
}
