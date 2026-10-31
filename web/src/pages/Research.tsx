import { useEffect, useMemo, useState } from "react";
import type { FormEvent } from "react";
import { Icon } from "../components/Icons";
import { PageHeader } from "../components/PageHeader";
import { useScrollReveal } from "../lib/motion";
import { formatInt } from "../lib/format";
import type { DatasetAnalysis, ResearchResult } from "../lib/types";

export type ResearchTab = "papers" | "landscape" | "gaps";

const TABS: { key: ResearchTab; label: string; icon: "research" | "gap" | "spark" }[] = [
  { key: "papers", label: "Pustaka", icon: "research" },
  { key: "landscape", label: "Lanskap & tren", icon: "spark" },
  { key: "gaps", label: "Research gap", icon: "gap" },
];

export function ResearchPage({
  data,
  result,
  busy,
  error,
  initialTab,
  onSearch,
}: {
  data: DatasetAnalysis | null;
  result: ResearchResult | null;
  busy: boolean;
  error: string;
  initialTab: ResearchTab;
  onSearch: (query: string, limit: number) => void;
}) {
  const [tab, setTab] = useState<ResearchTab>(initialTab);
  const [query, setQuery] = useState("");
  const [limit, setLimit] = useState(20);
  useScrollReveal(result);

  useEffect(() => {
    setTab(initialTab);
  }, [initialTab]);

  const suggestions = useMemo(() => buildSuggestions(data), [data]);

  useEffect(() => {
    if (!query && suggestions.length) setQuery(suggestions[0]);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [suggestions]);

  function submit(event: FormEvent) {
    event.preventDefault();
    const trimmed = query.trim();
    if (trimmed.length < 3) return;
    onSearch(trimmed, limit);
  }

  return (
    <div className="page">
      <PageHeader
        overline={initialTab === "gaps" ? "RESEARCH GAP" : "ACADEMIC RESEARCH"}
        title={initialTab === "gaps" ? "Peluang riset yang mungkin" : "Literatur & lanskap riset"}
        lead="Pencarian langsung ke OpenAlex, Crossref, Europe PMC, OpenAIRE, dan Semantic Scholar. Hasil digabung, dideduplikasi, lalu diperingkat."
        meta={
          <>
            <span className="meta-chip">5 sumber akademik</span>
            {data ? <span className="meta-chip">{data.filename}</span> : null}
            {result ? <span className="meta-chip">{formatInt(result.result_count)} paper</span> : null}
            {result?.cached ? <span className="meta-chip is-good">cache 10 menit</span> : null}
          </>
        }
      />

      <section className="panel research-search" data-reveal aria-labelledby="research-search-title">
        <div className="ml-head">
          <div>
            <span className="overline">PENCARIAN LITERATUR</span>
            <h2 id="research-search-title">Kata kunci riset</h2>
          </div>
          <span className="estimate-tag">DATA PUBLIK · BUKAN KLAIM ILMIAH</span>
        </div>

        <form className="search-bar" onSubmit={submit}>
          <label className="search-input">
            <Icon name="search" size={16} />
            <input
              value={query}
              onChange={(event) => setQuery(event.currentTarget.value)}
              placeholder="mis. student performance prediction"
              aria-label="Kata kunci riset"
              maxLength={300}
            />
          </label>
          <label className="search-limit">
            <span>Jumlah</span>
            <select value={limit} onChange={(event) => setLimit(Number(event.currentTarget.value))}>
              <option value={10}>10 paper</option>
              <option value={20}>20 paper</option>
              <option value={30}>30 paper</option>
              <option value={40}>40 paper</option>
            </select>
          </label>
          <button className="primary-button" type="submit" disabled={busy || query.trim().length < 3}>
            {busy ? (
              <>
                <span className="spinner" /> Mencari…
              </>
            ) : (
              <>
                <Icon name="search" size={16} /> Cari literatur
              </>
            )}
          </button>
        </form>

        {suggestions.length ? (
          <div className="suggestion-row">
            <span>Dari dataset kamu</span>
            {suggestions.slice(0, 5).map((suggestion) => (
              <button
                type="button"
                className="chip is-small"
                key={suggestion}
                onClick={() => setQuery(suggestion)}
              >
                {suggestion}
              </button>
            ))}
          </div>
        ) : null}

        {error ? (
          <p className="error-note" role="alert">
            <Icon name="alert" size={14} /> {error}
          </p>
        ) : null}

        {busy ? (
          <div className="research-loading" role="status">
            <span className="spinner" />
            <div>
              <b>Menghubungi sumber akademik…</b>
              <span>OpenAlex · Crossref · Europe PMC · OpenAIRE · Semantic Scholar</span>
            </div>
          </div>
        ) : null}
      </section>

      {result ? (
        <>
          <div className="provider-row">
            <span className="provider-label">PROVIDER</span>
            {Object.entries(result.provider_status).map(([name, status]) => (
              <span className={`provider-chip status-${status.toLowerCase()}`} key={name}>
                <i /> {name}
                <b>{statusLabel(status)}</b>
              </span>
            ))}
            {result.ranked ? <span className="provider-chip status-enabled"><i /> Peringkat dataset</span> : null}
          </div>

          <div className="tab-bar" role="tablist" aria-label="Bagian riset">
            {TABS.map((item) => (
              <button
                key={item.key}
                type="button"
                role="tab"
                aria-selected={tab === item.key}
                className={`tab-button${tab === item.key ? " is-active" : ""}`}
                onClick={() => setTab(item.key)}
              >
                <Icon name={item.icon} size={14} />
                {item.label}
                {item.key === "papers" ? <b>{result.result_count}</b> : null}
                {item.key === "gaps" && result.gaps.gaps.length ? <b>{result.gaps.gaps.length}</b> : null}
              </button>
            ))}
          </div>

          <div className="tab-panel" key={tab}>
            {tab === "papers" ? <PaperList result={result} /> : null}
            {tab === "landscape" ? <Landscape result={result} /> : null}
            {tab === "gaps" ? <Gaps result={result} /> : null}
          </div>
        </>
      ) : !busy ? (
        <section className="panel soft-empty" data-reveal>
          <Icon name="research" size={22} />
          <h3>Belum ada hasil pencarian</h3>
          <p>
            Tulis kata kunci di atas, atau pakai saran kata kunci dari dataset. DORA akan menggabungkan hasil dari lima
            indeks terbuka, membuang duplikat, lalu memeringkatkan berdasarkan kecocokan dengan dataset.
          </p>
        </section>
      ) : null}
    </div>
  );
}

function PaperList({ result }: { result: ResearchResult }) {
  if (!result.papers.length) {
    return (
      <section className="panel soft-empty">
        <Icon name="alert" size={20} />
        <h3>Tidak ada paper yang cocok</h3>
        <p>Coba kata kunci yang lebih umum, bahasa Inggris, atau kurangi jumlah kata.</p>
      </section>
    );
  }
  return (
    <section className="panel">
      <div className="table-head">
        <h3>Hasil pencarian</h3>
        <span>
          {formatInt(result.result_count)} paper unik dari {formatInt(result.candidate_count)} kandidat ·{" "}
          {result.ranked ? "diperingkat untuk dataset ini" : "urut sumber"}
        </span>
      </div>
      <ol className="paper-list">
        {result.papers.map((paper, index) => (
          <li className="paper-card" key={`${paper.title}-${index}`} style={{ ["--i" as string]: index }}>
            <span className="paper-index">{String(index + 1).padStart(2, "0")}</span>
            <div className="paper-body">
              <div className="paper-head">
                <a
                  className="paper-title"
                  href={paper.url || paper.doi || "#"}
                  target="_blank"
                  rel="noreferrer noopener"
                >
                  {paper.title}
                </a>
                {typeof paper.score === "number" ? (
                  <span className="paper-score" title="Skor relevansi terhadap dataset">
                    {Math.round(paper.score)}
                  </span>
                ) : null}
              </div>

              <p className="paper-authors">{formatAuthors(paper.authors)}</p>

              <div className="paper-meta">
                {paper.year ? <span>{paper.year}</span> : null}
                {paper.venue ? <span className="paper-venue" title={paper.venue}>{paper.venue}</span> : null}
                {typeof paper.citation_count === "number" ? (
                  <span>{formatInt(paper.citation_count)} sitasi</span>
                ) : null}
                {(paper.source ?? "")
                  .split(",")
                  .filter(Boolean)
                  .map((source) => (
                    <span className="chip is-small" key={source}>
                      {source.trim()}
                    </span>
                  ))}
                {paper.doi ? (
                  <a className="paper-link" href={paper.doi} target="_blank" rel="noreferrer noopener">
                    DOI
                  </a>
                ) : null}
              </div>

              {paper.reasons?.length ? (
                <ul className="paper-reasons">
                  {paper.reasons.slice(0, 3).map((reason) => (
                    <li key={reason}>{reason}</li>
                  ))}
                </ul>
              ) : null}

              {paper.abstract ? (
                <details className="paper-abstract">
                  <summary>Abstrak</summary>
                  <p>{paper.abstract}</p>
                </details>
              ) : null}
            </div>
          </li>
        ))}
      </ol>
    </section>
  );
}

function Landscape({ result }: { result: ResearchResult }) {
  const { landscape, trend } = result;
  const years = Object.entries(landscape.publication_years ?? {})
    .map(([year, count]) => ({ year: Number(year), count }))
    .sort((left, right) => left.year - right.year);
  const peak = Math.max(1, ...years.map((item) => item.count));

  return (
    <div className="split-grid">
      <section className="panel">
        <div className="table-head">
          <h3>Publikasi per tahun</h3>
          <span>
            {trend.year_range ? `${trend.year_range.start}–${trend.year_range.end}` : "—"} · puncak{" "}
            {trend.peak_year ?? "—"}
          </span>
        </div>
        <div className="panel-body">
          {years.length ? (
            <div className="year-chart" role="img" aria-label="Grafik publikasi per tahun">
              {years.map((item) => (
                <div className="year-bar" key={item.year}>
                  <span className="year-bar-count">{item.count}</span>
                  <i style={{ height: `${Math.max(6, (item.count / peak) * 100)}%` }} />
                  <span className="year-bar-label">{item.year}</span>
                </div>
              ))}
            </div>
          ) : (
            <p className="table-note">Tahun publikasi tidak tersedia pada metadata hasil ini.</p>
          )}

          <div className="trend-row">
            <div>
              <span className="pillar-label">Pertumbuhan</span>
              <b>
                {trend.growth?.percentage === undefined || trend.growth === null
                  ? "—"
                  : `${trend.growth.percentage > 0 ? "+" : ""}${Math.round(trend.growth.percentage)}%`}
              </b>
            </div>
            <div>
              <span className="pillar-label">Arah terbaru</span>
              <b>{directionLabel(trend.recent_direction)}</b>
            </div>
            <div>
              <span className="pillar-label">Total paper</span>
              <b>{formatInt(landscape.paper_count)}</b>
            </div>
          </div>
        </div>
      </section>

      <section className="panel">
        <div className="table-head">
          <h3>Metode & venue dominan</h3>
          <span>Heuristik dari judul dan abstrak</span>
        </div>
        <div className="panel-body">
          <CountList title="Metode" items={landscape.methods} />
          <CountList title="Venue" items={landscape.top_venues} />
          <CountList title="Sumber" items={landscape.sources} />
        </div>
      </section>
    </div>
  );
}

function CountList({ title, items }: { title: string; items: { name: string; count: number; percentage?: number }[] }) {
  if (!items?.length) return null;
  const peak = Math.max(1, ...items.map((item) => item.count));
  return (
    <div className="count-block">
      <span className="pillar-label">{title}</span>
      <ul className="bar-list">
        {items.slice(0, 6).map((item) => (
          <li key={item.name}>
            <div className="bar-list-head">
              <b title={item.name}>{item.name}</b>
              <span>{formatInt(item.count)}</span>
            </div>
            <div className="bar-track is-neutral">
              <i style={{ width: `${Math.max(4, (item.count / peak) * 100)}%` }} />
            </div>
          </li>
        ))}
      </ul>
    </div>
  );
}

function Gaps({ result }: { result: ResearchResult }) {
  const gaps = result.gaps?.gaps ?? [];
  return (
    <div className="gap-grid">
      <section className="panel">
        <div className="table-head">
          <h3>Kandidat research gap</h3>
          <span>
            {formatInt(result.gaps?.summary?.papers_analyzed ?? 0)} paper dianalisis ·{" "}
            {formatInt(result.gaps?.summary?.gap_count ?? gaps.length)} kandidat
          </span>
        </div>
        <div className="panel-body">
          {gaps.length ? (
            <div className="gap-list">
              {gaps.map((gap, index) => (
                <article className="gap-card" key={`${gap.type}-${index}`}>
                  <span className="gap-type">{gap.type.replaceAll("_", " ")}</span>
                  <h4>{gap.title}</h4>
                  <p>{gap.description}</p>
                  {gap.evidence ? (
                    <dl className="gap-evidence">
                      {Object.entries(gap.evidence).map(([key, value]) => (
                        <div key={key}>
                          <dt>{key.replaceAll("_", " ")}</dt>
                          <dd>{String(value)}</dd>
                        </div>
                      ))}
                    </dl>
                  ) : null}
                </article>
              ))}
            </div>
          ) : (
            <p className="table-note">Belum ada kandidat gap dari kumpulan paper ini.</p>
          )}
        </div>
      </section>

      <aside className="note-card">
        <span className="overline">CARA MEMBACA</span>
        <h3>Gap ini kandidat, bukan kesimpulan</h3>
        <p>
          Gap dihitung dari distribusi metode, tahun, dan dataset pada paper yang berhasil ditemukan saja. Jika pencarian
          sempit, gap bisa bias.
        </p>
        <ul>
          <li>Periksa manual 5–10 paper teratas sebelum mengklaim kebaruan.</li>
          <li>Gunakan kata kunci Inggris yang lebih luas untuk pembanding.</li>
          <li>Simpan bukti sitasi di laporan sebelum publikasi.</li>
        </ul>
      </aside>
    </div>
  );
}

function buildSuggestions(data: DatasetAnalysis | null): string[] {
  if (!data) return [];
  const representation = data.fingerprint.representation;
  const keywords = (representation.keywords ?? []).filter((keyword) => keyword.length > 2).slice(0, 4);
  const targets = representation.target_candidates ?? [];
  const suggestions: string[] = [];
  if (keywords.length) suggestions.push(keywords.slice(0, 3).join(" "));
  if (keywords.length >= 2) suggestions.push(`${keywords[0]} ${keywords[1]} prediction`);
  if (targets.length) suggestions.push(`${targets[0]} classification ${keywords[0] ?? ""}`.trim());
  if (keywords.length) suggestions.push(`${keywords[0]} machine learning`);
  return Array.from(new Set(suggestions)).slice(0, 5);
}

function formatAuthors(authors?: string[]): string {
  if (!authors?.length) return "Penulis tidak tercantum";
  if (authors.length <= 3) return authors.join(", ");
  return `${authors.slice(0, 3).join(", ")} +${authors.length - 3}`;
}

function statusLabel(status: string): string {
  const labels: Record<string, string> = {
    ENABLED: "aktif",
    RATE_LIMITED: "dibatasi",
    ERROR: "gagal",
    NOT_CONFIGURED: "tanpa key",
    DISABLED: "nonaktif",
  };
  return labels[status] ?? status.toLowerCase();
}

function directionLabel(direction: string): string {
  const labels: Record<string, string> = {
    RISING: "naik",
    FALLING: "turun",
    STABLE: "stabil",
    UNKNOWN: "tidak diketahui",
  };
  return labels[direction] ?? direction.toLowerCase();
}
