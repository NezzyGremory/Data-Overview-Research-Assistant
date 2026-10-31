import { useMemo, useState } from "react";
import { Icon } from "../components/Icons";
import { PageHeader } from "../components/PageHeader";
import { useScrollReveal } from "../lib/motion";
import { formatInt, formatNumber, formatPercent, shortName } from "../lib/format";
import type { ColumnStat, DatasetAnalysis, QualityItem } from "../lib/types";

type TabKey = "diagnosis" | "statistik" | "distribusi" | "kosong" | "outlier" | "duplikat" | "korelasi" | "fingerprint";

const TABS: {
  key: TabKey;
  label: string;
  icon: "shield" | "table" | "analysis" | "alert" | "node" | "search" | "file";
}[] = [
  { key: "diagnosis", label: "Diagnosis", icon: "shield" },
  { key: "statistik", label: "Statistik", icon: "table" },
  { key: "distribusi", label: "Distribusi", icon: "analysis" },
  { key: "kosong", label: "Nilai kosong", icon: "alert" },
  { key: "outlier", label: "Outlier", icon: "analysis" },
  { key: "duplikat", label: "Duplikat", icon: "file" },
  { key: "korelasi", label: "Korelasi", icon: "node" },
  { key: "fingerprint", label: "Fingerprint", icon: "search" },
];

export function AnalysisPage({ data }: { data: DatasetAnalysis }) {
  const [tab, setTab] = useState<TabKey>("diagnosis");
  useScrollReveal(tab);

  const critical = data.quality.filter((item) => item.severity === "critical").length;
  const warning = data.quality.filter((item) => item.severity === "warning").length;

  return (
    <div className="page">
      <PageHeader
        overline="ANALYSIS"
        title="Diagnosis & statistik"
        lead="Empat pilar diagnosis untuk setiap isu, ditambah statistik deskriptif, korelasi, dan fingerprint dataset."
        meta={
          <>
            <span className="meta-chip">{formatInt(data.profile.rows)} baris</span>
            <span className="meta-chip">{formatInt(data.profile.columns)} kolom</span>
            <span className={`meta-chip${critical ? " is-critical" : warning ? " is-warning" : " is-good"}`}>
              {critical ? `${critical} kritis` : warning ? `${warning} peringatan` : "Tidak ada isu besar"}
            </span>
            {data.limits.truncated ? (
              <span className="meta-chip">{data.limits.columns_analyzed} kolom dianalisis</span>
            ) : null}
          </>
        }
      />

      <div className="tab-bar" role="tablist" aria-label="Bagian analisis">
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
            {item.key === "diagnosis" && data.quality.length ? <b>{data.quality.length}</b> : null}
          </button>
        ))}
      </div>

      <div className="tab-panel" key={tab}>
        {tab === "diagnosis" ? <Diagnosis data={data} /> : null}
        {tab === "statistik" ? <Statistics data={data} /> : null}
        {tab === "distribusi" ? <Distributions data={data} /> : null}
        {tab === "kosong" ? <MissingPanel data={data} /> : null}
        {tab === "outlier" ? <OutlierPanel data={data} /> : null}
        {tab === "duplikat" ? <Duplicates data={data} /> : null}
        {tab === "korelasi" ? <CorrelationPanel data={data} /> : null}
        {tab === "fingerprint" ? <FingerprintPanel data={data} /> : null}
      </div>
    </div>
  );
}

function Diagnosis({ data }: { data: DatasetAnalysis }) {
  if (!data.quality.length) {
    return (
      <section className="panel soft-empty">
        <Icon name="check" size={20} />
        <h3>Tidak ada isu kualitas yang terdeteksi</h3>
        <p>Semua pemeriksaan otomatis (duplikat, nilai kosong, outlier, identifier, kolom konstan) lolos.</p>
      </section>
    );
  }
  return (
    <div className="diagnosis-list">
      {data.quality.map((item, index) => (
        <DiagnosisCard item={item} key={`${item.category}-${index}`} index={index} />
      ))}
    </div>
  );
}

function DiagnosisCard({ item, index }: { item: QualityItem; index: number }) {
  return (
    <article className={`diagnosis severity-${item.severity}`} style={{ ["--i" as string]: index }}>
      <header className="diagnosis-head">
        <span className="severity-pill">
          <i />
          {item.severity === "critical" ? "KRITIS" : item.severity === "warning" ? "PERINGATAN" : "INFO"}
        </span>
        <span className="diagnosis-cat">{item.category.replaceAll("_", " ")}</span>
        <h3>{item.title}</h3>
      </header>
      <div className="diagnosis-grid">
        <div>
          <span className="pillar-label">1 · Masalah</span>
          <p>{item.problem}</p>
        </div>
        <div>
          <span className="pillar-label">2 · Besar masalah</span>
          <p>{item.magnitude}</p>
        </div>
        <div>
          <span className="pillar-label">3 · Dampak</span>
          <p>{item.impact}</p>
        </div>
        <div>
          <span className="pillar-label">4 · Opsi penanganan</span>
          <ul>
            {item.options.map((option) => (
              <li key={option}>{option}</li>
            ))}
          </ul>
        </div>
      </div>
      {item.affected_columns.length ? (
        <div className="affected">
          <span>Kolom terkait</span>
          {item.affected_columns.slice(0, 8).map((column) => (
            <span className="chip is-small" key={column}>
              {column}
            </span>
          ))}
          {item.affected_columns.length > 8 ? <span className="chip is-small">+{item.affected_columns.length - 8}</span> : null}
        </div>
      ) : null}
    </article>
  );
}

type SortKey =
  | "column"
  | "unique_count"
  | "missing_percentage"
  | "mean"
  | "median"
  | "std"
  | "min"
  | "max"
  | "skewness";

function Statistics({ data }: { data: DatasetAnalysis }) {
  const [query, setQuery] = useState("");
  const [sort, setSort] = useState<{ key: SortKey; dir: "asc" | "desc" }>({ key: "column", dir: "asc" });

  const rows = useMemo(() => {
    const term = query.trim().toLowerCase();
    const filtered = data.statistics.filter((stat) => !term || stat.column.toLowerCase().includes(term));
    const sorted = [...filtered].sort((left, right) => {
      if (sort.key === "column") {
        return left.column.localeCompare(right.column);
      }
      const a = left[sort.key] as number | null | undefined;
      const b = right[sort.key] as number | null | undefined;
      if (a === null || a === undefined) return b === null || b === undefined ? 0 : 1;
      if (b === null || b === undefined) return -1;
      return (a as number) - (b as number);
    });
    return sort.dir === "asc" || sort.key === "column" ? sorted : sorted.reverse();
  }, [data.statistics, query, sort]);

  function toggle(key: SortKey) {
    setSort((current) =>
      current.key === key
        ? { key, dir: current.dir === "asc" ? "desc" : "asc" }
        : { key, dir: key === "column" ? "asc" : "desc" },
    );
  }

  return (
    <section className="panel">
      <div className="table-head">
        <h3>Statistik deskriptif</h3>
        <label className="search-field">
          <Icon name="search" size={14} />
          <input
            value={query}
            onChange={(event) => setQuery(event.currentTarget.value)}
            placeholder="Cari kolom…"
            aria-label="Cari kolom"
          />
        </label>
      </div>
      <div className="table-scroll">
        <table className="data-table">
          <thead>
            <tr>
              <SortableTh label="Kolom" sortKey="column" sort={sort} onSort={toggle} />
              <th>Tipe</th>
              <SortableTh label="Unik" sortKey="unique_count" sort={sort} onSort={toggle} numeric />
              <SortableTh label="Kosong" sortKey="missing_percentage" sort={sort} onSort={toggle} numeric />
              <SortableTh label="Rata-rata" sortKey="mean" sort={sort} onSort={toggle} numeric />
              <SortableTh label="Median" sortKey="median" sort={sort} onSort={toggle} numeric />
              <SortableTh label="Std" sortKey="std" sort={sort} onSort={toggle} numeric />
              <SortableTh label="Min" sortKey="min" sort={sort} onSort={toggle} numeric />
              <SortableTh label="Maks" sortKey="max" sort={sort} onSort={toggle} numeric />
              <SortableTh label="Kemiringan" sortKey="skewness" sort={sort} onSort={toggle} numeric />
            </tr>
          </thead>
          <tbody>
            {rows.map((stat) => (
              <tr key={stat.column}>
                <td className="strong-cell" title={stat.column}>
                  {stat.column}
                </td>
                <td>
                  <span className="type-tag">{stat.dtype}</span>
                </td>
                <td className="num">{formatInt(stat.unique_count)}</td>
                <td className={`num${stat.missing_percentage > 0 ? " is-warn" : ""}`}>{formatPercent(stat.missing_percentage)}</td>
                <td className="num">{isNumeric(stat) ? formatNumber(stat.mean) : <TopValue stat={stat} />}</td>
                <td className="num">{isNumeric(stat) ? formatNumber(stat.median) : "—"}</td>
                <td className="num">{isNumeric(stat) ? formatNumber(stat.std) : "—"}</td>
                <td className="num">{isNumeric(stat) ? formatNumber(stat.min) : "—"}</td>
                <td className="num">{isNumeric(stat) ? formatNumber(stat.max) : "—"}</td>
                <td className="num">{isNumeric(stat) ? formatNumber(stat.skewness) : "—"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {!rows.length ? <p className="table-note">Tidak ada kolom yang cocok dengan pencarian.</p> : null}
    </section>
  );
}

function SortableTh({
  label,
  sortKey,
  sort,
  onSort,
  numeric = false,
}: {
  label: string;
  sortKey: SortKey;
  sort: { key: SortKey; dir: "asc" | "desc" };
  onSort: (key: SortKey) => void;
  numeric?: boolean;
}) {
  const active = sort.key === sortKey;
  return (
    <th className={numeric ? "num" : undefined} aria-sort={active ? (sort.dir === "asc" ? "ascending" : "descending") : "none"}>
      <button type="button" className={`sort-button${active ? " is-active" : ""}`} onClick={() => onSort(sortKey)}>
        {label}
        <span aria-hidden="true">{active ? (sort.dir === "asc" ? "↑" : "↓") : "↕"}</span>
      </button>
    </th>
  );
}

function Distributions({ data }: { data: DatasetAnalysis }) {
  const columns = data.distributions ?? [];
  const [column, setColumn] = useState(columns[0]?.column ?? "");
  const current = columns.find((item) => item.column === column) ?? columns[0];

  if (!columns.length) {
    return (
      <section className="panel soft-empty">
        <Icon name="analysis" size={20} />
        <h3>Tidak ada distribusi yang bisa dihitung</h3>
        <p>Semua kolom pada batas analisis kosong atau bertipe tidak didukung.</p>
      </section>
    );
  }

  const total = current ? current.bins.reduce((sum, bin) => sum + bin.count, 0) : 0;
  const peak = current ? Math.max(1, ...current.bins.map((bin) => bin.count)) : 1;

  return (
    <div className="split-grid">
      <section className="panel">
        <div className="table-head">
          <h3>Sebaran nilai</h3>
          <label className="search-limit">
            <span>Kolom</span>
            <select value={current?.column ?? ""} onChange={(event) => setColumn(event.currentTarget.value)}>
              {columns.map((item) => (
                <option value={item.column} key={item.column}>
                  {item.column} · {item.kind === "numeric" ? "numerik" : "kategori"}
                </option>
              ))}
            </select>
          </label>
        </div>
        <div className="panel-body">
          <div className="distribution-head">
            <span className={`type-tag${current?.kind === "numeric" ? " is-num" : ""}`}>
              {current?.kind === "numeric" ? "numerik" : "kategori"}
            </span>
            <span>{formatInt(total)} nilai terhitung</span>
            {current?.missing ? <span className="is-warn">{formatInt(current.missing)} kosong</span> : null}
          </div>
          <ul className="bar-list">
            {current?.bins.map((bin) => (
              <li key={bin.label}>
                <div className="bar-list-head">
                  <b title={bin.label}>{bin.label}</b>
                  <span>
                    {formatInt(bin.count)}
                    {total ? ` · ${formatPercent((bin.count / total) * 100)}` : ""}
                  </span>
                </div>
                <div className="bar-track">
                  <i style={{ width: `${Math.max(2, (bin.count / peak) * 100)}%` }} />
                </div>
              </li>
            ))}
          </ul>
          {current?.kind === "categorical" ? (
            <p className="table-note">Menampilkan maksimal 8 nilai terbanyak. Sampel 20.000 baris dipakai untuk dataset besar.</p>
          ) : (
            <p className="table-note">Histogram memakai 10 bin dengan rentang sama dari nilai minimum sampai maksimum.</p>
          )}
        </div>
      </section>

      <section className="panel">
        <div className="table-head">
          <h3>Ringkasan semua kolom</h3>
          <span>{formatInt(columns.length)} kolom</span>
        </div>
        <div className="panel-body">
          <ul className="mini-list">
            {columns.map((item) => {
              const itemTotal = item.bins.reduce((sum, bin) => sum + bin.count, 0);
              const top = [...item.bins].sort((left, right) => right.count - left.count)[0];
              return (
                <li key={item.column}>
                  <button
                    type="button"
                    className={`mini-row${item.column === current?.column ? " is-active" : ""}`}
                    onClick={() => setColumn(item.column)}
                  >
                    <span className="mini-row-name" title={item.column}>
                      {item.column}
                    </span>
                    <span className="mini-row-meta">
                      {item.kind === "numeric" ? "10 bin" : `${item.bins.length} kategori`} · {formatInt(itemTotal)} nilai
                    </span>
                    {top ? (
                      <span className="mini-row-top" title={top.label}>
                        {top.label}
                      </span>
                    ) : null}
                  </button>
                </li>
              );
            })}
          </ul>
        </div>
      </section>
    </div>
  );
}

function Duplicates({ data }: { data: DatasetAnalysis }) {
  const detail = data.duplicates_detail ?? { count: 0, rows: [] };
  const columns = data.column_names ?? [];

  return (
    <section className="panel">
      <div className="table-head">
        <h3>Baris duplikat</h3>
        <span>
          {formatInt(data.duplicates.duplicate_count)} duplikat · {formatPercent(data.duplicates.duplicate_percentage)} dari{" "}
          {formatInt(data.profile.rows)} baris
        </span>
      </div>
      <div className="panel-body">
        {detail.count ? (
          <>
            <p className="table-note">
              Ditampilkan maksimal 6 baris contoh. Duplikat dihitung dari seluruh kolom yang dibaca.
            </p>
            <div className="table-scroll">
              <table className="data-table is-compact">
                <thead>
                  <tr>
                    <th>#</th>
                    {columns.map((column) => (
                      <th key={column}>{column}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {detail.rows.map((row, index) => (
                    <tr key={index}>
                      <td className="num">{index + 1}</td>
                      {columns.map((column) => (
                        <td key={column} title={String(row[column] ?? "")}>
                          {row[column] === null || row[column] === undefined || row[column] === "" ? (
                            <span className="is-muted">kosong</span>
                          ) : (
                            String(row[column])
                          )}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </>
        ) : (
          <>
            <p className="empty-line">
              <Icon name="check" size={16} /> Tidak ada baris duplikat penuh pada dataset ini.
            </p>
            <p className="table-note">
              Duplikat sebagian (nilai mirip pada beberapa kolom) tidak dihitung di sini; gunakan tab Fingerprint untuk
              melihat ringkasan struktur.
            </p>
          </>
        )}
      </div>
    </section>
  );
}

function isNumeric(stat: ColumnStat): boolean {
  return stat.mean !== undefined && stat.mean !== null;
}

function TopValue({ stat }: { stat: ColumnStat }) {
  const top = stat.top_values?.[0];
  if (!top) return <>—</>;
  return (
    <span className="top-value" title={`${top.value} (${formatInt(top.frequency)} baris)`}>
      {top.value}
    </span>
  );
}

function MissingPanel({ data }: { data: DatasetAnalysis }) {
  const columns = useMemo(
    () => [...data.missing_values].sort((a, b) => b.missing_percentage - a.missing_percentage),
    [data.missing_values],
  );
  const affected = columns.filter((column) => column.missing_count > 0);
  const withZero = columns.filter((column) => column.zero_count > 0).slice(0, 8);

  return (
    <div className="split-grid">
      <section className="panel">
        <div className="table-head">
          <h3>Nilai kosong per kolom</h3>
          <span>{affected.length} kolom terdampak</span>
        </div>
        <div className="panel-body">
          {affected.length ? (
            <ul className="bar-list">
              {affected.map((column) => (
                <li key={column.column}>
                  <div className="bar-list-head">
                    <b title={column.column}>{column.column}</b>
                    <span>
                      {formatInt(column.missing_count)} sel · {formatPercent(column.missing_percentage)}
                    </span>
                  </div>
                  <div className={`bar-track${column.missing_percentage >= 40 ? " is-critical" : column.missing_percentage >= 5 ? " is-warn" : ""}`}>
                    <i style={{ width: `${Math.max(1.5, Math.min(100, column.missing_percentage))}%` }} />
                  </div>
                </li>
              ))}
            </ul>
          ) : (
            <p className="table-note">Tidak ada sel kosong pada kolom yang dianalisis.</p>
          )}
        </div>
      </section>

      <section className="panel">
        <div className="table-head">
          <h3>Nilai nol</h3>
          <span>Perlu dicek apakah 0 berarti data atau kekosongan</span>
        </div>
        <div className="panel-body">
          {withZero.length ? (
            <ul className="bar-list">
              {withZero.map((column) => (
                <li key={column.column}>
                  <div className="bar-list-head">
                    <b title={column.column}>{column.column}</b>
                    <span>{formatInt(column.zero_count)} nol · {formatPercent(column.zero_percentage)}</span>
                  </div>
                  <div className="bar-track is-neutral">
                    <i style={{ width: `${Math.max(1.5, Math.min(100, column.zero_percentage))}%` }} />
                  </div>
                </li>
              ))}
            </ul>
          ) : (
            <p className="table-note">Tidak ada nilai nol yang mencolok.</p>
          )}
        </div>
      </section>
    </div>
  );
}

function OutlierPanel({ data }: { data: DatasetAnalysis }) {
  const rows = useMemo(
    () => [...data.outliers].filter((row) => row.outlier_count > 0).sort((a, b) => b.outlier_count - a.outlier_count),
    [data.outliers],
  );

  return (
    <section className="panel">
      <div className="table-head">
        <h3>Outlier numerik (IQR 1,5×)</h3>
        <span>{rows.length} kolom dengan observasi ekstrem</span>
      </div>
      {rows.length ? (
        <div className="table-scroll">
          <table className="data-table">
            <thead>
              <tr>
                <th>Kolom</th>
                <th className="num">Outlier</th>
                <th className="num">Porsi</th>
                <th className="num">Batas bawah</th>
                <th className="num">Batas atas</th>
                <th>Sebaran</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => (
                <tr key={row.column}>
                  <td className="strong-cell">{row.column}</td>
                  <td className="num">{formatInt(row.outlier_count)}</td>
                  <td className={`num${row.outlier_percentage > 5 ? " is-warn" : ""}`}>{formatPercent(row.outlier_percentage)}</td>
                  <td className="num">{formatNumber(row.lower_bound)}</td>
                  <td className="num">{formatNumber(row.upper_bound)}</td>
                  <td>
                    <div className="bar-track is-neutral">
                      <i style={{ width: `${Math.max(1.5, Math.min(100, row.outlier_percentage))}%` }} />
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <p className="panel-body table-note">Tidak ada outlier yang melewati batas 1,5×IQR.</p>
      )}
      <p className="ml-foot">
        Outlier bukan otomatis kesalahan. Periksa konteks kolom sebelum menghapus atau melakukan winsorizing.
      </p>
    </section>
  );
}

function CorrelationPanel({ data }: { data: DatasetAnalysis }) {
  const { columns, values } = data.correlations;
  if (!columns.length) {
    return (
      <section className="panel soft-empty">
        <Icon name="analysis" size={20} />
        <h3>Korelasi belum bisa dihitung</h3>
        <p>Dibutuhkan minimal dua kolom numerik. Dataset ini belum memenuhinya.</p>
      </section>
    );
  }
  return (
    <section className="panel">
      <div className="table-head">
        <h3>Korelasi Pearson</h3>
        <span>
          {columns.length} kolom numerik · maksimal 24 kolom ditampilkan
        </span>
      </div>
      <div className="heatmap-scroll">
        <div className="heatmap" style={{ ["--cols" as string]: columns.length }}>
          <div className="heatmap-corner" />
          {columns.map((column) => (
            <div className="heatmap-col" key={`c-${column}`} title={column}>
              {shortName(column, 9)}
            </div>
          ))}
          {values.map((row, rowIndex) => (
            <div className="heatmap-row" key={`r-${columns[rowIndex]}`}>
              <div className="heatmap-row-label" title={columns[rowIndex]}>
                {shortName(columns[rowIndex], 12)}
              </div>
              {row.map((value, columnIndex) => (
                <div
                  className="heatmap-cell"
                  key={`${rowIndex}-${columnIndex}`}
                  title={`${columns[rowIndex]} × ${columns[columnIndex]} = ${value === null ? "—" : formatNumber(value, 3)}`}
                  style={{
                    background: heatColor(value),
                    color: value !== null && Math.abs(value) > 0.45 ? "#fff" : "var(--ink)",
                  }}
                >
                  {columns.length <= 10 && value !== null ? formatNumber(value, 2) : ""}
                </div>
              ))}
            </div>
          ))}
        </div>
      </div>
      <div className="heatmap-legend">
        <span>−1</span>
        <i className="heatmap-swatch" style={{ background: heatColor(-1) }} />
        <i className="heatmap-swatch" style={{ background: heatColor(-0.4) }} />
        <i className="heatmap-swatch" style={{ background: heatColor(0) }} />
        <i className="heatmap-swatch" style={{ background: heatColor(0.4) }} />
        <i className="heatmap-swatch" style={{ background: heatColor(1) }} />
        <span>+1</span>
      </div>
    </section>
  );
}

function heatColor(value: number | null): string {
  if (value === null || Number.isNaN(value)) return "#f3f7fd";
  const alpha = Math.min(1, Math.abs(value));
  return value >= 0
    ? `rgba(50, 103, 246, ${(alpha * 0.82).toFixed(3)})`
    : `rgba(240, 138, 62, ${(alpha * 0.78).toFixed(3)})`;
}

function FingerprintPanel({ data }: { data: DatasetAnalysis }) {
  const signature = data.fingerprint.representation.column_signature ?? [];
  return (
    <div className="split-grid">
      <section className="panel">
        <div className="table-head">
          <h3>Identitas dataset</h3>
          <span>Heuristik, bukan identitas absolut</span>
        </div>
        <div className="panel-body">
          <div className="fingerprint-hash">
            <span>SHA-256</span>
            <code>{data.fingerprint.hash || "—"}</code>
          </div>
          <div className="chip-row">
            {(data.fingerprint.representation.target_candidates ?? []).map((candidate) => (
              <span className="chip is-primary" key={candidate}>
                target: {candidate}
              </span>
            ))}
            {(data.fingerprint.representation.keywords ?? []).slice(0, 10).map((keyword) => (
              <span className="chip is-small" key={keyword}>
                {keyword}
              </span>
            ))}
          </div>
          <ValueList title="Karakteristik" value={data.fingerprint.representation.characteristics} />
          <ValueList title="Kategori" value={data.fingerprint.representation.categories} />
        </div>
      </section>

      <section className="panel">
        <div className="table-head">
          <h3>Kolom bertanda</h3>
          <span>Identifier dan kandidat target</span>
        </div>
        <div className="panel-body">
          {signature.length ? (
            <ul className="flag-list">
              {signature.slice(0, 40).map((column) => (
                <li key={column.name}>
                  <b>{column.name}</b>
                  <span className="flag-tags">
                    {column.semantic_type ? <span className="chip is-small">{column.semantic_type}</span> : null}
                    {column.potential_id ? <span className="chip is-small is-warn">identifier</span> : null}
                    {column.target_candidate ? <span className="chip is-small is-primary">target</span> : null}
                  </span>
                </li>
              ))}
            </ul>
          ) : (
            <p className="table-note">Tidak ada kolom bertanda khusus.</p>
          )}
        </div>
      </section>
    </div>
  );
}

function ValueList({ title, value }: { title: string; value: Record<string, unknown> | undefined }) {
  if (!value) return null;
  const entries = Object.entries(value).filter(([, item]) => typeof item !== "object" || item === null);
  if (!entries.length) return null;
  return (
    <div className="value-list">
      <span className="pillar-label">{title}</span>
      <dl>
        {entries.slice(0, 12).map(([key, item]) => (
          <div key={key}>
            <dt>{key.replaceAll("_", " ")}</dt>
            <dd>{String(item)}</dd>
          </div>
        ))}
      </dl>
    </div>
  );
}
