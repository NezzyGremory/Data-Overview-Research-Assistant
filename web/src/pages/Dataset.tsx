import { useMemo, useState } from "react";
import { Icon } from "../components/Icons";
import { PageHeader } from "../components/PageHeader";
import { useScrollReveal } from "../lib/motion";
import { fileExtension, formatBytes, formatInt, formatNumber, formatPercent } from "../lib/format";
import type { CleanReport, DatasetAnalysis } from "../lib/types";

export type DatasetPageProps = {
  data: DatasetAnalysis;
  hasSourceFile: boolean;
  cleaning: boolean;
  cleanReport: CleanReport | null;
  onClean: (strategy: "impute" | "drop_rows") => void;
};

export function DatasetPage({ data, hasSourceFile, cleaning, cleanReport, onClean }: DatasetPageProps) {
  useScrollReveal(data.filename);
  const [strategy, setStrategy] = useState<"impute" | "drop_rows">("impute");
  const [rowFilter, setRowFilter] = useState("");
  const [columnQuery, setColumnQuery] = useState("");

  const columnRows = useMemo(
    () => data.statistics.slice(0, 120).map((stat) => {
      const top = stat.top_values?.[0]?.value;
      const first = firstPreviewValue(data.preview, stat.column);
      const numeric = stat.mean !== undefined && stat.mean !== null ? `μ ${formatNumber(stat.mean)}` : null;
      return { ...stat, sample: top ?? first ?? numeric };
    }),
    [data.statistics, data.preview],
  );

  const visibleColumns = useMemo(() => {
    const term = columnQuery.trim().toLowerCase();
    return term ? columnRows.filter((row) => row.column.toLowerCase().includes(term)) : columnRows;
  }, [columnRows, columnQuery]);

  const visibleRows = useMemo(() => {
    const term = rowFilter.trim().toLowerCase();
    const rows = data.preview.map((row, index) => ({ row, index }));
    if (!term) return rows;
    return rows.filter(({ row }) =>
      data.column_names.some((column) => String(row[column] ?? "").toLowerCase().includes(term)),
    );
  }, [data.preview, data.column_names, rowFilter]);

  function downloadDictionary() {
    const header = ["column", "dtype", "unique_count", "missing_percentage", "sample_value"];
    const lines = [header.join(",")];
    for (const row of columnRows) {
      lines.push(
        [row.column, row.dtype, row.unique_count, row.missing_percentage.toFixed(2), row.sample ?? ""]
          .map(csvCell)
          .join(","),
      );
    }
    const blob = new Blob(["\uFEFF" + lines.join("\r\n")], { type: "text/csv;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `${data.filename.replace(/\.[^.]+$/, "")}_kamus_data.csv`;
    anchor.click();
    window.setTimeout(() => URL.revokeObjectURL(url), 20000);
  }

  return (
    <div className="page">
      <PageHeader
        overline="DATASET"
        title="Struktur & isi dataset"
        lead="Semua yang terbaca dari file CSV: pratinjau baris, tipe kolom, dan nilai unik per kolom."
        meta={
          <>
            <span className="meta-chip">
              <b>{fileExtension(data.filename)}</b> {data.filename}
            </span>
            <span className="meta-chip">{formatInt(data.profile.rows)} baris</span>
            <span className="meta-chip">{formatInt(data.profile.columns)} kolom</span>
            <span className="meta-chip">{formatBytes(data.profile.memory_usage)} memori</span>
          </>
        }
      />

      <section className="panel" data-reveal>
        <div className="table-head">
          <h3>Pratinjau baris</h3>
          <label className="search-field">
            <Icon name="search" size={14} />
            <input
              value={rowFilter}
              onChange={(event) => setRowFilter(event.currentTarget.value)}
              placeholder="Cari isi baris…"
              aria-label="Cari isi baris"
            />
          </label>
        </div>
        <div className="table-scroll preview-scroll">
          <table className="preview-table">
            <thead>
              <tr>
                <th className="index-cell">#</th>
                {data.column_names.map((column) => (
                  <th key={column}>{column}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {visibleRows.map(({ row, index }, position) => (
                <tr key={index} style={{ ["--i" as string]: position }}>
                  <td className="index-cell">{index + 1}</td>
                  {data.column_names.map((column) => (
                    <td key={column} title={String(row[column] ?? "")}>
                      {String(row[column] ?? "—")}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="table-note">
          {rowFilter
            ? `${formatInt(visibleRows.length)} dari ${formatInt(data.preview.length)} baris pratinjau cocok.`
            : `${data.preview.length} baris pertama dari ${formatInt(data.profile.rows)} baris.`}
        </p>
      </section>

      <section className="panel" data-reveal>
        <div className="table-head">
          <h3>Profil kolom</h3>
          <div className="table-tools">
            <label className="search-field">
              <Icon name="search" size={14} />
              <input
                value={columnQuery}
                onChange={(event) => setColumnQuery(event.currentTarget.value)}
                placeholder="Cari kolom…"
                aria-label="Cari kolom"
              />
            </label>
            <button type="button" className="ghost-button" onClick={downloadDictionary}>
              <Icon name="download" size={14} /> Kamus data
            </button>
          </div>
        </div>
        <div className="table-scroll">
          <table className="data-table">
            <thead>
              <tr>
                <th>Kolom</th>
                <th>Tipe</th>
                <th className="num">Unik</th>
                <th className="num">Kosong</th>
                <th>Contoh nilai</th>
              </tr>
            </thead>
            <tbody>
              {visibleColumns.map((stat) => (
                <tr key={stat.column}>
                  <td className="strong-cell" title={stat.column}>
                    {stat.column}
                  </td>
                  <td>
                    <span className={`type-tag t-${typeClass(stat.dtype)}`}>{stat.dtype}</span>
                  </td>
                  <td className="num">{formatInt(stat.unique_count)}</td>
                  <td className={`num${stat.missing_percentage > 0 ? " is-warn" : ""}`}>
                    {formatPercent(stat.missing_percentage)}
                  </td>
                  <td className="sample-cell" title={String(stat.sample ?? "")}>
                    {stat.sample === null || stat.sample === undefined ? "—" : String(stat.sample)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="table-note">
          {visibleColumns.length} kolom dianalisis
          {data.limits.truncated ? ` · ${data.limits.columns_total - data.limits.columns_analyzed} kolom lain tidak dimuat` : ""}
        </p>
      </section>

      <section className="panel clean-panel" data-reveal aria-labelledby="clean-title">
        <div className="ml-head">
          <div>
            <span className="overline">PEMBERSIHAN DASAR</span>
            <h2 id="clean-title">Bersihkan lalu unduh CSV</h2>
          </div>
          <span className="estimate-tag">HASIL DIUNDUH · TIDAK DISIMPAN</span>
        </div>

        {hasSourceFile ? (
          <>
            <div className="clean-options" role="radiogroup" aria-label="Metode pembersihan">
              <button
                type="button"
                role="radio"
                aria-checked={strategy === "impute"}
                className={`clean-option${strategy === "impute" ? " is-selected" : ""}`}
                onClick={() => setStrategy("impute")}
              >
                <span className="clean-option-head">
                  <b>Imputasi</b>
                  <span className="chip is-small">median · modus</span>
                </span>
                <span>Isi sel kosong dengan median (numerik) atau modus (kategorikal). Baris tetap utuh.</span>
              </button>
              <button
                type="button"
                role="radio"
                aria-checked={strategy === "drop_rows"}
                className={`clean-option${strategy === "drop_rows" ? " is-selected" : ""}`}
                onClick={() => setStrategy("drop_rows")}
              >
                <span className="clean-option-head">
                  <b>Hapus baris kosong</b>
                  <span className="chip is-small">listwise</span>
                </span>
                <span>Buang setiap baris yang masih memiliki sel kosong. Ukuran sampel berkurang.</span>
              </button>
            </div>
            <div className="clean-actions">
              <button className="primary-button" type="button" onClick={() => onClean(strategy)} disabled={cleaning}>
                {cleaning ? (
                  <>
                    <span className="spinner" /> Memproses…
                  </>
                ) : (
                  <>
                    <Icon name="download" size={16} /> Bersihkan & unduh
                  </>
                )}
              </button>
              <span className="clean-note">
                <Icon name="shield" size={13} /> File asli tidak diubah. Hasil dikirim langsung ke perangkatmu.
              </span>
            </div>
          </>
        ) : (
          <p className="clean-note is-blocked">
            <Icon name="alert" size={14} /> File sumber sudah tidak ada di memori browser. Unggah ulang dataset dari
            halaman Overview untuk membersihkan.
          </p>
        )}

        {cleanReport ? (
          <div className="clean-result" role="status">
            <div className="clean-result-grid">
              <div>
                <span>BARIS SEBELUM</span>
                <b>{formatInt(cleanReport.rows_before)}</b>
              </div>
              <div>
                <span>BARIS SESUDAH</span>
                <b>{formatInt(cleanReport.rows_after)}</b>
              </div>
              <div>
                <span>BARIS DIBUANG</span>
                <b>{formatInt(cleanReport.rows_dropped)}</b>
              </div>
              <div>
                <span>SEL KOSONG SISA</span>
                <b>{formatInt(cleanReport.missing_after)}</b>
              </div>
            </div>
            {cleanReport.imputed_columns.length ? (
              <div className="clean-imputed">
                <span>
                  {formatInt(cleanReport.cells_imputed)} sel kosong diisi (median untuk numerik, modus untuk kategorikal) pada
                  kolom:
                </span>
                <div className="chip-row">
                  {cleanReport.imputed_columns.slice(0, 12).map((column) => (
                    <span className="chip is-small" key={column}>
                      {column}
                    </span>
                  ))}
                  {cleanReport.imputed_columns.length > 12 ? (
                    <span className="chip is-small">+{cleanReport.imputed_columns.length - 12}</span>
                  ) : null}
                </div>
              </div>
            ) : null}
            {cleanReport.dropped_empty_columns.length ? (
              <p className="clean-warning">
                Kolom tanpa nilai sama sekali dibuang: {cleanReport.dropped_empty_columns.join(", ")}.
              </p>
            ) : null}
            {cleanReport.warnings.map((warning) => (
              <p className="clean-warning" key={warning}>
                {warning}
              </p>
            ))}
          </div>
        ) : null}
      </section>
    </div>
  );
}

function typeClass(dtype: string): string {
  if (/^(int|float|uint|Int|Float|boolean)/.test(dtype)) return "number";
  if (/datetime|date/.test(dtype)) return "date";
  if (/categor|object|string|str/.test(dtype)) return "text";
  return "other";
}

function firstPreviewValue(
  rows: Record<string, string | number | boolean | null>[],
  column: string,
): string | null {
  for (const row of rows) {
    const value = row[column];
    if (value !== null && value !== undefined && String(value).length) return String(value);
  }
  return null;
}

function csvCell(value: string | number | null | undefined): string {
  const text = value === null || value === undefined ? "" : String(value);
  return /[",\r\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}
