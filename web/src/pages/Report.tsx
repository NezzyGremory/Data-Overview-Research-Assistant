import { useMemo } from "react";
import { Icon } from "../components/Icons";
import { PageHeader } from "../components/PageHeader";
import { useScrollReveal } from "../lib/motion";
import { formatBytes, formatInt, formatNumber, formatPercent } from "../lib/format";
import type { DatasetAnalysis } from "../lib/types";

export function ReportPage({ data }: { data: DatasetAnalysis }) {
  useScrollReveal(data.filename);
  const today = useMemo(
    () => new Date().toLocaleDateString("id-ID", { day: "numeric", month: "long", year: "numeric" }),
    [],
  );
  const pairs = useMemo(() => topPairs(data), [data]);

  const download = () => {
    const html = buildReportHtml(data, today);
    const blob = new Blob([html], { type: "text/html;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `laporan_${data.filename.replace(/\.[^.]+$/, "") || "dataset"}.html`;
    document.body.appendChild(anchor);
    anchor.click();
    anchor.remove();
    window.setTimeout(() => URL.revokeObjectURL(url), 20_000);
  };

  return (
    <div className="page">
      <PageHeader
        overline="RESEARCH REPORT"
        title="Laporan siap cetak"
        lead="Rangkuman diagnosis, statistik, dan rekomendasi metode. Cetak ke PDF atau unduh sebagai HTML mandiri."
        actions={
          <div className="report-actions">
            <button className="primary-button" type="button" onClick={() => window.print()}>
              <Icon name="print" size={16} /> Cetak / PDF
            </button>
            <button className="ghost-button" type="button" onClick={download}>
              <Icon name="download" size={15} /> Unduh HTML
            </button>
          </div>
        }
      />

      <article className="report-sheet" data-reveal>
        <header className="report-head">
          <div>
            <span className="report-brand">DORA</span>
            <span className="report-brand-sub">Data Overview &amp; Research Assistant</span>
          </div>
          <div className="report-meta">
            <span>Laporan analisis dataset</span>
            <b>{today}</b>
          </div>
        </header>

        <h1 className="report-title">{data.filename}</h1>
        <p className="report-sub">
          {formatInt(data.profile.rows)} baris · {formatInt(data.profile.columns)} kolom ·{" "}
          {formatBytes(data.profile.memory_usage)} memori · fingerprint{" "}
          <code>{data.fingerprint.hash.slice(0, 16)}…</code>
        </p>

        <section className="report-section">
          <h2>1 · Ringkasan</h2>
          <div className="report-metrics">
            <ReportMetric label="Baris" value={formatInt(data.profile.rows)} />
            <ReportMetric label="Kolom" value={formatInt(data.profile.columns)} />
            <ReportMetric label="Sel kosong" value={formatInt(data.profile.missing_values)} />
            <ReportMetric label="Baris duplikat" value={formatInt(data.profile.duplicate_rows)} />
            <ReportMetric label="Kolom numerik" value={formatInt(data.profile.numeric_columns)} />
            <ReportMetric label="Kolom kategorikal" value={formatInt(data.profile.categorical_columns)} />
          </div>
        </section>

        <section className="report-section">
          <h2>2 · Diagnosis kualitas data</h2>
          {data.quality.length ? (
            <ol className="report-list">
              {data.quality.map((item, index) => (
                <li key={`${item.category}-${index}`}>
                  <b>
                    {item.title} <span className={`report-sev sev-${item.severity}`}>{item.severity}</span>
                  </b>
                  <p>{item.problem}</p>
                  <p className="report-soft">{item.magnitude}</p>
                  <p>
                    <i>Dampak:</i> {item.impact}
                  </p>
                  <ul>
                    {item.options.map((option) => (
                      <li key={option}>{option}</li>
                    ))}
                  </ul>
                </li>
              ))}
            </ol>
          ) : (
            <p className="report-soft">Tidak ada isu kualitas yang terdeteksi oleh pemeriksaan otomatis.</p>
          )}
        </section>

        <section className="report-section">
          <h2>3 · Statistik kolom</h2>
          <table className="report-table">
            <thead>
              <tr>
                <th>Kolom</th>
                <th>Tipe</th>
                <th>Unik</th>
                <th>Kosong</th>
                <th>Rata-rata</th>
                <th>Median</th>
                <th>Min</th>
                <th>Maks</th>
              </tr>
            </thead>
            <tbody>
              {data.statistics.map((stat) => (
                <tr key={stat.column}>
                  <td>{stat.column}</td>
                  <td>{stat.dtype}</td>
                  <td>{formatInt(stat.unique_count)}</td>
                  <td>{formatPercent(stat.missing_percentage)}</td>
                  <td>{stat.mean === null || stat.mean === undefined ? "—" : formatNumber(stat.mean)}</td>
                  <td>{stat.median === null || stat.median === undefined ? "—" : formatNumber(stat.median)}</td>
                  <td>{stat.min === null || stat.min === undefined ? "—" : formatNumber(stat.min)}</td>
                  <td>{stat.max === null || stat.max === undefined ? "—" : formatNumber(stat.max)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>

        <section className="report-section">
          <h2>4 · Korelasi terkuat</h2>
          {pairs.length ? (
            <table className="report-table">
              <thead>
                <tr>
                  <th>Kolom A</th>
                  <th>Kolom B</th>
                  <th>r (Pearson)</th>
                </tr>
              </thead>
              <tbody>
                {pairs.map((pair) => (
                  <tr key={`${pair.a}-${pair.b}`}>
                    <td>{pair.a}</td>
                    <td>{pair.b}</td>
                    <td>{formatNumber(pair.value, 3)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : (
            <p className="report-soft">Korelasi belum dapat dihitung (butuh minimal dua kolom numerik).</p>
          )}
        </section>

        <section className="report-section">
          <h2>5 · Rekomendasi metode</h2>
          <p className="report-soft">
            Task utama: <b>{data.ml_intelligence.task_detection.primary_task?.label || "belum terdeteksi"}</b>
          </p>
          <ol className="report-list">
            {data.ml_intelligence.recommendation.recommendations.slice(0, 6).map((method) => (
              <li key={method.method_id}>
                <b>{method.method}</b>{" "}
                <span className="report-soft">
                  {method.category ? `· ${method.category}` : ""} {typeof method.score === "number" ? `· skor ${Math.round(method.score)}` : ""}
                </span>
                <p>{method.description}</p>
              </li>
            ))}
          </ol>
        </section>

        <footer className="report-foot">
          <p>
            Laporan ini dihasilkan otomatis oleh DORA dari file yang diunggah pada {today}. Analisis berjalan sementara:
            file sumber sudah dihapus dari server setelah diproses.
          </p>
          <p>
            Skor dan rekomendasi bersifat estimasi heuristik. Model belum dilatih, diuji, atau dibandingkan pada dataset
            ini. Verifikasi statistik dan validasi metodologis tetap menjadi tanggung jawab peneliti.
          </p>
        </footer>
      </article>
    </div>
  );
}

function ReportMetric({ label, value }: { label: string; value: string }) {
  return (
    <div className="report-metric">
      <span>{label}</span>
      <b>{value}</b>
    </div>
  );
}

function topPairs(data: DatasetAnalysis) {
  const { columns, values } = data.correlations;
  const pairs: { a: string; b: string; value: number }[] = [];
  for (let row = 0; row < columns.length; row += 1) {
    for (let column = row + 1; column < columns.length; column += 1) {
      const value = values[row]?.[column];
      if (value === null || value === undefined) continue;
      pairs.push({ a: columns[row], b: columns[column], value });
    }
  }
  return pairs.sort((left, right) => Math.abs(right.value) - Math.abs(left.value)).slice(0, 8);
}

/* ---------------------------------------------------------------------------
   Standalone HTML export (no external assets, safe to open offline)
   --------------------------------------------------------------------------- */

function escapeHtml(value: string): string {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function buildReportHtml(data: DatasetAnalysis, today: string): string {
  const rows = data.statistics
    .map(
      (stat) => `<tr>
        <td>${escapeHtml(stat.column)}</td>
        <td>${escapeHtml(stat.dtype)}</td>
        <td class="num">${formatInt(stat.unique_count)}</td>
        <td class="num">${formatPercent(stat.missing_percentage)}</td>
        <td class="num">${stat.mean === null || stat.mean === undefined ? "—" : formatNumber(stat.mean)}</td>
        <td class="num">${stat.median === null || stat.median === undefined ? "—" : formatNumber(stat.median)}</td>
        <td class="num">${stat.min === null || stat.min === undefined ? "—" : formatNumber(stat.min)}</td>
        <td class="num">${stat.max === null || stat.max === undefined ? "—" : formatNumber(stat.max)}</td>
      </tr>`,
    )
    .join("");

  const diagnoses = data.quality.length
    ? data.quality
        .map(
          (item) => `<li>
            <h3>${escapeHtml(item.title)} <span class="sev sev-${item.severity}">${item.severity}</span></h3>
            <p>${escapeHtml(item.problem)}</p>
            <p class="soft">${escapeHtml(item.magnitude)}</p>
            <p><strong>Dampak:</strong> ${escapeHtml(item.impact)}</p>
            <ul>${item.options.map((option) => `<li>${escapeHtml(option)}</li>`).join("")}</ul>
          </li>`,
        )
        .join("")
    : "<li>Tidak ada isu kualitas yang terdeteksi.</li>";

  const pairRows = topPairs(data)
    .map(
      (pair) => `<tr><td>${escapeHtml(pair.a)}</td><td>${escapeHtml(pair.b)}</td><td class="num">${formatNumber(pair.value, 3)}</td></tr>`,
    )
    .join("");

  const methods = data.ml_intelligence.recommendation.recommendations
    .slice(0, 6)
    .map(
      (method) =>
        `<li><strong>${escapeHtml(method.method)}</strong> <span class="soft">${escapeHtml(
          method.category ?? "",
        )}${typeof method.score === "number" ? ` · skor ${Math.round(method.score)}` : ""}</span><p>${escapeHtml(
          method.description ?? "",
        )}</p></li>`,
    )
    .join("");

  return `<!doctype html>
<html lang="id">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Laporan ${escapeHtml(data.filename)} · DORA</title>
<style>
  :root { color-scheme: light; }
  * { box-sizing: border-box; }
  body { margin: 0; padding: 40px 24px 64px; background: #f5f9ff; color: #172033;
    font: 14px/1.6 "Segoe UI", Inter, system-ui, sans-serif; }
  main { max-width: 860px; margin: 0 auto; background: #fff; border: 1px solid #dce8f7;
    border-radius: 18px; padding: 44px 48px 52px; }
  header { display: flex; justify-content: space-between; align-items: flex-start;
    border-bottom: 2px solid #3267f6; padding-bottom: 18px; margin-bottom: 26px; }
  .brand { font-size: 24px; font-weight: 800; letter-spacing: -.03em; color: #3267f6; }
  .brand small { display: block; font-size: 10px; font-weight: 700; letter-spacing: .14em;
    color: #7c8aa5; text-transform: uppercase; }
  .meta { text-align: right; font-size: 11px; color: #7c8aa5; }
  .meta b { display: block; color: #172033; font-size: 13px; }
  h1 { font-size: 26px; letter-spacing: -.02em; margin: 0 0 6px; word-break: break-word; }
  .sub { margin: 0 0 26px; color: #5b6b85; font-size: 12.5px; }
  h2 { font-size: 15px; margin: 30px 0 12px; letter-spacing: -.01em; }
  h3 { font-size: 13px; margin: 0 0 6px; }
  p { margin: 0 0 8px; }
  .soft { color: #7c8aa5; }
  .sev { font-size: 9px; font-weight: 800; letter-spacing: .1em; text-transform: uppercase;
    padding: 2px 7px; border-radius: 999px; border: 1px solid; vertical-align: middle; }
  .sev-critical { color: #c0341f; border-color: #f6c7bd; background: #fdf0ed; }
  .sev-warning { color: #9a6200; border-color: #f6dcae; background: #fdf7ec; }
  .sev-info { color: #2a63c9; border-color: #cfe0ff; background: #eff5ff; }
  .metrics { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; }
  .metric { border: 1px solid #dce8f7; border-radius: 10px; padding: 10px 12px; }
  .metric span { display: block; font-size: 9px; letter-spacing: .1em; color: #7c8aa5;
    text-transform: uppercase; font-weight: 700; }
  .metric b { font-size: 17px; }
  ol, ul { padding-left: 20px; }
  ol > li { margin-bottom: 16px; }
  table { width: 100%; border-collapse: collapse; font-size: 11.5px; }
  th, td { border-bottom: 1px solid #e8f0fb; padding: 7px 8px; text-align: left; }
  th { background: #f5f9ff; font-size: 10px; letter-spacing: .06em; text-transform: uppercase; color: #5b6b85; }
  .num { text-align: right; font-variant-numeric: tabular-nums; }
  footer { margin-top: 32px; padding-top: 16px; border-top: 1px solid #e8f0fb;
    font-size: 11px; color: #7c8aa5; }
  code { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 11px; }
  @media print { body { background: #fff; padding: 0; } main { border: 0; border-radius: 0; padding: 0; } }
</style>
</head>
<body>
<main>
  <header>
    <div class="brand">DORA<small>Data Overview &amp; Research Assistant</small></div>
    <div class="meta">Laporan analisis dataset<b>${escapeHtml(today)}</b></div>
  </header>
  <h1>${escapeHtml(data.filename)}</h1>
  <p class="sub">${formatInt(data.profile.rows)} baris · ${formatInt(data.profile.columns)} kolom · ${formatBytes(
    data.profile.memory_usage,
  )} memori · fingerprint <code>${escapeHtml(data.fingerprint.hash.slice(0, 16))}…</code></p>

  <h2>1 · Ringkasan</h2>
  <div class="metrics">
    <div class="metric"><span>Baris</span><b>${formatInt(data.profile.rows)}</b></div>
    <div class="metric"><span>Kolom</span><b>${formatInt(data.profile.columns)}</b></div>
    <div class="metric"><span>Sel kosong</span><b>${formatInt(data.profile.missing_values)}</b></div>
    <div class="metric"><span>Baris duplikat</span><b>${formatInt(data.profile.duplicate_rows)}</b></div>
    <div class="metric"><span>Kolom numerik</span><b>${formatInt(data.profile.numeric_columns)}</b></div>
    <div class="metric"><span>Kolom kategorikal</span><b>${formatInt(data.profile.categorical_columns)}</b></div>
  </div>

  <h2>2 · Diagnosis kualitas data</h2>
  <ol>${diagnoses}</ol>

  <h2>3 · Statistik kolom</h2>
  <table>
    <thead><tr><th>Kolom</th><th>Tipe</th><th class="num">Unik</th><th class="num">Kosong</th>
      <th class="num">Rata-rata</th><th class="num">Median</th><th class="num">Min</th><th class="num">Maks</th></tr></thead>
    <tbody>${rows}</tbody>
  </table>

  <h2>4 · Korelasi terkuat</h2>
  ${
    pairRows
      ? `<table><thead><tr><th>Kolom A</th><th>Kolom B</th><th class="num">r</th></tr></thead><tbody>${pairRows}</tbody></table>`
      : '<p class="soft">Korelasi belum dapat dihitung.</p>'
  }

  <h2>5 · Rekomendasi metode</h2>
  <p class="soft">Task utama: <strong>${escapeHtml(
    data.ml_intelligence.task_detection.primary_task?.label || "belum terdeteksi",
  )}</strong></p>
  <ol>${methods}</ol>

  <footer>
    <p>Laporan dihasilkan otomatis oleh DORA pada ${escapeHtml(today)}. File sumber sudah dihapus dari server setelah diproses.</p>
    <p>Skor dan rekomendasi bersifat estimasi heuristik. Model belum dilatih atau diuji pada dataset ini.</p>
  </footer>
</main>
</body>
</html>`;
}
