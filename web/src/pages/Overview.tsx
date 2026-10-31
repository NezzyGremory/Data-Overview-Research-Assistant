import { useMemo, useState } from "react";
import type { FormEvent } from "react";
import { Icon } from "../components/Icons";
import { DataStack } from "../components/Ambient";
import { CatalogPanel, MlPanel } from "../components/MlPanels";
import { useCountUp, useRingSweep, useScrollReveal } from "../lib/motion";
import type { DatasetProfile, MlCatalog } from "../lib/types";

const MAX_FILE_BYTES = 10 * 1024 * 1024;
const ACCEPTED = ".csv";
const numberFormat = new Intl.NumberFormat("id-ID");

function formatBytes(value: number): string {
  if (!Number.isFinite(value) || value <= 0) return "0 B";
  if (value < 1024) return `${Math.round(value)} B`;
  if (value < 1024 * 1024) return `${(value / 1024).toFixed(1)} KB`;
  return `${(value / (1024 * 1024)).toFixed(1)} MB`;
}

function fileExtension(filename: string): string {
  const clean = filename.split(/[\\/]/).pop() ?? filename;
  const dot = clean.lastIndexOf(".");
  return dot > 0 ? clean.slice(dot + 1).toUpperCase().slice(0, 4) : "DATA";
}

export type OverviewProps = {
  profile: DatasetProfile | null;
  catalog: MlCatalog | null;
  busy: boolean;
  error: string;
  onAnalyze: (file: File) => void;
  onOpenAnalysis: () => void;
  onOpenDatasetSearch?: () => void;
};

export function Overview({
  profile,
  catalog,
  busy,
  error,
  onAnalyze,
  onOpenAnalysis,
  onOpenDatasetSearch,
}: OverviewProps) {
  useScrollReveal(profile);

  return (
    <div className="page">
      <HeroBand
        hasProfile={Boolean(profile)}
        onOpenAnalysis={onOpenAnalysis}
        onOpenDatasetSearch={onOpenDatasetSearch}
      />
      {profile ? (
        <>
          <ResultPanel profile={profile} />
          <MlPanel profile={profile} />
        </>
      ) : (
        <UploadPanel busy={busy} error={error} onAnalyze={onAnalyze} />
      )}
      {catalog ? <CatalogPanel catalog={catalog} /> : null}
    </div>
  );
}

function HeroBand({
  hasProfile,
  onOpenAnalysis,
  onOpenDatasetSearch,
}: {
  hasProfile: boolean;
  onOpenAnalysis: () => void;
  onOpenDatasetSearch?: () => void;
}) {
  return (
    <section className="hero-band hero-band-centered" aria-labelledby="hero-title">
      <div className="hero-copy hero-copy-centered">
        <div className="eyebrow-wrap">
          <span className="eyebrow rise">
            <i /> RUANG KERJA ANALISIS &amp; RISET DATASET
          </span>
        </div>

        <h1 className="hero-title rise rise-1" id="hero-title">
          Mulai dari <span className="outline">dataset.</span>
        </h1>

        <p className="hero-lead rise rise-2">
          {hasProfile
            ? "Profil dataset aktif. Eksplorasi diagnosis kualitas data 4 pilar, rekomendasi model machine learning, serta perankingan literatur ilmiah."
            : "Platform analisis dataset komprehensif dan riset akademik. Eksplorasi struktur, deteksi kualitas otomatis, estimasi model ML, hingga telaah riset literatur."}
        </p>

        {/* Interactive 3D DORA Crystal & Data Orbit positioned in the CENTER */}
        <div className="hero-centerpiece rise rise-3">
          <DataStack />
        </div>

        <div className="hero-actions rise rise-4">
          <a className="primary-button" href="#dataset-workspace">
            {hasProfile ? "Lihat Ringkasan Data" : "Mulai Analisis CSV"}
            <Icon name="arrow" size={16} />
          </a>
          {onOpenDatasetSearch && (
            <button className="ghost-button" type="button" onClick={onOpenDatasetSearch}>
              <Icon name="search" size={15} /> Cari Dataset Online
            </button>
          )}
          {hasProfile ? (
            <button className="ghost-button" type="button" onClick={onOpenAnalysis}>
              <Icon name="analysis" size={15} /> Buka Analysis
            </button>
          ) : null}
        </div>


      </div>
    </section>
  );
}

function UploadPanel({
  busy,
  error,
  onAnalyze,
}: {
  busy: boolean;
  error: string;
  onAnalyze: (file: File) => void;
}) {
  const [file, setFile] = useState<File | null>(null);
  const [dragging, setDragging] = useState(false);
  const [localError, setLocalError] = useState("");

  const accept = (selected: File | null) => {
    setLocalError("");
    if (!selected) {
      setFile(null);
      return;
    }
    if (selected.size > MAX_FILE_BYTES) {
      setFile(null);
      setLocalError("Ukuran file melebihi batas 10 MB.");
      return;
    }
    if (!/\.csv$/i.test(selected.name)) {
      setFile(null);
      setLocalError("DORA versi web hanya membaca file CSV (.csv).");
      return;
    }
    setFile(selected);
  };

  const submit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (file) onAnalyze(file);
  };

  return (
    <section className="upload-grid" id="dataset-workspace" data-reveal aria-label="Unggah dataset">
      <div className="panel upload-panel-solo">
        <div className="panel-head">
          <div>
            <span className="overline">PROFIL DATASET</span>
            <h2>Unggah file untuk memulai</h2>
            <p>Mendukung format CSV hingga 10 MB. Struktur data dan kualitas awal dianalisis secara instan.</p>
          </div>
        </div>
        <div className="panel-body">
          <form onSubmit={submit}>
            <label
              className={`dropzone${file ? " has-file" : ""}${dragging ? " is-drag" : ""}`}
              htmlFor="dataset-file"
              onDragOver={(event) => {
                event.preventDefault();
                setDragging(true);
              }}
              onDragLeave={() => setDragging(false)}
              onDrop={(event) => {
                event.preventDefault();
                setDragging(false);
                accept(event.dataTransfer.files?.[0] ?? null);
              }}
            >
              <span className="drop-icon">
                <Icon name={file ? "file" : "upload"} size={22} />
              </span>
              <span className="drop-title">{file ? file.name : "Tarik file ke sini"}</span>
              <span className="drop-hint">
                {file
                  ? `${formatBytes(file.size)} · siap diproses`
                  : "File CSV (.csv) · maks. 10 MB"}
              </span>
              <span className="drop-browse">{file ? "Ganti file" : "Telusuri file"}</span>
              <input
                id="dataset-file"
                type="file"
                accept={ACCEPTED}
                onChange={(event) => accept(event.currentTarget.files?.[0] ?? null)}
              />
            </label>
            {file ? (
              <p className="selected-file">
                <Icon name="check" size={13} /> <b>{fileExtension(file.name)}</b> terpilih, belum diunggah
              </p>
            ) : null}
            <div className="upload-actions">
              <button className="primary-button" type="submit" disabled={!file || busy}>
                {busy ? (
                  <>
                    <span className="spinner" /> Memprofilkan dataset…
                  </>
                ) : (
                  <>
                    Buat profil dataset <Icon name="arrow" size={16} />
                  </>
                )}
              </button>
            </div>
          </form>
          {localError || error ? (
            <p className="error-note" role="alert">
              <Icon name="alert" size={14} /> {localError || error}
            </p>
          ) : null}
        </div>
      </div>
    </section>
  );
}

function Metric({
  label,
  value,
  unit,
  display,
  percent,
  warn,
}: {
  label: string;
  value: number;
  unit: string;
  display?: (value: number) => string;
  percent?: number;
  warn?: boolean;
}) {
  const animated = useCountUp(value);
  const text = display ? display(animated) : numberFormat.format(Math.round(animated));
  return (
    <article className="metric">
      <span className="metric-label">{label}</span>
      <strong className="metric-value">{text}</strong>
      <span className="metric-unit">{unit}</span>
      {typeof percent === "number" ? (
        <div className={`metric-bar${warn ? " is-warn" : ""}`}>
          <i style={{ width: percent === 0 ? 0 : `${Math.max(2, Math.min(100, percent))}%` }} />
        </div>
      ) : null}
    </article>
  );
}

function QualityRing({ score }: { score: number }) {
  const radius = 56;
  const { circumference, offset } = useRingSweep(score, radius);
  return (
    <div className="ring-wrap">
      <div className="ring">
        <svg
          width="112"
          height="112"
          viewBox="0 0 132 132"
          role="img"
          aria-label={`Skor kesiapan data ${score} dari 100`}
        >
          <defs>
            <linearGradient id="ringGradient" x1="0" y1="0" x2="1" y2="1">
              <stop offset="0%" stopColor="#4f8cff" />
              <stop offset="100%" stopColor="#7fb2ff" />
            </linearGradient>
          </defs>
          <circle className="ring-track" cx="66" cy="66" r={radius} fill="none" strokeWidth="9" />
          <circle
            className="ring-value"
            cx="66"
            cy="66"
            r={radius}
            fill="none"
            strokeWidth="9"
            strokeDasharray={circumference}
            strokeDashoffset={offset}
          />
        </svg>
        <div className="ring-center">
          <strong>{Math.round(score)}</strong>
          <span>DARI 100</span>
        </div>
      </div>
    </div>
  );
}

function Composition({ profile }: { profile: DatasetProfile["profile"] }) {
  const total = Math.max(1, profile.columns);
  const parts = [
    { label: "Numerik", count: profile.numeric_columns, color: "#4f8cff" },
    { label: "Kategorikal", count: profile.categorical_columns, color: "#7fb2ff" },
    { label: "Tanggal", count: profile.datetime_columns, color: "#b9d6ff" },
  ];
  const dominant = [...parts].sort((a, b) => b.count - a.count)[0];
  return (
    <div>
      <h3>Komposisi kolom</h3>
      <p className="card-sub">{profile.columns} kolom terbaca dari file</p>
      <div className="composition">
        {parts.map((part) => (
          <i
            key={part.label}
            style={{ width: `${(part.count / total) * 100}%`, background: part.color }}
            title={`${part.label}: ${part.count}`}
          />
        ))}
      </div>
      <div className="composition-legend">
        {parts.map((part) => (
          <span key={part.label}>
            <i className="legend-dot" style={{ background: part.color }} />
            {part.label} <b>{part.count}</b>
          </span>
        ))}
      </div>
      <span className="composition-note">
        Tipe dominan <b>{dominant.label.toLowerCase()}</b> · {dominant.count} dari {profile.columns} kolom
      </span>
    </div>
  );
}

function QualityCard({
  stats,
  score,
}: {
  stats: DatasetProfile["profile"];
  score: number;
  completeness: number;
  uniqueness: number;
}) {
  return (
    <div className="result-body">
      <div className="card-soft quality-card">
        <Composition profile={stats} />
        <div className="quality-score">
          <span className="quality-label">KESIAPAN DATA</span>
          <QualityRing score={score} />
        </div>
      </div>
    </div>
  );
}

function ResultPanel({ profile }: { profile: DatasetProfile }) {
  const stats = profile.profile;
  const cells = Math.max(1, stats.rows * stats.columns);
  const missingRate = (stats.missing_values / cells) * 100;
  const duplicateRate = (stats.duplicate_rows / Math.max(1, stats.rows)) * 100;
  const completeness = 100 - missingRate;
  const uniqueness = 100 - duplicateRate;
  const score = Math.max(0, Math.min(100, completeness * 0.65 + uniqueness * 0.35));

  const verdict = useMemo(() => {
    if (score >= 88) return "Siap dianalisis";
    if (score >= 70) return "Cukup baik";
    if (score >= 50) return "Perlu perhatian";
    return "Perlu dibersihkan";
  }, [score]);

  return (
    <section className="panel" data-reveal id="dataset-workspace" aria-live="polite" aria-label="Ringkasan dataset">
      <div className="result-head">
        <div className="result-title">
          <span className="file-badge">{fileExtension(profile.filename)}</span>
          <div>
            <span className="overline">RINGKASAN DATA</span>
            <h2 title={profile.filename}>{profile.filename}</h2>
          </div>
        </div>
        <span className="ready-chip">
          <i /> PROFIL SELESAI · {verdict.toUpperCase()}
        </span>
      </div>

      <div className="metric-grid">
        <Metric label="BARIS" value={stats.rows} unit="observasi" />
        <Metric label="KOLOM" value={stats.columns} unit="variabel" />
        <Metric
          label="SEL KOSONG"
          value={stats.missing_values}
          unit={`${missingRate.toFixed(missingRate > 0 && missingRate < 1 ? 2 : 1)}% dari ${numberFormat.format(cells)} sel`}
          percent={missingRate}
          warn={missingRate > 5}
        />
        <Metric
          label="BARIS DUPLIKAT"
          value={stats.duplicate_rows}
          unit={`${duplicateRate.toFixed(1)}% dari baris`}
          percent={duplicateRate}
          warn={duplicateRate > 2}
        />
      </div>

      <QualityCard stats={stats} score={score} completeness={completeness} uniqueness={uniqueness} />

      <div className="table-panel">
        <div className="table-head">
          <h3>Pratinjau data</h3>
          <span>
            10 baris pertama · {profile.column_names.length} kolom · {formatBytes(stats.memory_usage)}
          </span>
        </div>
        <div className="table-scroll">
          <table className="preview-table">
            <thead>
              <tr>
                {profile.column_names.map((column) => (
                  <th key={column}>{column}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {profile.preview.map((row, index) => (
                <tr key={index} style={{ ["--i" as string]: index }}>
                  {profile.column_names.map((column) => (
                    <td key={column} title={String(row[column] ?? "")}>
                      {String(row[column] ?? "—")}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </section>
  );
}
