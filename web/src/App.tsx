import { useEffect, useState } from "react";
import { Ambient } from "./components/Ambient";
import { Icon } from "./components/Icons";
import type { IconName } from "./components/Icons";
import { EmptyState } from "./components/PageHeader";
import { AnalysisPage } from "./pages/Analysis";
import { DatasetPage } from "./pages/Dataset";
import { DatasetSearchPage } from "./pages/DatasetSearch";
import { MlPage } from "./pages/Ml";
import { Overview } from "./pages/Overview";
import { ReportPage } from "./pages/Report";
import { ResearchPage } from "./pages/Research";
import { previewAnalysis, previewCatalog } from "./dev/preview";
import type { CleanReport, DatasetAnalysis, MlCatalog, ResearchResult } from "./lib/types";

const apiBase = (import.meta.env.VITE_API_URL || "http://127.0.0.1:8000").replace(/\/$/, "");
const previewParam = new URLSearchParams(window.location.search).get("preview");
const previewMode = import.meta.env.DEV && previewParam !== null;
const previewWithData = previewMode && previewParam !== "empty";

type PageKey = "overview" | "dataset" | "analysis" | "ml" | "research" | "gap" | "report" | "datasetsearch";

type NavItem = {
  key: PageKey;
  label: string;
  icon: IconName;
};

type DesktopItem = {
  label: string;
  icon: IconName;
};

const NAV_WORKSPACE: NavItem[] = [
  { key: "datasetsearch", label: "Dataset Search", icon: "search" },
  { key: "dataset", label: "Dataset", icon: "dataset" },
  { key: "analysis", label: "Analysis", icon: "analysis" },
  { key: "ml", label: "ML Intelligence", icon: "ml" },
  { key: "overview", label: "Overview", icon: "overview" },
];

const NAV_RESEARCH: NavItem[] = [
  { key: "research", label: "Academic Research", icon: "research" },
  { key: "gap", label: "Research Gap", icon: "gap" },
  { key: "report", label: "Research Report", icon: "report" },
];

/* Desktop-only items that require local execution / credentials */
const DESKTOP_TOOLS: DesktopItem[] = [
  { label: "Editor Analysis", icon: "editor" },
  { label: "Node Editor", icon: "node" },
];

const PAGE_TITLES: Record<PageKey, string> = {
  overview: "Overview",
  dataset: "Dataset",
  analysis: "Analysis",
  ml: "ML Intelligence",
  research: "Academic Research",
  gap: "Research Gap",
  report: "Research Report",
  datasetsearch: "Dataset Search",
};

const initialPageParam = new URLSearchParams(window.location.search).get("page") as PageKey | null;

export default function App() {
  const [page, setPage] = useState<PageKey>(
    initialPageParam && PAGE_TITLES[initialPageParam] ? initialPageParam : "overview"
  );
  const [analysis, setAnalysis] = useState<DatasetAnalysis | null>(previewWithData ? previewAnalysis : null);
  const [sourceFile, setSourceFile] = useState<File | null>(null);
  const [catalog, setCatalog] = useState<MlCatalog | null>(previewWithData ? previewCatalog : null);
  const [catalogMessage, setCatalogMessage] = useState("");
  const [busy, setBusy] = useState(false);
  const [cleaning, setCleaning] = useState(false);
  const [error, setError] = useState("");
  const [cleanReport, setCleanReport] = useState<CleanReport | null>(null);
  const [research, setResearch] = useState<ResearchResult | null>(null);
  const [researchBusy, setResearchBusy] = useState(false);
  const [researchError, setResearchError] = useState("");

  useEffect(() => {
    if (previewMode) return;
    let active = true;
    fetch(`${apiBase}/api/ml/catalog`)
      .then(async (response) => {
        const body = await response.json();
        if (!response.ok) throw new Error(body.detail || "Katalog metode belum dapat dimuat.");
        return body as MlCatalog;
      })
      .then((body) => {
        if (!active) return;
        setCatalog(body);
        setCatalogMessage("");
      })
      .catch((cause: unknown) => {
        if (!active) return;
        setCatalogMessage(cause instanceof Error ? cause.message : "Katalog metode belum dapat dimuat.");
      });
    return () => {
      active = false;
    };
  }, []);

  useEffect(() => {
    window.scrollTo({ top: 0, behavior: "auto" });
  }, [page]);

  async function analyzeDataset(file: File) {
    setError("");
    setBusy(true);
    try {
      const query = new URLSearchParams({ filename: file.name });
      const response = await fetch(`${apiBase}/api/datasets/analyze?${query}`, {
        method: "POST",
        headers: { "Content-Type": "application/octet-stream" },
        body: file,
      });
      const body = await response.json();
      if (!response.ok) throw new Error(body.detail || "Dataset belum dapat diproses.");
      setAnalysis(body as DatasetAnalysis);
      setSourceFile(file);
      setCleanReport(null);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Dataset belum dapat diproses.");
    } finally {
      setBusy(false);
    }
  }

  async function cleanDataset(strategy: "impute" | "drop_rows") {
    if (!sourceFile) {
      setError("File sumber sudah tidak ada di memori browser. Unggah ulang dataset.");
      setPage("overview");
      return;
    }
    setCleaning(true);
    setError("");
    try {
      const query = new URLSearchParams({ filename: sourceFile.name, strategy });
      const response = await fetch(`${apiBase}/api/datasets/clean?${query}`, {
        method: "POST",
        headers: { "Content-Type": "application/octet-stream" },
        body: sourceFile,
      });
      if (!response.ok) {
        const body = await response.json().catch(() => ({ detail: "Dataset belum dapat dibersihkan." }));
        throw new Error(body.detail || "Dataset belum dapat dibersihkan.");
      }
      const reportHeader = response.headers.get("X-Dora-Report");
      const blob = await response.blob();
      const filename = filenameFromDisposition(response.headers.get("Content-Disposition")) || "dataset_bersih.csv";
      downloadBlob(blob, filename);
      if (reportHeader) setCleanReport(JSON.parse(decodeBase64(reportHeader)) as CleanReport);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Dataset belum dapat dibersihkan.");
    } finally {
      setCleaning(false);
    }
  }

  async function runResearch(query: string, limit: number) {
    setResearchBusy(true);
    setResearchError("");
    try {
      const response = await fetch(`${apiBase}/api/research/search`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query,
          limit,
          fingerprint: analysis ? { representation: analysis.fingerprint.representation } : null,
          ml_result: analysis ? analysis.ml_intelligence : null,
        }),
      });
      const body = await response.json();
      if (!response.ok) throw new Error(body.detail || "Pencarian literatur gagal.");
      setResearch(body as ResearchResult);
    } catch (cause) {
      setResearchError(cause instanceof Error ? cause.message : "Pencarian literatur gagal.");
    } finally {
      setResearchBusy(false);
    }
  }

  const activeDataset = analysis !== null;

  return (
    <>
      <Ambient />
      <div className="app">
        <aside className="sidebar">
          <div className="sidebar-panel">
            <div className="brand">
              <img className="brand-logo" src="/dora-mark.png" alt="" width={44} height={54} />
              <div>
                <div className="brand-name">
                  DORA<span>.</span>
                </div>
                <div className="brand-sub">Data Overview &amp; Research Assistant</div>
              </div>
            </div>

            <nav aria-label="Navigasi workspace">
              <div className="nav-group">
                <div className="nav-label">RUANG KERJA</div>
                {NAV_WORKSPACE.map((item) => (
                  <NavButton
                    key={item.key}
                    item={item}
                    active={page === item.key}
                    disabled={
                      item.key !== "overview" &&
                      item.key !== "datasetsearch" &&
                      !activeDataset
                    }
                    onClick={() => setPage(item.key)}
                  />
                ))}
              </div>

              <div className="nav-group">
                <div className="nav-label">RISET &amp; PUBLIKASI</div>
                {NAV_RESEARCH.map((item) => (
                  <NavButton
                    key={item.key}
                    item={item}
                    active={page === item.key}
                    disabled={item.key === "report" && !activeDataset}
                    onClick={() => setPage(item.key)}
                  />
                ))}
              </div>

              <div className="nav-group">
                <div className="nav-label">ALAT TAMBAHAN</div>
                {DESKTOP_TOOLS.map((item) => (
                  <DesktopOnly key={item.label} item={item} />
                ))}
              </div>
            </nav>

            <div className="sidebar-foot">
              <div className="side-status">
                <span className="side-status-label">STATUS</span>
                <strong title={analysis?.filename}>{analysis ? analysis.filename : "Belum ada dataset"}</strong>
                <span>
                  {analysis
                    ? `${new Intl.NumberFormat("id-ID").format(analysis.profile.rows)} baris · ${analysis.profile.columns} kolom`
                    : "Siap menerima file"}
                </span>
              </div>
            </div>
          </div>
        </aside>

        <div className="main">
          <header className="topbar">
            <div className="crumbs">
              RUANG KERJA <i /> <b>{PAGE_TITLES[page].toUpperCase()}</b>
              {analysis ? (
                <span className="crumb-file">
                  <i /> <b>{analysis.filename}</b>
                </span>
              ) : null}
            </div>
            <div className="topbar-actions">
              {analysis ? (
                <span className="status-pill">
                  <i /> DATASET AKTIF
                </span>
              ) : (
                <span className="status-pill is-idle">
                  <i /> SIAP MENERIMA DATASET
                </span>
              )}
              {analysis ? (
                <button
                  className="ghost-button"
                  type="button"
                  onClick={() => {
                    setAnalysis(null);
                    setSourceFile(null);
                    setCleanReport(null);
                    setError("");
                    setPage("overview");
                  }}
                >
                  <Icon name="upload" size={14} /> Ganti dataset
                </button>
              ) : null}
            </div>
          </header>

          {error ? (
            <p className="error-note banner" role="alert">
              <Icon name="alert" size={14} /> {error}
            </p>
          ) : null}

          <div className="page-shell" key={page}>
            {page === "overview" ? (
              <Overview
                profile={analysis}
                catalog={catalog}
                busy={busy}
                error={error}
                onAnalyze={analyzeDataset}
                onOpenAnalysis={() => setPage("analysis")}
                onOpenDatasetSearch={() => setPage("datasetsearch")}
              />
            ) : null}

            {page === "datasetsearch" ? (
              <DatasetSearchPage onGoToUpload={() => setPage("overview")} />
            ) : null}

            {page === "dataset" ? (
              analysis ? (
                <DatasetPage
                  data={analysis}
                  hasSourceFile={Boolean(sourceFile)}
                  cleaning={cleaning}
                  cleanReport={cleanReport}
                  onClean={cleanDataset}
                />
              ) : (
                <NoDataset onUpload={() => setPage("overview")} />
              )
            ) : null}

            {page === "analysis" ? (
              analysis ? <AnalysisPage data={analysis} /> : <NoDataset onUpload={() => setPage("overview")} />
            ) : null}

            {page === "ml" ? (
              analysis ? <MlPage data={analysis} catalog={catalog} /> : <NoDataset onUpload={() => setPage("overview")} />
            ) : null}

            {page === "research" || page === "gap" ? (
              <ResearchPage
                data={analysis}
                result={research}
                busy={researchBusy}
                error={researchError}
                initialTab={page === "gap" ? "gaps" : "papers"}
                onSearch={runResearch}
              />
            ) : null}

            {page === "report" ? (
              analysis ? <ReportPage data={analysis} /> : <NoDataset onUpload={() => setPage("overview")} />
            ) : null}
          </div>

          {page === "overview" && catalogMessage ? (
            <p className="ml-foot" role="status">
              {catalogMessage}
            </p>
          ) : null}
        </div>
      </div>
    </>
  );
}

function NavButton({
  item,
  active,
  disabled,
  onClick,
}: {
  item: NavItem;
  active: boolean;
  disabled: boolean;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      className={`nav-item${active ? " is-active" : ""}${disabled ? " is-dim" : ""}`}
      aria-current={active ? "page" : undefined}
      disabled={disabled}
      title={disabled ? "Unggah dataset dulu untuk membuka halaman ini" : item.label}
      onClick={onClick}
    >
      <Icon name={item.icon} size={16} />
      {item.label}
    </button>
  );
}

function DesktopOnly({ item }: { item: DesktopItem }) {
  return (
    <div className="nav-item is-locked" title="Fitur ini hanya tersedia di aplikasi desktop DORA">
      <Icon name={item.icon} size={16} />
      {item.label}
      <span className="nav-tag">APP</span>
    </div>
  );
}

function NoDataset({ onUpload }: { onUpload: () => void }) {
  return (
    <EmptyState
      title="Belum ada dataset di sesi ini"
      description="Unggah satu file terlebih dahulu. Analisis, pembersihan, dan laporan memakai dataset yang sama."
      onUpload={onUpload}
    />
  );
}

function filenameFromDisposition(header: string | null): string {
  if (!header) return "";
  const match = /filename="?([^";]+)"?/i.exec(header);
  return match ? match[1] : "";
}

function decodeBase64(value: string): string {
  const binary = atob(value);
  const bytes = Uint8Array.from(binary, (character) => character.charCodeAt(0));
  return new TextDecoder().decode(bytes);
}

function downloadBlob(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  // Revoking immediately can cancel the download in some browsers.
  window.setTimeout(() => URL.revokeObjectURL(url), 20_000);
}
