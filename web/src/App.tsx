import { useEffect, useState, type ReactNode } from "react";

type IconName = "arrow" | "chart" | "database" | "file" | "flask" | "menu" | "search" | "sparkle";

function Icon({ name, size = 20 }: { name: IconName; size?: number }) {
  const paths: Record<IconName, ReactNode> = {
    arrow: <path d="M5 12h14m-6-6 6 6-6 6" />,
    chart: <><path d="M4 19V5m0 14h16" /><path d="m7 14 4-4 3 2 5-6" /></>,
    database: <><ellipse cx="12" cy="5" rx="8" ry="3" /><path d="M4 5v14c0 1.7 3.6 3 8 3s8-1.3 8-3V5M4 12c0 1.7 3.6 3 8 3s8-1.3 8-3" /></>,
    file: <><path d="M13 3H6a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V10z" /><path d="M13 3v7h7M8 15h8M8 18h5" /></>,
    flask: <><path d="M9 3h6m-5 0v6l-5.7 9.4A2.4 2.4 0 0 0 6.4 22h11.2a2.4 2.4 0 0 0 2.1-3.6L14 9V3" /><path d="M8 16h8" /></>,
    menu: <><path d="M4 7h16M4 12h16M4 17h16" /></>,
    search: <><circle cx="10.8" cy="10.8" r="6.8" /><path d="m16 16 4.5 4.5" /></>,
    sparkle: <><path d="m12 3 1.7 5.3L19 10l-5.3 1.7L12 17l-1.7-5.3L5 10l5.3-1.7z" /><path d="m19 15 .8 2.2L22 18l-2.2.8L19 21l-.8-2.2L16 18l2.2-.8z" /></>,
  };

  return (
    <svg
      aria-hidden="true"
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.7"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      {paths[name]}
    </svg>
  );
}

const features = [
  {
    number: "01",
    icon: "database" as const,
    title: "Pahami dataset",
    description: "Lihat struktur, kualitas, nilai kosong, dan pola penting sebelum mulai menganalisis.",
    tag: "DATA WORKSPACE",
  },
  {
    number: "02",
    icon: "chart" as const,
    title: "Pilih metode yang tepat",
    description: "Deteksi jenis tugas dan bandingkan metode machine learning berdasarkan karakter data.",
    tag: "ML INTELLIGENCE",
  },
  {
    number: "03",
    icon: "flask" as const,
    title: "Bangun riset yang kuat",
    description: "Telusuri literatur, petakan tren, dan susun temuan dalam alur kerja yang terarah.",
    tag: "ACADEMIC RESEARCH",
  },
];

const workflow = [
  { step: "01", title: "Temukan", text: "Mulai dari dataset atau pertanyaan riset." },
  { step: "02", title: "Pahami", text: "Kenali kualitas data dan ruang analisisnya." },
  { step: "03", title: "Kembangkan", text: "Rangkai eksperimen, metode, dan laporan." },
];

function App() {
  const [notice, setNotice] = useState("");
  const [menuOpen, setMenuOpen] = useState(false);

  useEffect(() => {
    if (!notice) return undefined;
    const timeout = window.setTimeout(() => setNotice(""), 3600);
    return () => window.clearTimeout(timeout);
  }, [notice]);

  function showWorkspaceNotice() {
    setNotice("Workspace publik sedang disiapkan. Akses analisis akan dibuka setelah akun dan perlindungan data siap.");
    setMenuOpen(false);
  }

  return (
    <main>
      <header className="site-header">
        <a className="brand" href="#top" aria-label="DORA, kembali ke awal">
          <span className="brand-mark"><Icon name="sparkle" size={21} /></span>
          <span className="brand-name">DORA<span className="brand-period">.</span></span>
          <span className="brand-divider" />
          <span className="brand-subtitle">Research workspace</span>
        </a>

        <button
          className="mobile-menu-button"
          type="button"
          aria-label={menuOpen ? "Tutup menu" : "Buka menu"}
          aria-expanded={menuOpen}
          onClick={() => setMenuOpen((open) => !open)}
        >
          <Icon name="menu" />
        </button>

        <nav className={`site-nav${menuOpen ? " is-open" : ""}`} aria-label="Navigasi utama">
          <a href="#fitur" onClick={() => setMenuOpen(false)}>Fitur</a>
          <a href="#alur" onClick={() => setMenuOpen(false)}>Alur kerja</a>
          <a href="#tentang" onClick={() => setMenuOpen(false)}>Tentang DORA</a>
          <button className="nav-login" type="button" onClick={showWorkspaceNotice}>Masuk</button>
        </nav>
      </header>

      <section className="hero section-wrap" id="top">
        <div className="hero-copy">
          <div className="eyebrow"><span className="eyebrow-dot" /> RUANG KERJA RISET DIGITAL</div>
          <h1>Ubah data menjadi <span>arah riset.</span></h1>
          <p className="hero-description">
            Dari memahami dataset hingga merancang eksperimen machine learning dan menelusuri literatur —
            satu alur kerja yang membantu riset bergerak lebih jelas.
          </p>
          <div className="hero-actions">
            <a className="button button-primary" href="#fitur">Jelajahi DORA <Icon name="arrow" size={18} /></a>
            <button className="button button-quiet" type="button" onClick={showWorkspaceNotice}>Masuk workspace</button>
          </div>
          <div className="hero-note"><span className="note-dot" /> Dirancang untuk riset berbasis data</div>
        </div>

        <div className="hero-visual" aria-label="Ilustrasi pratinjau workspace DORA">
          <div className="visual-glow" />
          <div className="preview-window">
            <div className="preview-topbar">
              <div className="window-dots"><i /><i /><i /></div>
              <span>ANALYSIS / OVERVIEW</span>
              <span className="preview-status"><i /> CONTOH</span>
            </div>
            <div className="preview-heading">
              <div>
                <span className="preview-kicker">DATASET SNAPSHOT</span>
                <h2>Student performance</h2>
                <p>student_performance.csv · Pratinjau ilustratif</p>
              </div>
              <span className="file-chip"><Icon name="file" size={16} /> CSV</span>
            </div>
            <div className="preview-metrics">
              <div><span>BARIS</span><strong>18.420</strong><small>observasi</small></div>
              <div><span>KOLOM</span><strong>24</strong><small>variabel</small></div>
              <div><span>KUALITAS</span><strong className="quality-value">Baik <i /></strong><small>siap ditinjau</small></div>
            </div>
            <div className="chart-card">
              <div className="chart-title"><span>Distribusi nilai</span><span>Ringkasan data <b>↗</b></span></div>
              <div className="chart-area">
                <div className="chart-y-labels"><span>100</span><span>75</span><span>50</span><span>25</span></div>
                <div className="chart-plot">
                  <div className="chart-grid"><i /><i /><i /><i /></div>
                  <svg viewBox="0 0 430 125" preserveAspectRatio="none" role="img" aria-label="Grafik contoh distribusi nilai">
                    <defs><linearGradient id="areaFill" x1="0" x2="0" y1="0" y2="1"><stop offset="0%" stopColor="#a5c940" stopOpacity=".28" /><stop offset="100%" stopColor="#a5c940" stopOpacity="0" /></linearGradient></defs>
                    <path d="M0 104 C35 99 44 73 78 76 S127 104 159 83 S204 48 237 62 S280 89 315 52 S364 34 389 40 S416 29 430 16 V125 H0Z" fill="url(#areaFill)" />
                    <path d="M0 104 C35 99 44 73 78 76 S127 104 159 83 S204 48 237 62 S280 89 315 52 S364 34 389 40 S416 29 430 16" fill="none" stroke="#86a92b" strokeWidth="3" vectorEffect="non-scaling-stroke" />
                    <circle cx="315" cy="52" r="4.5" fill="#fff" stroke="#86a92b" strokeWidth="3" vectorEffect="non-scaling-stroke" />
                  </svg>
                  <div className="chart-x-labels"><span>0–20</span><span>21–40</span><span>41–60</span><span>61–80</span><span>81–100</span></div>
                </div>
              </div>
            </div>
            <div className="preview-footer"><span><i /> DATA PROFILING</span><span>Analisis contoh <b>→</b></span></div>
          </div>
          <div className="floating-note"><span className="floating-icon"><Icon name="sparkle" size={17} /></span><span><b>Mulai dengan konteks</b><small>Setiap analisis punya arah</small></span></div>
          <span className="visual-index">D / 01</span>
        </div>
      </section>

      <div className="trust-strip section-wrap" id="tentang">
        <span>DIRANCANG UNTUK PROSES RISET YANG LEBIH TERSTRUKTUR</span>
        <div><i /> Dataset <b>·</b> Machine learning <b>·</b> Literatur <b>·</b> Laporan</div>
      </div>

      <section className="features-section section-wrap" id="fitur">
        <div className="section-heading">
          <div><span className="section-kicker">SATU WORKSPACE, BANYAK KEMUNGKINAN</span><h2>Riset punya banyak bagian.<br /><span>DORA menghubungkannya.</span></h2></div>
          <p>Alat yang mendukung langkah penting riset, dengan hasil yang lebih mudah dipahami dan ditindaklanjuti.</p>
        </div>
        <div className="feature-grid">
          {features.map((feature) => (
            <article className="feature-card" key={feature.number}>
              <div className="feature-top"><span className="feature-icon"><Icon name={feature.icon} size={21} /></span><span className="feature-number">{feature.number}</span></div>
              <span className="feature-tag">{feature.tag}</span>
              <h3>{feature.title}</h3>
              <p>{feature.description}</p>
              <button className="text-link" type="button" onClick={showWorkspaceNotice}>Lihat alur <Icon name="arrow" size={16} /></button>
            </article>
          ))}
        </div>
      </section>

      <section className="workflow-section" id="alur">
        <div className="section-wrap workflow-inner">
          <div className="workflow-heading"><span className="section-kicker">DARI PERTANYAAN KE TEMUAN</span><h2>Mulai dari yang ingin<br />kamu <span>pahami.</span></h2><p>Riset tidak selalu berjalan lurus. DORA membantu menjaga konteks saat kamu berpindah dari data ke metode, lalu ke kesimpulan.</p></div>
          <div className="workflow-list">
            {workflow.map((item, index) => (
              <div className="workflow-step" key={item.step}>
                <span className="workflow-index">{item.step}</span>
                <div><h3>{item.title}</h3><p>{item.text}</p></div>
                {index < workflow.length - 1 && <span className="workflow-connector" aria-hidden="true" />}
              </div>
            ))}
          </div>
        </div>
      </section>

      <section className="closing section-wrap">
        <div className="closing-mark"><Icon name="sparkle" size={25} /></div>
        <span className="section-kicker">DORA · DATASET RESEARCH</span>
        <h2>Bangun riset dengan<br /><span>lebih banyak kejelasan.</span></h2>
        <p>Workspace publik sedang disiapkan. Kami memulai dengan pengalaman riset yang aman dan dapat diandalkan.</p>
        <button className="button button-primary" type="button" onClick={showWorkspaceNotice}>Ikuti perkembangan <Icon name="arrow" size={18} /></button>
      </section>

      <footer className="site-footer section-wrap">
        <a className="brand footer-brand" href="#top"><span className="brand-mark"><Icon name="sparkle" size={18} /></span><span className="brand-name">DORA<span className="brand-period">.</span></span></a>
        <span>Ruang kerja riset berbasis data.</span>
        <span>© 2026 DORA</span>
      </footer>

      {notice && <div className="toast" role="status" aria-live="polite"><span className="toast-spinner" /><span>{notice}</span><button type="button" aria-label="Tutup pemberitahuan" onClick={() => setNotice("")}>×</button></div>}
    </main>
  );
}

export default App;
