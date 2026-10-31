import { Icon } from "../components/Icons";
import { CatalogPanel, MlPanel } from "../components/MlPanels";
import { PageHeader } from "../components/PageHeader";
import { useScrollReveal } from "../lib/motion";
import { formatInt } from "../lib/format";
import type { DatasetAnalysis, MlCatalog } from "../lib/types";

export function MlPage({ data, catalog }: { data: DatasetAnalysis; catalog: MlCatalog | null }) {
  useScrollReveal(data.filename);
  const recommendations = data.ml_intelligence.recommendation.recommendations ?? [];
  const topScore = recommendations.reduce((best, item) => Math.max(best, item.score ?? 0), 0);

  return (
    <div className="page">
      <PageHeader
        overline="ML INTELLIGENCE"
        title="Arah metode & kesiapan model"
        lead="Task detector membaca struktur kolom, lalu knowledge base DORA memberi peringkat metode yang masuk akal untuk dataset ini."
        meta={
          <>
            <span className="meta-chip">{formatInt(recommendations.length)} metode disarankan</span>
            <span className="meta-chip">{formatInt(Object.keys(catalog?.methods ?? {}).length)} algoritma di katalog</span>
            <span className="meta-chip is-warning">Estimasi · belum dilatih</span>
          </>
        }
      />

      <MlPanel profile={data} showMethods={false} />

      <section className="panel" data-reveal>
        <div className="table-head">
          <h3>Peringkat metode</h3>
          <span>Skor relatif terhadap dataset ini</span>
        </div>
        <div className="panel-body">
          <ul className="rank-list">
            {recommendations.map((method, index) => {
              const score = Math.max(0, Math.min(100, method.score ?? 0));
              return (
                <li key={method.method_id}>
                  <span className="rank-index">{String(index + 1).padStart(2, "0")}</span>
                  <div className="rank-body">
                    <div className="rank-head">
                      <b>{method.method}</b>
                      <span>{method.category || "Machine learning"}</span>
                    </div>
                    <p>{method.description}</p>
                    <div className="rank-track">
                      <i style={{ width: `${score}%` }} />
                    </div>
                  </div>
                  <span className="rank-score" title="Skor saran">
                    {Math.round(score)}
                  </span>
                </li>
              );
            })}
            {!recommendations.length ? (
              <li className="table-note">Belum ada metode yang dapat disarankan untuk dataset ini.</li>
            ) : null}
          </ul>
          {topScore ? null : (
            <p className="ml-foot">
              <Icon name="alert" size={13} /> Skor rendah karena task utama belum terdeteksi kuat.
            </p>
          )}
        </div>
      </section>

      {catalog ? <CatalogPanel catalog={catalog} /> : null}
    </div>
  );
}
