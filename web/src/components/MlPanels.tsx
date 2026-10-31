import { useMemo, useState } from "react";
import { Icon } from "./Icons";
import type { DatasetProfile, MlCatalog } from "../lib/types";

export function MlPanel({ profile, showMethods = true }: { profile: DatasetProfile; showMethods?: boolean }) {
  const insights = profile.ml_intelligence;
  const primary = insights.task_detection.primary_task ?? null;
  const tasks = insights.task_detection.tasks ?? [];
  const recommendations = (insights.recommendation.recommendations ?? []).slice(0, 3);
  const confidence = typeof primary?.score === "number" ? Math.max(0, Math.min(100, primary.score)) : null;

  return (
    <section className="panel" data-reveal style={{ ["--reveal-delay" as string]: "70ms" }} aria-labelledby="ml-title">
      <div className="ml-head">
        <div>
          <span className="overline">ML INTELLIGENCE</span>
          <h2 id="ml-title">Arah analisis yang mungkin</h2>
        </div>
        <span className="estimate-tag">ESTIMASI · BELUM DIEVALUASI</span>
      </div>

      <div className="ml-grid">
        <article className="task-primary">
          <span className="label">ESTIMASI TASK UTAMA</span>
          <h3>{primary?.label || "Belum terdeteksi"}</h3>
          {primary?.target ? (
            <p>
              Kandidat target: <b>{primary.target}</b>
            </p>
          ) : (
            <p>Struktur kolom belum menunjukkan target yang jelas.</p>
          )}
          {confidence !== null ? (
            <div className="confidence">
              <div className="confidence-bar">
                <i style={{ width: `${confidence}%` }} />
              </div>
              <b>{Math.round(confidence)}%</b>
            </div>
          ) : null}
        </article>

        <article className="card-soft">
          <h3>Task yang dapat dipertimbangkan</h3>
          <p className="card-sub">{insights.task_detection.message || "Berdasarkan tipe dan sebaran kolom."}</p>
          <div className="task-chips">
            {tasks.length ? (
              tasks.slice(0, 5).map((task) => (
                <span className={`chip${task.task === primary?.task ? " is-primary" : ""}`} key={task.task}>
                  {task.label || task.task.replaceAll("_", " ")}
                  {typeof task.score === "number" ? <b>{Math.round(task.score)}%</b> : null}
                </span>
              ))
            ) : (
              <span className="chip">Belum ada task yang cukup kuat</span>
            )}
          </div>
        </article>
      </div>

      {showMethods ? (
        <>
          <div className="method-grid">
            {recommendations.length ? (
              recommendations.map((method, index) => (
                <article className="method" key={method.method_id}>
                  <span className="method-rank">{String(index + 1).padStart(2, "0")}</span>
                  <span className="method-cat">{method.category || "Machine learning"}</span>
                  <h4>{method.method}</h4>
                  <p>{method.description}</p>
                  {typeof method.score === "number" ? (
                    <span className="method-score">Skor saran {Math.round(method.score)} / 100</span>
                  ) : null}
                </article>
              ))
            ) : (
              <article className="method">
                <span className="method-cat">Belum ada</span>
                <h4>Metode belum dapat disarankan</h4>
                <p>Task utama belum terdeteksi dengan cukup kuat pada dataset ini.</p>
              </article>
            )}
          </div>

          <p className="ml-foot">
            Saran berasal dari task detector dan knowledge base metode DORA. Model belum dilatih atau dibandingkan pada
            dataset ini — verifikasi tetap diperlukan.
          </p>
        </>
      ) : null}
    </section>
  );
}

export function CatalogPanel({ catalog }: { catalog: MlCatalog }) {
  const [selectedTask, setSelectedTask] = useState("all");
  const labels = useMemo(() => new Map(catalog.tasks.map((task) => [task.id, task.label])), [catalog.tasks]);
  const entries = Object.entries(catalog.methods).filter(
    ([, method]) => selectedTask === "all" || method.suitable_tasks?.includes(selectedTask),
  );

  return (
    <details className="panel" data-reveal>
      <summary className="catalog-summary">
        <span>
          <span className="overline">KATALOG METODE</span>
          <h2 className="catalog-title">{Object.keys(catalog.methods).length} algoritma dari knowledge base</h2>
        </span>
        <span className="ghost-button">
          <Icon name="spark" size={15} /> Jelajahi
        </span>
      </summary>
      <div className="panel-body catalog-body">
        <label className="catalog-filter">
          Filter berdasarkan task
          <select value={selectedTask} onChange={(event) => setSelectedTask(event.currentTarget.value)}>
            <option value="all">Semua task</option>
            {catalog.tasks.map((task) => (
              <option value={task.id} key={task.id}>
                {task.label}
              </option>
            ))}
          </select>
        </label>
        <div className="method-grid catalog-grid">
          {entries.map(([methodId, method]) => (
            <article className="method" key={methodId}>
              <span className="method-cat">{method.category || "Machine learning"}</span>
              <h4>{method.name}</h4>
              <p>{method.description}</p>
              <div className="task-chips catalog-chips">
                {(method.suitable_tasks ?? []).slice(0, 3).map((task) => (
                  <span className="chip is-small" key={task}>
                    {labels.get(task) ?? task}
                  </span>
                ))}
              </div>
            </article>
          ))}
        </div>
      </div>
    </details>
  );
}
