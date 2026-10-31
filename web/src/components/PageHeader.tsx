import type { ReactNode } from "react";

export function PageHeader({
  overline,
  title,
  lead,
  actions,
  meta,
}: {
  overline: string;
  title: string;
  lead: string;
  actions?: ReactNode;
  meta?: ReactNode;
}) {
  return (
    <header className="page-head" data-reveal>
      <div className="page-head-copy">
        <span className="overline">{overline}</span>
        <h1 className="page-title">{title}</h1>
        <p className="page-lead">{lead}</p>
        {meta ? <div className="page-meta">{meta}</div> : null}
      </div>
      {actions ? <div className="page-head-actions">{actions}</div> : null}
    </header>
  );
}

export function EmptyState({
  title,
  description,
  onUpload,
}: {
  title: string;
  description: string;
  onUpload: () => void;
}) {
  return (
    <section className="empty-state panel" data-reveal>
      <span className="empty-icon">
        <svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round">
          <path d="M12 16.5V4.5" />
          <path d="m7.5 9 4.5-4.5L16.5 9" />
          <path d="M4.5 16.5v2a2 2 0 0 0 2 2h11a2 2 0 0 0 2-2v-2" />
        </svg>
      </span>
      <h2>{title}</h2>
      <p>{description}</p>
      <button className="primary-button" type="button" onClick={onUpload}>
        Ke halaman unggah
      </button>
    </section>
  );
}
