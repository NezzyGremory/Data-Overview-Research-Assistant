import type { ReactNode } from "react";

export type IconName =
  | "logo"
  | "overview"
  | "dataset"
  | "analysis"
  | "ml"
  | "lock"
  | "upload"
  | "shield"
  | "check"
  | "arrow"
  | "file"
  | "logout"
  | "alert"
  | "search"
  | "spark"
  | "research"
  | "gap"
  | "report"
  | "editor"
  | "node"
  | "download"
  | "print"
  | "table"
  | "wrench"
  | "monitor";

const paths: Record<IconName, ReactNode> = {
  logo: (
    <>
      <path d="M12 3.2 14.1 9l5.8 2.1-5.8 2.1L12 19l-2.1-5.8L4.1 11.1 9.9 9z" />
      <path d="M18.4 15.6l.8 2.2 2.2.8-2.2.8-.8 2.2-.8-2.2-2.2-.8 2.2-.8z" />
    </>
  ),
  overview: (
    <>
      <rect x="3.5" y="3.5" width="7" height="7" rx="2" />
      <rect x="13.5" y="3.5" width="7" height="7" rx="2" />
      <rect x="3.5" y="13.5" width="7" height="7" rx="2" />
      <rect x="13.5" y="13.5" width="7" height="7" rx="2" />
    </>
  ),
  dataset: (
    <>
      <ellipse cx="12" cy="5.5" rx="7.5" ry="2.8" />
      <path d="M4.5 5.5v13c0 1.6 3.4 2.8 7.5 2.8s7.5-1.2 7.5-2.8v-13" />
      <path d="M4.5 12c0 1.6 3.4 2.8 7.5 2.8s7.5-1.2 7.5-2.8" />
    </>
  ),
  analysis: (
    <>
      <path d="M4 20V4" />
      <path d="M4 20h16" />
      <path d="m7.5 15 3.5-4 3 2.2L19 7.5" />
    </>
  ),
  ml: (
    <>
      <path d="M11 3.5 12.6 8l4.4 1.6L12.6 11 11 15.5 9.4 11 5 9.6 9.4 8z" />
      <path d="M17.6 13.4l.9 2.1 2.1.9-2.1.9-.9 2.1-.9-2.1-2.1-.9 2.1-.9z" />
    </>
  ),
  lock: (
    <>
      <rect x="4.5" y="10" width="15" height="10.5" rx="2.6" />
      <path d="M8.2 10V7.6a3.8 3.8 0 0 1 7.6 0V10" />
    </>
  ),
  upload: (
    <>
      <path d="M12 16.5V4.5" />
      <path d="m7.5 9 4.5-4.5L16.5 9" />
      <path d="M4.5 16.5v2a2 2 0 0 0 2 2h11a2 2 0 0 0 2-2v-2" />
    </>
  ),
  shield: (
    <>
      <path d="M12 3.2 19 6v6c0 4-3 7-7 8.8C8 19 5 16 5 12V6z" />
      <path d="m9.2 11.8 2 2 3.6-3.8" />
    </>
  ),
  check: <path d="m5 12.5 4.5 4.5L19 6.5" />,
  arrow: (
    <>
      <path d="M5 12h14" />
      <path d="m13 6 6 6-6 6" />
    </>
  ),
  file: (
    <>
      <path d="M13.5 3.5H7a2 2 0 0 0-2 2v13a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V9z" />
      <path d="M13.5 3.5V9H19" />
    </>
  ),
  logout: (
    <>
      <path d="M15 5.5h2.6a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2H15" />
      <path d="M10 8.5 6 12l4 3.5" />
      <path d="M6 12h9" />
    </>
  ),
  alert: (
    <>
      <path d="M12 4.5 3.6 19h16.8z" />
      <path d="M12 10v4.2" />
      <path d="M12 16.8v.2" />
    </>
  ),
  search: (
    <>
      <circle cx="10.8" cy="10.8" r="6.5" />
      <path d="m15.6 15.6 4.4 4.4" />
    </>
  ),
  spark: (
    <>
      <path d="M12 3.5 13.8 9l5.5 1.8-5.5 1.8L12 18l-1.8-5.4-5.5-1.8L10.2 9z" />
    </>
  ),
  research: (
    <>
      <path d="M4.5 5.5a2 2 0 0 1 2-2H11v16H6.5a2 2 0 0 1-2-2z" />
      <path d="M19.5 5.5a2 2 0 0 0-2-2H13v16h4.5a2 2 0 0 0 2-2z" />
    </>
  ),
  gap: (
    <>
      <circle cx="12" cy="12" r="7.5" />
      <circle cx="12" cy="12" r="2.6" />
      <path d="M12 2.2v3M12 18.8v3M2.2 12h3M18.8 12h3" />
    </>
  ),
  report: (
    <>
      <path d="M14 3.5H7a2 2 0 0 0-2 2v13a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8.5z" />
      <path d="M14 3.5v5h5" />
      <path d="M8.5 13h7M8.5 16.5h4.5" />
    </>
  ),
  editor: (
    <>
      <path d="M4 7.5h10M18 7.5h2M4 16.5h4M12 16.5h8" />
      <circle cx="16" cy="7.5" r="2" />
      <circle cx="10" cy="16.5" r="2" />
    </>
  ),
  node: (
    <>
      <circle cx="6" cy="6" r="2.4" />
      <circle cx="18" cy="8" r="2.4" />
      <circle cx="10" cy="18" r="2.4" />
      <path d="m8.2 7.2 7.6.6M7.4 8.1l1.9 7.6M16.8 10l-5.4 6.2" />
    </>
  ),
  download: (
    <>
      <path d="M12 4.5V15" />
      <path d="m7.5 10.5 4.5 4.5 4.5-4.5" />
      <path d="M4.5 19.5h15" />
    </>
  ),
  print: (
    <>
      <path d="M7 8.5V4.5h10v4" />
      <rect x="4" y="8.5" width="16" height="7.5" rx="2" />
      <path d="M7 14h10v6H7z" />
    </>
  ),
  table: (
    <>
      <rect x="3.5" y="4.5" width="17" height="15" rx="2.4" />
      <path d="M3.5 9.5h17M3.5 14.5h17M9.5 9.5v10" />
    </>
  ),
  wrench: (
    <>
      <path d="M14.7 6.1a3.9 3.9 0 0 0 5.2 5.2L11 20.2a2.6 2.6 0 0 1-3.7-3.7z" />
      <path d="m5.5 4.5 2.2 2.2M4 7.6l2.2-2.2" />
    </>
  ),
  monitor: (
    <>
      <rect x="3" y="4.5" width="18" height="12" rx="2" />
      <path d="M9 20h6M12 16.5V20" />
    </>
  ),
};

export function Icon({ name, size = 18 }: { name: IconName; size?: number }) {
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
