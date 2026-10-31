const integer = new Intl.NumberFormat("id-ID");
const oneDecimal = new Intl.NumberFormat("id-ID", { maximumFractionDigits: 1 });
const threeDecimal = new Intl.NumberFormat("id-ID", { maximumFractionDigits: 3 });

export function formatInt(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "—";
  return integer.format(Math.round(value));
}

export function formatNumber(value: number | null | undefined, digits = 2): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "—";
  return (digits >= 3 ? threeDecimal : oneDecimal).format(value);
}

export function formatPercent(value: number | null | undefined, digits = 1): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "—";
  return `${formatNumber(value, digits)}%`;
}

export function formatBytes(value: number | null | undefined): string {
  if (!Number.isFinite(value ?? NaN) || (value ?? 0) <= 0) return "0 B";
  const size = value as number;
  if (size < 1024) return `${Math.round(size)} B`;
  if (size < 1024 * 1024) return `${(size / 1024).toFixed(1)} KB`;
  return `${(size / (1024 * 1024)).toFixed(1)} MB`;
}

export function fileExtension(filename: string): string {
  const clean = filename.split(/[\\/]/).pop() ?? filename;
  const dot = clean.lastIndexOf(".");
  return dot > 0 ? clean.slice(dot + 1).toUpperCase().slice(0, 4) : "DATA";
}

export function shortName(name: string, limit = 16): string {
  return name.length > limit ? `${name.slice(0, limit - 1)}…` : name;
}
