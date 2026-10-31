export type DatasetProfile = {
  filename: string;
  profile: {
    rows: number;
    columns: number;
    memory_usage: number;
    duplicate_rows: number;
    missing_values: number;
    numeric_columns: number;
    categorical_columns: number;
    datetime_columns: number;
  };
  column_names: string[];
  preview: Record<string, string | number | boolean | null>[];
  ml_intelligence: MlInsights;
};

export type MlTaskCandidate = {
  task: string;
  label?: string;
  score?: number;
  target?: string | null;
  reasons?: string[];
};

export type MlRecommendation = {
  method_id: string;
  method: string;
  category?: string;
  score?: number;
  description?: string;
};

export type MlInsights = {
  status: string;
  task_detection: {
    primary_task?: MlTaskCandidate | null;
    tasks: MlTaskCandidate[];
    message?: string;
  };
  recommendation: {
    recommendations: MlRecommendation[];
    recommendation_count: number;
    message: string;
  };
  evaluation_status: string;
};

export type MlMethod = {
  name: string;
  category?: string;
  description: string;
  suitable_tasks: string[];
};

export type MlCatalog = {
  tasks: { id: string; label: string }[];
  methods: Record<string, MlMethod>;
};

export type ColumnStat = {
  column: string;
  dtype: string;
  missing_count: number;
  missing_percentage: number;
  unique_count: number;
  unique_percentage: number;
  mean?: number | null;
  median?: number | null;
  std?: number | null;
  min?: number | null;
  max?: number | null;
  q1?: number | null;
  q3?: number | null;
  skewness?: number | null;
  category_count?: number;
  top_values?: { value: string; frequency: number }[];
};

export type MissingStat = {
  column: string;
  missing_count: number;
  missing_percentage: number;
  zero_count: number;
  zero_percentage: number;
};

export type OutlierStat = {
  column: string;
  outlier_count: number;
  outlier_percentage: number;
  lower_bound: number;
  upper_bound: number;
};

export type QualityItem = {
  category: string;
  title: string;
  severity: "critical" | "warning" | "info";
  problem: string;
  magnitude: string;
  impact: string;
  options: string[];
  affected_columns: string[];
  metrics: Record<string, unknown>;
};

export type FingerprintColumn = {
  name: string;
  semantic_type?: string;
  potential_id?: boolean;
  target_candidate?: boolean;
  missing_percentage?: number;
  unique_count?: number;
};

export type DatasetFingerprint = {
  hash: string;
  representation: {
    characteristics?: Record<string, unknown>;
    target_candidates?: string[];
    keywords?: string[];
    categories?: Record<string, unknown>;
    column_signature?: FingerprintColumn[];
  };
};

export type CleanReport = {
  strategy: string;
  rows_before: number;
  rows_after: number;
  rows_dropped: number;
  percent_dropped: number;
  columns_after: number;
  missing_before: number;
  missing_after: number;
  imputed_columns: string[];
  cells_imputed: number;
  dropped_empty_columns: string[];
  warnings: string[];
};

export type DistributionBin = {
  label: string;
  count: number;
};

export type ColumnDistribution = {
  column: string;
  kind: "numeric" | "categorical";
  bins: DistributionBin[];
  missing: number;
};

export type DuplicateDetail = {
  count: number;
  rows: Record<string, string | number | boolean | null>[];
};

export type DatasetAnalysis = DatasetProfile & {
  statistics: ColumnStat[];
  missing_values: MissingStat[];
  outliers: OutlierStat[];
  duplicates: { duplicate_count: number; duplicate_percentage: number };
  duplicates_detail: DuplicateDetail;
  distributions: ColumnDistribution[];
  correlations: { columns: string[]; values: (number | null)[][] };
  fingerprint: DatasetFingerprint;
  quality: QualityItem[];
  limits: {
    columns_total: number;
    columns_analyzed: number;
    rows_total: number;
    truncated: boolean;
  };
};

export type ResearchPaper = {
  title: string;
  authors?: string[];
  abstract?: string;
  year?: number | null;
  doi?: string | null;
  venue?: string | null;
  url?: string | null;
  citation_count?: number;
  source?: string;
  keywords?: string[];
  score?: number;
  reasons?: string[];
};

export type CountedItem = {
  name: string;
  count: number;
  percentage?: number;
};

export type ResearchLandscape = {
  status: string;
  paper_count: number;
  publication_years: Record<string, number>;
  top_venues: CountedItem[];
  sources: CountedItem[];
  methods: CountedItem[];
  research_activity: { year: string; paper_count: number }[];
  summary?: Record<string, unknown>;
};

export type ResearchTrend = {
  status: string;
  publication_trend: { year: number; paper_count: number }[];
  growth: {
    first_year?: number;
    first_count?: number;
    last_year?: number;
    last_count?: number;
    percentage?: number;
  } | null;
  peak_year: number | null;
  peak_paper_count: number;
  recent_direction: string;
  year_range?: { start: number; end: number } | null;
  summary?: Record<string, unknown>;
};

export type ResearchGap = {
  type: string;
  title: string;
  description: string;
  evidence?: Record<string, unknown>;
};

export type ResearchGapResult = {
  status: string;
  gaps: ResearchGap[];
  summary: { gap_count: number; papers_analyzed: number };
  warning?: string;
};

export type ResearchResult = {
  status: string;
  query: string;
  papers: ResearchPaper[];
  result_count: number;
  candidate_count: number;
  sources: string[];
  provider_status: Record<string, string>;
  message: string;
  ranked: boolean;
  landscape: ResearchLandscape;
  trend: ResearchTrend;
  gaps: ResearchGapResult;
  cached: boolean;
};
