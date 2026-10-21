from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import html
import math
import re
import pandas as pd

try:
    from jinja2 import Environment, FileSystemLoader, select_autoescape
    HAS_JINJA2 = True
except ImportError:
    HAS_JINJA2 = False


class ReportGenerator:
    """Comprehensive Research & ML Report Generator.

    Combines:
    1. Dataset Profile & Structural Diagnostics (Pillars 1-4)
    2. Data Quality & Health Issues
    3. Empirical Machine Learning Benchmarks & Model Ranking
    4. Academic Literature Review & Research Gaps
    5. AI Synthesis & Academic Insights
    """

    def __init__(self, template_dir: Optional[Union[str, Path]] = None) -> None:
        if template_dir is None:
            self.template_dir = Path(__file__).resolve().parent / "templates"
        else:
            self.template_dir = Path(template_dir)

        self._template_file = "report.html"

        if HAS_JINJA2 and self.template_dir.exists():
            self.jinja_env = Environment(
                loader=FileSystemLoader(str(self.template_dir)),
                autoescape=select_autoescape(["html", "xml"]),
            )
        else:
            self.jinja_env = None

    def generate(
        self,
        dataset_name: str = "Dataset",
        analysis_data: Optional[Dict[str, Any]] = None,
        ml_data: Optional[Dict[str, Any]] = None,
        research_data: Optional[Dict[str, Any]] = None,
        ai_synthesis: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generate a complete, self-contained HTML report."""
        context = self._build_context(
            dataset_name=dataset_name,
            analysis_data=analysis_data or {},
            ml_data=ml_data or {},
            research_data=research_data or {},
            ai_synthesis=ai_synthesis or "",
            metadata=metadata or {},
        )

        if self.jinja_env:
            try:
                template = self.jinja_env.get_template(self._template_file)
                return template.render(**context)
            except Exception as exc:
                # If template rendering encounters an error, fall back to fallback builder
                print(f"Warning: Jinja2 template rendering error: {exc}. Using fallback builder.")

        return self._generate_fallback_html(context)

    def export_to_file(
        self,
        filepath: Union[str, Path],
        dataset_name: str = "Dataset",
        analysis_data: Optional[Dict[str, Any]] = None,
        ml_data: Optional[Dict[str, Any]] = None,
        research_data: Optional[Dict[str, Any]] = None,
        ai_synthesis: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Generate and save the report to an HTML file."""
        html_content = self.generate(
            dataset_name=dataset_name,
            analysis_data=analysis_data,
            ml_data=ml_data,
            research_data=research_data,
            ai_synthesis=ai_synthesis,
            metadata=metadata,
        )
        target = Path(filepath)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(html_content, encoding="utf-8")
        return target

    def _build_context(
        self,
        dataset_name: str,
        analysis_data: Dict[str, Any],
        ml_data: Dict[str, Any],
        research_data: Dict[str, Any],
        ai_synthesis: str,
        metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Normalize raw results from all engines into clean template context."""
        profile = analysis_data.get("profile") or {}
        missing_values = analysis_data.get("missing_values") or []
        duplicates = analysis_data.get("duplicates") or {}
        raw_quality = analysis_data.get("data_quality") or []

        total_rows = int(profile.get("total_rows") or profile.get("row_count") or profile.get("rows") or 0)
        total_cols = int(profile.get("total_columns") or profile.get("column_count") or profile.get("columns") or 0)
        memory_usage = profile.get("memory_usage_str") or profile.get("memory_usage") or "N/A"
        if isinstance(memory_usage, (int, float)):
            memory_usage = f"{memory_usage / (1024 * 1024):.2f} MB"

        # Missing values
        total_cells = total_rows * total_cols
        missing_cells = 0
        if isinstance(missing_values, list):
            for m in missing_values:
                missing_cells += int(m.get("missing_count", 0))
        elif isinstance(missing_values, dict):
            missing_cells = int(missing_values.get("total_missing", 0))
        if not missing_cells:
            missing_cells = int(profile.get("missing_values") or 0)

        missing_pct = round((missing_cells / total_cells) * 100, 2) if total_cells > 0 else 0.0

        # Duplicates
        duplicate_count = int(
            duplicates.get("duplicate_count", 0)
            if isinstance(duplicates, dict)
            else 0
        )
        if not duplicate_count:
            duplicate_count = int(profile.get("duplicate_rows") or 0)

        # Columns summary
        dtypes = profile.get("dtypes") or {}
        col_summary = []
        numeric_count = 0
        cat_count = 0

        # Parse missing lookup
        missing_map = {}
        if isinstance(missing_values, list):
            for m in missing_values:
                col_name = m.get("column")
                if col_name:
                    missing_map[col_name] = (m.get("missing_count", 0), m.get("missing_percentage", 0.0))

        for col_name, dtype_str in dtypes.items():
            dtype_lower = str(dtype_str).lower()
            if any(t in dtype_lower for t in ["int", "float", "double", "num"]):
                numeric_count += 1
            else:
                cat_count += 1

            null_cnt, null_p = missing_map.get(col_name, (0, 0.0))
            col_summary.append({
                "name": col_name,
                "dtype": str(dtype_str),
                "null_count": f"{null_cnt:,}",
                "null_pct": f"{null_p:.1f}",
                "unique_count": "-",
                "sample_val": "-",
            })
        if not dtypes:
            numeric_count = int(profile.get("numeric_columns") or 0)
            cat_count = int(profile.get("categorical_columns") or 0)

        # Quality items
        quality_diagnoses = []
        critical_count = 0
        for item in raw_quality:
            if hasattr(item, "category"):
                q_dict = {
                    "category": item.category,
                    "title": item.title,
                    "severity": item.severity,
                    "problem": item.problem,
                    "magnitude": item.magnitude,
                    "impact": item.impact,
                    "options": item.options or [],
                }
            elif isinstance(item, dict):
                q_dict = {
                    "category": item.get("category", "info"),
                    "title": item.get("title", "Diagnostik"),
                    "severity": item.get("severity", "info"),
                    "problem": item.get("problem", ""),
                    "magnitude": item.get("magnitude", ""),
                    "impact": item.get("impact", ""),
                    "options": item.get("options", []),
                }
            else:
                continue

            if q_dict["severity"] == "critical":
                critical_count += 1
            quality_diagnoses.append(q_dict)

        # Readiness Score (0-100)
        score = 100.0
        score -= min(missing_pct * 1.5, 30.0)
        dup_pct = (duplicate_count / total_rows * 100) if total_rows > 0 else 0
        score -= min(dup_pct * 1.2, 20.0)
        score -= critical_count * 15.0
        readiness_score = max(int(round(score)), 10)

        # ML Context
        task_detection = ml_data.get("task_detection", {})
        primary_task = ml_data.get("primary_task", {}) or task_detection.get("primary_task", {})
        task_name = primary_task.get("task") or "Belum Ditentukan"
        target_col = primary_task.get("target")

        task_display_map = {
            "binary_classification": "Klasifikasi Biner (Supervised)",
            "multiclass_classification": "Klasifikasi Multikelas (Supervised)",
            "regression": "Regresi Numerik (Supervised)",
            "clustering": "Clustering (Unsupervised)",
            "anomaly_detection": "Deteksi Anomali (Unsupervised)",
        }
        primary_task_display = task_display_map.get(task_name, task_name.replace("_", " ").title())

        ml_readiness = analysis_data.get("ml_readiness") or {}
        if not ml_readiness and isinstance(metadata.get("dataframe"), pd.DataFrame):
            from app.analyzer.ml_readiness import analyze_ml_readiness

            recommendation = ml_data.get("recommendation") or {}
            ml_readiness = analyze_ml_readiness(
                metadata["dataframe"],
                missing_values=missing_values,
                duplicates=duplicates,
                outliers=analysis_data.get("outliers") or [],
                quality_issues=raw_quality,
                target=target_col,
                task=task_name if task_name != "unknown_task" else None,
                recommended_models=recommendation.get("recommendations") or [],
                ml_result=ml_data,
            )

        readiness_quality = ml_readiness.get("quality_summary") or {}
        report_quality_rows = [
            {"check": "Missing Values", "value": readiness_quality.get("missing_cells", 0)},
            {"check": "Duplicate Rows", "value": readiness_quality.get("duplicate_rows", 0)},
            {"check": "Outlier Values", "value": readiness_quality.get("outlier_values", 0)},
            {"check": "Constant Columns", "value": ", ".join(readiness_quality.get("constant_columns", [])) or "None"},
            {"check": "ID-like Columns", "value": ", ".join(readiness_quality.get("id_like_columns", [])) or "None"},
            {"check": "Mixed-type Columns", "value": ", ".join(readiness_quality.get("mixed_type_columns", [])) or "None"},
        ] if ml_readiness else []

        readiness_balance = ml_readiness.get("class_balance") or {}
        readiness_preprocessing = ml_readiness.get("preprocessing") or {}
        report_preprocessing_rows = []
        for item in readiness_preprocessing.get("encoding", []):
            report_preprocessing_rows.append({
                "area": "Encoding", "scope": item.get("column"), "recommendation": item.get("method")
            })
        for item in readiness_preprocessing.get("scaling", []):
            report_preprocessing_rows.append({
                "area": "Scaling", "scope": ", ".join(item.get("columns", [])), "recommendation": item.get("method")
            })
        if readiness_preprocessing:
            report_preprocessing_rows.append({
                "area": "Imbalance Handling", "scope": "Target", "recommendation": readiness_preprocessing.get("imbalance", "None")
            })
            for item in readiness_preprocessing.get("missing_strategy", []):
                report_preprocessing_rows.append({
                    "area": "Missing Value Strategy", "scope": item.get("column"), "recommendation": item.get("method")
                })

        readiness_features = ml_readiness.get("feature_selection") or {}
        report_feature_rows = [
            {"hint": "Constant", "feature": column, "result": "Review"}
            for column in readiness_features.get("constant_columns", [])
        ]
        report_feature_rows += [
            {"hint": "Near-constant", "feature": column, "result": "Review low variance"}
            for column in readiness_features.get("near_constant_columns", [])
        ]
        report_feature_rows += [
            {
                "hint": "High Correlation",
                "feature": f"{item.get('left')} / {item.get('right')}",
                "result": f"|r| = {item.get('correlation', 0):.3f}",
            }
            for item in readiness_features.get("correlated_pairs", [])
        ]
        report_feature_rows += [
            {
                "hint": "Quick Feature Importance",
                "feature": item.get("feature"),
                "result": f"{item.get('importance', 0):.4f}",
            }
            for item in readiness_features.get("importance", [])
        ]

        evaluation = ml_data.get("evaluation", {})
        eval_results = evaluation.get("results") or []
        ml_evaluated = bool(eval_results)

        candidate_models = []
        winner_model = None
        best_model_status_label = None

        if ml_evaluated:
            # Winner model
            best_info = ml_data.get("best_method") or (eval_results[0] if eval_results else {})
            winner_method_name = best_info.get("method") or best_info.get("method_id") or "Model Unggulan"
            winner_metrics = best_info.get("metrics") or {}
            
            # Extract primary metric
            pri_metric_name = "Score"
            pri_metric_val = 0.0
            if "accuracy" in winner_metrics:
                pri_metric_name = "Accuracy"
                pri_metric_val = winner_metrics["accuracy"]
            elif "r2" in winner_metrics:
                pri_metric_name = "R²"
                pri_metric_val = winner_metrics["r2"]
            elif "f1_macro" in winner_metrics:
                pri_metric_name = "F1 Macro"
                pri_metric_val = winner_metrics["f1_macro"]
            elif "silhouette" in winner_metrics:
                pri_metric_name = "Silhouette"
                pri_metric_val = winner_metrics["silhouette"]
            elif winner_metrics:
                k, v = next(iter(winner_metrics.items()))
                pri_metric_name = str(k).title()
                pri_metric_val = v

            sec_display = ""
            if "f1_macro" in winner_metrics and pri_metric_name != "F1 Macro":
                sec_display = f"F1-Macro: {winner_metrics['f1_macro']:.4f}"
            elif "rmse" in winner_metrics:
                sec_display = f"RMSE: {winner_metrics['rmse']:.4f}"

            winner_model = {
                "method": winner_method_name,
                "primary_metric_display": f"{pri_metric_name}: {pri_metric_val:.4f}" if isinstance(pri_metric_val, (int, float)) else str(pri_metric_val),
                "secondary_metric_display": sec_display,
                "folds": best_info.get("folds", 5),
                "rationale": "Model dengan performa rata-rata tertinggi dan variansi terkecil pada validasi k-fold cross validation.",
            }
            # The benchmark winner is the best performer supported by the
            # measured results, so the report labels it as such.
            best_model_status_label = "Best Empirical Performer"

            # Map recommendations knowledge base for strengths and limitations
            knowledge_map = {}
            for rec in ml_data.get("recommendation", {}).get("recommendations", []):
                m_id = rec.get("method_id")
                if m_id:
                    knowledge_map[m_id] = rec

            for res in eval_results:
                m_id = res.get("method_id") or res.get("method")
                m_name = res.get("method") or m_id
                metrics = res.get("metrics") or {}
                
                # Format primary metric
                val_str = "-"
                for m_key in ["accuracy", "r2", "f1_macro", "silhouette"]:
                    if m_key in metrics and isinstance(metrics[m_key], (int, float)):
                        val_str = f"{m_key.title()}: {metrics[m_key]:.4f}"
                        break
                if val_str == "-" and metrics:
                    first_k, first_v = next(iter(metrics.items()))
                    val_str = f"{first_k}: {first_v:.4f}" if isinstance(first_v, (int, float)) else str(first_v)

                std_val = metrics.get("cv_std") or metrics.get("accuracy_std") or metrics.get("r2_std")
                std_str = f"± {std_val:.4f}" if isinstance(std_val, (int, float)) else "-"

                rec_k = knowledge_map.get(m_id, {})
                strengths = rec_k.get("strengths") or ["Efisien", "Robust"]
                limitations = rec_k.get("limitations") or ["Sensitif terhadap hiperparameter"]

                candidate_models.append({
                    "method": m_name,
                    "primary_metric_str": val_str,
                    "std_str": std_str,
                    "strengths": ", ".join(strengths[:2]) if isinstance(strengths, list) else str(strengths),
                    "limitations": ", ".join(limitations[:2]) if isinstance(limitations, list) else str(limitations),
                })

        # Research Context
        # A search that returned papers counts as executed even when the
        # caller does not report an explicit success status.
        research_executed = bool(
            research_data
            and (
                research_data.get("papers")
                or research_data.get("status") in {"SUCCESS", "PARTIAL"}
            )
        )
        domain_info = research_data.get("domain") or {}
        domain_name = domain_info.get("domain") or domain_info.get("primary_domain") or "Multidisciplinary / General"

        keywords_info = research_data.get("keywords") or {}
        keywords_list = keywords_info.get("keywords") or keywords_info.get("primary_keywords") or []
        if isinstance(keywords_list, str):
            keywords_list = [k.strip() for k in keywords_list.split(",") if k.strip()]

        raw_papers = research_data.get("papers") or []
        papers_list = []
        for p in raw_papers[:15]:
            authors = p.get("authors") or []
            if isinstance(authors, list):
                authors_str = ", ".join(authors[:3]) + (" et al." if len(authors) > 3 else "")
            else:
                authors_str = str(authors)

            rel_score = p.get("relevance_score")
            rel_display = f"{rel_score * 100:.1f}% Relevan" if isinstance(rel_score, (int, float)) and rel_score <= 1.0 else (f"{rel_score:.1f}% Relevan" if isinstance(rel_score, (int, float)) else "Relevan")

            c_count = p.get("citation_count")
            c_display = f"{c_count:,} Sitasi" if isinstance(c_count, int) else "Sitasi: -"

            papers_list.append({
                "title": p.get("title") or "Untitled Paper",
                "authors_display": authors_str or "Penulis Tidak Diketahui",
                "year": p.get("year"),
                "citation_display": c_display,
                "relevance_display": rel_display,
                "abstract": p.get("abstract") or p.get("snippet") or "",
                "url": p.get("url") or p.get("pdf_url") or "",
            })

        # Research Gaps
        gaps_info = research_data.get("gaps") or {}
        research_gaps = []
        if isinstance(gaps_info, dict):
            gap_items = gaps_info.get("gaps") or gaps_info.get("potential_gaps") or gaps_info.get("items") or []
            for item in gap_items:
                if isinstance(item, dict):
                    title = item.get("title") or item.get("gap") or item.get("type") or "Kandidat celah penelitian"
                    description = item.get("description") or item.get("evidence") or ""
                    research_gaps.append(f"{title}: {description}" if description else str(title))
                else:
                    research_gaps.append(str(item))
        elif isinstance(gaps_info, list):
            research_gaps = [
                f"{item.get('title') or item.get('gap') or 'Kandidat celah penelitian'}: {item.get('description') or item.get('evidence') or ''}".rstrip(": ")
                if isinstance(item, dict) else str(item)
                for item in gaps_info
            ]

        gap_explanation = research_data.get("gap_explanation") or {}
        if isinstance(gap_explanation, dict):
            gap_explanation = gap_explanation.get("text") or ""
        elif not isinstance(gap_explanation, str):
            gap_explanation = ""
        gap_explanation_lines = []
        for line in str(gap_explanation).splitlines():
            cleaned = re.sub(r"^\s*(?:#{1,6}\s*|[-*+]\s+|\d+[.)]\s+)", "", line)
            cleaned = cleaned.replace("**", "").replace("__", "").replace("`", "").strip()
            if cleaned:
                gap_explanation_lines.append(cleaned)

        landscape = research_data.get("landscape") or {}
        trend = research_data.get("trend") or {}
        method_distribution = landscape.get("methods") or [] if isinstance(landscape, dict) else []
        if isinstance(method_distribution, dict):
            method_distribution = [{"name": name, "percentage": value} for name, value in method_distribution.items()]
        publication_trend = trend.get("publication_trend") or [] if isinstance(trend, dict) else []

        best_model_name = winner_model["method"] if winner_model else "Belum Dievaluasi"
        best_model_score = winner_model["primary_metric_display"] if winner_model else "Jalankan ML Intelligence"

        chart_paths = self._build_charts(
            dataframe=metadata.get("dataframe"),
            analysis_data=analysis_data,
            ml_data=ml_data,
            research_data=research_data,
            chart_dir=metadata.get("chart_dir"),
        )

        return {
            "dataset_name": dataset_name,
            "generated_at": datetime.now().strftime("%d %B %Y, %H:%M:%S"),
            "primary_task_display": primary_task_display,
            "target_column": target_col,
            "readiness_score": readiness_score,
            "total_rows_fmt": f"{total_rows:,}",
            "total_columns_fmt": f"{total_cols:,}",
            "missing_percentage": missing_pct,
            "missing_cells_fmt": f"{missing_cells:,}",
            "best_model_name": best_model_name,
            "best_model_score": best_model_score,
            "best_model_status_label": best_model_status_label,
            "papers_count": len(raw_papers),
            "memory_usage": memory_usage,
            "numeric_col_count": numeric_count,
            "categorical_col_count": cat_count,
            "duplicate_count_fmt": f"{duplicate_count:,}",
            "column_summary": col_summary,
            "quality_diagnoses": quality_diagnoses,
            "quality_issues_count": len(quality_diagnoses),
            "critical_issues_count": critical_count,
            "validation_protocol": "5-Fold Cross Validation" if ml_evaluated else "Belum Dievaluasi",
            "ml_evaluated": ml_evaluated,
            "winner_model": winner_model,
            "candidate_models": candidate_models,
            "research_executed": research_executed,
            "research_domain": domain_name,
            "keywords_list": keywords_list,
            "research_gaps": research_gaps,
            "gap_explanation": gap_explanation,
            "gap_explanation_lines": gap_explanation_lines,
            "method_distribution": method_distribution[:10],
            "publication_trend": publication_trend[-15:],
            "chart_paths": chart_paths,
            "papers_list": papers_list,
            "ai_synthesis": ai_synthesis,
            "ml_readiness_available": bool(ml_readiness),
            "readiness_summary_note": ml_readiness.get("summary_note", ""),
            "readiness_quality_summary": readiness_quality.get("summary", "Not applicable for this dataset"),
            "readiness_quality_rows": report_quality_rows,
            "readiness_class_balance": readiness_balance,
            "readiness_preprocessing_summary": readiness_preprocessing.get("summary", "Not applicable for this dataset"),
            "readiness_preprocessing_rows": report_preprocessing_rows,
            "readiness_fit_note": readiness_preprocessing.get("fit_note", "Fit preprocessing on the training set only to avoid data leakage."),
            "readiness_feature_summary": readiness_features.get("summary", "Not applicable for this dataset"),
            "readiness_feature_rows": report_feature_rows,
            "readiness_feature_importance_note": readiness_features.get("importance_note", ""),
            "readiness_pipeline_checklist": ml_readiness.get("pipeline_checklist", []),
        }

    @staticmethod
    def _build_charts(dataframe, analysis_data, ml_data, research_data, chart_dir) -> Dict[str, str]:
        """Create local, aggregate-only chart images for the PDF report."""
        if dataframe is None or not chart_dir:
            return {}
        try:
            import numpy as np
            from matplotlib.backends.backend_agg import FigureCanvasAgg
            from matplotlib.figure import Figure
        except ImportError:
            return {}

        output_dir = Path(chart_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        charts: Dict[str, str] = {}

        def save_figure(name: str, figure) -> None:
            target = output_dir / name
            FigureCanvasAgg(figure)
            figure.savefig(target, dpi=170, bbox_inches="tight", facecolor="white")
            charts[name.removesuffix(".png")] = name
            figure.clear()

        try:
            numeric = dataframe.select_dtypes(include=["number"])
            usable_columns = []
            for name in numeric.columns:
                normalized_name = re.sub(r"(?<=[a-z])(?=[A-Z])", "_", str(name)).lower()
                tokens = normalized_name.replace("-", "_").split("_")
                if not any(token in {"id", "identifier"} for token in tokens):
                    usable_columns.append(name)
            visual_numeric = numeric.loc[:, usable_columns]
            numeric_names = [str(name) for name in visual_numeric.columns]

            # Histogram panels describe real numeric-column distributions.
            histogram_columns = [name for name in visual_numeric.columns if visual_numeric[name].nunique(dropna=True) > 1][:4]
            if histogram_columns:
                figure = Figure(figsize=(8.0, 5.2), constrained_layout=True)
                axes = figure.subplots(2, 2).ravel()
                for index, axis in enumerate(axes):
                    if index >= len(histogram_columns):
                        axis.set_visible(False)
                        continue
                    column = histogram_columns[index]
                    values = visual_numeric[column].replace([np.inf, -np.inf], np.nan).dropna()
                    if len(values) > 100_000:
                        values = values.sample(100_000, random_state=42)
                    axis.hist(values, bins="fd", color="#3975c6", edgecolor="white")
                    axis.set_title(str(column), fontsize=9)
                    axis.set_ylabel("Frekuensi")
                    axis.grid(axis="y", alpha=0.2)
                figure.suptitle("Distribusi variabel numerik", fontsize=12)
                save_figure("histograms.png", figure)

            # Correlation heatmap, capped to keep a readable printed figure.
            if len(numeric_names) >= 2:
                stored_correlations = analysis_data.get("correlations")
                selected_columns = visual_numeric.columns[:12]
                if (
                    hasattr(stored_correlations, "corr")
                    and hasattr(stored_correlations, "columns")
                    and all(column in stored_correlations.columns for column in selected_columns)
                ):
                    matrix = stored_correlations.loc[selected_columns, selected_columns]
                else:
                    selected = visual_numeric.loc[:, selected_columns].replace([np.inf, -np.inf], np.nan)
                    if len(selected) > 50_000:
                        selected = selected.sample(50_000, random_state=42)
                    matrix = selected.corr()
                if matrix.shape[0] >= 2:
                    size = max(6.0, min(9.0, 0.55 * len(matrix) + 3.0))
                    figure = Figure(figsize=(size, size * 0.82), constrained_layout=True)
                    axis = figure.subplots()
                    image = axis.imshow(matrix.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
                    labels = [str(name) for name in matrix.columns]
                    axis.set_xticks(range(len(labels)), labels=labels, rotation=45, ha="right", fontsize=8)
                    axis.set_yticks(range(len(labels)), labels=labels, fontsize=8)
                    for row in range(len(labels)):
                        for col in range(len(labels)):
                            value = matrix.iloc[row, col]
                            if math.isfinite(float(value)):
                                axis.text(col, row, f"{value:.2f}", ha="center", va="center", fontsize=7)
                    figure.colorbar(image, ax=axis, shrink=0.78, label="Korelasi Pearson")
                    axis.set_title("Korelasi antarvariabel numerik", fontsize=11)
                    save_figure("correlation.png", figure)

            # Missing-value bars from the analyzed dataset itself.
            missing_rows = analysis_data.get("missing_values") or []
            if isinstance(missing_rows, list) and missing_rows:
                missing = {
                    str(item.get("column")): int(item.get("missing_count", 0))
                    for item in missing_rows if isinstance(item, dict)
                }
                missing = {name: count for name, count in missing.items() if count > 0}
                missing = dict(sorted(missing.items(), key=lambda item: item[1], reverse=True)[:12])
            else:
                missing = dataframe.isna().sum().sort_values(ascending=False)
                missing = missing[missing > 0].head(12)
                missing = {str(name): int(count) for name, count in missing.items()}
            if missing:
                missing_names = list(missing)
                missing_counts = np.asarray(list(missing.values()), dtype=float)
                percentages = missing_counts / max(1, len(dataframe)) * 100
                figure = Figure(figsize=(8.0, 4.2), constrained_layout=True)
                axis = figure.subplots()
                positions = np.arange(len(missing_names))
                axis.barh(positions, percentages, color="#d88945")
                axis.set_yticks(positions, labels=missing_names, fontsize=8)
                axis.invert_yaxis()
                axis.set_xlabel("Persentase nilai kosong (%)")
                axis.set_title("Nilai kosong per variabel", fontsize=11)
                axis.grid(axis="x", alpha=0.2)
                save_figure("missing_values.png", figure)

            # Compare models only when a common numeric metric is available.
            evaluation = ml_data.get("evaluation") or {}
            model_rows = evaluation.get("results") or []
            metric_order = ("accuracy", "f1_macro", "f1_score", "r2", "r2_score", "silhouette", "silhouette_score", "rmse", "mae")
            metric_name = next((
                key for key in metric_order
                if model_rows and all(isinstance(row.get("metrics", {}).get(key), (int, float)) for row in model_rows)
            ), None)
            if metric_name:
                labels = [str(row.get("method") or row.get("method_id") or "Model") for row in model_rows]
                scores = [float(row["metrics"][metric_name]) for row in model_rows]
                figure = Figure(figsize=(8.0, max(3.2, min(7.0, 0.42 * len(labels) + 1.5))), constrained_layout=True)
                axis = figure.subplots()
                positions = np.arange(len(labels))
                axis.barh(positions, scores, color="#4c87c9")
                axis.set_yticks(positions, labels=labels, fontsize=8)
                axis.invert_yaxis()
                axis.set_xlabel(metric_name.replace("_", " ").title())
                axis.set_title("Perbandingan hasil evaluasi model", fontsize=11)
                axis.grid(axis="x", alpha=0.2)
                save_figure("model_comparison.png", figure)

            # Research charts come only from papers that were actually found.
            trend = (research_data.get("trend") or {}).get("publication_trend") or []
            if trend:
                years = [str(item.get("year")) for item in trend if item.get("year") is not None]
                counts = [int(item.get("paper_count", 0)) for item in trend if item.get("year") is not None]
                if years:
                    figure = Figure(figsize=(8.0, 3.8), constrained_layout=True)
                    axis = figure.subplots()
                    axis.bar(years, counts, color="#56866a")
                    axis.set_ylabel("Jumlah paper")
                    axis.set_title("Jumlah publikasi per tahun", fontsize=11)
                    axis.tick_params(axis="x", rotation=35, labelsize=8)
                    axis.grid(axis="y", alpha=0.2)
                    save_figure("publication_trend.png", figure)

            landscape = research_data.get("landscape") or {}
            methods = landscape.get("methods") or []
            if isinstance(methods, list) and methods:
                methods = [item for item in methods if isinstance(item, dict) and item.get("name")]
                methods = methods[:10]
                if methods:
                    figure = Figure(figsize=(8.0, max(3.2, min(6.0, 0.38 * len(methods) + 1.4))), constrained_layout=True)
                    axis = figure.subplots()
                    names = [str(item["name"]) for item in methods]
                    values = [float(item.get("percentage", item.get("count", 0)) or 0) for item in methods]
                    positions = np.arange(len(names))
                    axis.barh(positions, values, color="#8b6eae")
                    axis.set_yticks(positions, labels=names, fontsize=8)
                    axis.invert_yaxis()
                    axis.set_xlabel("Persentase paper (%)" if any("percentage" in item for item in methods) else "Jumlah paper")
                    axis.set_title("Distribusi metode pada literatur", fontsize=11)
                    axis.grid(axis="x", alpha=0.2)
                    save_figure("research_methods.png", figure)
        except Exception as exc:
            print(f"Warning: report charts could not be generated: {exc}")

        return charts

    def _generate_fallback_html(self, ctx: Dict[str, Any]) -> str:
        """Lightweight HTML generator when Jinja2 is unavailable."""
        dataset_name = html.escape(str(ctx.get("dataset_name", "Dataset")))
        generated_at = html.escape(str(ctx.get("generated_at", "")))
        total_rows = ctx.get("total_rows_fmt", "0")
        total_cols = ctx.get("total_columns_fmt", "0")

        return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{dataset_name} - Research Report</title>
<style>
body {{ font-family: 'Times New Roman', Times, serif; background: #ffffff; color: #1e293b; padding: 30px; }}
.card {{ background: white; padding: 24px; border-radius: 12px; border: 1px solid #e2e8f0; margin-bottom: 20px; }}
h1 {{ color: #1e3a8a; }}
</style>
</head>
<body>
<div class="card">
<h1>{dataset_name}</h1>
<p>Generated: {generated_at}</p>
<p>Rows: {total_rows} | Columns: {total_cols}</p>
</div>
</body>
</html>"""
