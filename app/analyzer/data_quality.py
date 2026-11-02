from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd


@dataclass
class QualityDiagnosisItem:
    """
    Satu item diagnosis kualitas data berdasarkan 4 pilar Project Brief:
    1. APA MASALAHNYA?
    2. SEBERAPA BESAR MASALAHNYA?
    3. APA DAMPAKNYA?
    4. APA OPSI PENANGANANNYA?
    """
    category: str                   # 'missing_values' | 'duplicates' | 'invalid_values' | 'outliers' | 'identifiers' | 'constant_columns' | 'class_imbalance'
    title: str                      # Judul ringkas diagnosis
    severity: str                   # 'critical' | 'warning' | 'info'
    problem: str                    # APA MASALAHNYA?
    magnitude: str                  # SEBERAPA BESAR MASALAHNYA?
    impact: str                     # APA DAMPAKNYA?
    options: List[str]              # APA OPSI PENANGANANNYA?
    affected_columns: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)


class DataQualityDiagnoser:
    """
    Diagnosis kualitas data komprehensif berbasis evidence empiris.
    
    Mengikuti panduan Stage 3 Project Brief:
    Menghasilkan diagnosis 4-pilar terstruktur untuk setiap isu kualitas data.
    """

    def diagnose(
        self,
        dataframe: pd.DataFrame,
        profile: Optional[Dict[str, Any]] = None,
        missing_values: Optional[List[Dict[str, Any]]] = None,
        duplicates: Optional[Dict[str, Any]] = None,
        outliers: Optional[List[Dict[str, Any]]] = None,
        fingerprint: Optional[Dict[str, Any]] = None,
    ) -> List[QualityDiagnosisItem]:
        diagnoses: List[QualityDiagnosisItem] = []
        total_rows = len(dataframe)
        total_cols = len(dataframe.columns)

        if total_rows == 0 or total_cols == 0:
            return diagnoses

        # =========================================================
        # 1. DUPLICATE ROWS
        # =========================================================
        dup_count = int(
            duplicates.get("duplicate_count", 0)
            if duplicates
            else dataframe.duplicated().sum()
        )
        if dup_count > 0:
            dup_pct = (dup_count / total_rows) * 100
            sev = "critical" if dup_pct > 10 else ("warning" if dup_pct > 2 else "info")
            diagnoses.append(
                QualityDiagnosisItem(
                    category="duplicates",
                    title="Baris Duplikat Terdeteksi",
                    severity=sev,
                    problem=f"Ditemukan {dup_count:,} baris duplikat identik di dalam dataset.",
                    magnitude=f"{dup_count:,} baris ({dup_pct:.2f}% dari total {total_rows:,} baris).",
                    impact=(
                        "Menyebabkan risiko data leakage jika baris duplikat terpecah ke split train dan test; "
                        "menggelembungkan metrik evaluasi secara artifisial dan menyebabkan model overfitting pada observasi berulang."
                    ),
                    options=[
                        "Hapus baris duplikat (pertahankan instance pertama) melalui Dataset Versions & Research Trail.",
                        "Dataset original v0 tetap immutable dan tersimpan aman di sistem.",
                        "Periksa proses pengumpulan data jika duplikat bersumber dari penggabungan multi-tabel.",
                    ],
                    metrics={"duplicate_count": dup_count, "percentage": dup_pct},
                )
            )

        # =========================================================
        # 2. MISSING VALUES DIAGNOSIS
        # =========================================================
        missing_cols = []
        if missing_values:
            for item in missing_values:
                c_name = item.get("column", "")
                c_count = int(item.get("missing_count", 0))
                c_pct = float(item.get("missing_percentage", 0))
                if c_count > 0:
                    missing_cols.append((c_name, c_count, c_pct))
        else:
            for col in dataframe.columns:
                c_count = int(dataframe[col].isna().sum())
                if c_count > 0:
                    c_pct = (c_count / total_rows) * 100
                    missing_cols.append((str(col), c_count, c_pct))

        if missing_cols:
            # Sort by percentage descending
            missing_cols.sort(key=lambda x: x[2], reverse=True)
            high_missing = [c for c in missing_cols if c[2] >= 40]
            moderate_missing = [c for c in missing_cols if 5 <= c[2] < 40]
            low_missing = [c for c in missing_cols if c[2] < 5]

            affected_names = [c[0] for c in missing_cols]
            total_missing_cells = sum(c[1] for c in missing_cols)
            total_cells = total_rows * total_cols
            overall_pct = (total_missing_cells / total_cells) * 100

            sev = "critical" if high_missing else ("warning" if moderate_missing else "info")

            detail_parts = [f"{c[0]} ({c[1]:,} missing, {c[2]:.1f}%)" for c in missing_cols[:4]]
            if len(missing_cols) > 4:
                detail_parts.append(f"+{len(missing_cols) - 4} kolom lainnya")

            diagnoses.append(
                QualityDiagnosisItem(
                    category="missing_values",
                    title="Nilai Kosong (Missing Values) Terdeteksi",
                    severity=sev,
                    problem=f"{len(missing_cols)} kolom memiliki nilai kosong: {', '.join(detail_parts)}.",
                    magnitude=(
                        f"Total {total_missing_cells:,} sel kosong ({overall_pct:.2f}% dari seluruh data). "
                        f"Kolom dengan missing tertinggi: {missing_cols[0][0]} ({missing_cols[0][2]:.1f}%)."
                    ),
                    impact=(
                        "Algoritma machine learning (seperti Scikit-Learn standard) tidak dapat memproses nilai NaN langsung. "
                        "Menghapus baris secara listwise (drop na) dapat memangkas ukuran sampel penelitian secara signifikan dan memicu selection bias."
                    ),
                    options=[
                        "Kolom dengan missing < 5%: Imputasi statistik sederhana (Median untuk numerik skewed, Modus untuk kategorikal).",
                        "Kolom dengan missing 5%–40%: KNN Imputation atau Iterative Imputer untuk mempertahankan korelasi antar fitur.",
                        "Kolom dengan missing > 50%: Pertimbangkan dropping kolom atau buat binary indicator column (is_missing) agar tidak memasukkan noise.",
                    ],
                    affected_columns=affected_names,
                    metrics={"columns_count": len(missing_cols), "total_missing": total_missing_cells},
                )
            )

        # =========================================================
        # 3. NON-FINITE NUMERIC VALUES (distinct from missing values)
        # =========================================================
        non_finite_columns = []
        for column in dataframe.select_dtypes(include=["number"]).columns:
            try:
                values = dataframe[column].to_numpy(dtype=float, na_value=np.nan)
                count = int(np.isinf(values).sum())
            except (TypeError, ValueError, OverflowError):
                continue
            if count:
                non_finite_columns.append((str(column), count))

        if non_finite_columns:
            total_invalid = sum(count for _, count in non_finite_columns)
            percentage = total_invalid / max(1, total_rows * total_cols) * 100
            names = [name for name, _ in non_finite_columns]
            diagnoses.append(
                QualityDiagnosisItem(
                    category="invalid_values",
                    title="Nilai Numerik Tidak Terhingga",
                    severity="critical",
                    problem=(
                        f"Ditemukan {total_invalid:,} nilai +Inf/-Inf pada kolom "
                        + ", ".join(names)
                        + "."
                    ),
                    magnitude=f"{total_invalid:,} sel ({percentage:.2f}% dari seluruh sel).",
                    impact=(
                        "Nilai tak terhingga bukan nilai kosong biasa dan dapat membuat "
                        "statistik, scaling, atau training model gagal maupun menghasilkan metrik tidak valid."
                    ),
                    options=[
                        "Periksa sumber dan satuan nilai; tentukan apakah nilai tersebut kesalahan perhitungan.",
                        "Jika memang invalid, ubah menjadi missing lalu pilih strategi imputasi pada pipeline training.",
                    ],
                    affected_columns=names,
                    metrics={
                        "non_finite_count": total_invalid,
                        "count_by_column": dict(non_finite_columns),
                        "percentage_of_cells": percentage,
                    },
                )
            )

        # =========================================================
        # 4. OUTLIERS (IQR METHOD)
        # =========================================================
        outlier_cols = []
        if outliers:
            for item in outliers:
                c_name = item.get("column", "")
                c_count = int(item.get("outlier_count", 0))
                c_pct = float(item.get("outlier_percentage", 0))
                if c_count > 0:
                    outlier_cols.append((c_name, c_count, c_pct))
        else:
            num_cols = [
                column for column in dataframe.select_dtypes(include=["number"]).columns
                if not pd.api.types.is_bool_dtype(dataframe[column].dtype)
            ]
            for col in num_cols:
                series = dataframe[col].dropna()
                try:
                    series = series[np.isfinite(series.to_numpy(dtype=float))]
                except (TypeError, ValueError, OverflowError):
                    continue
                if len(series) > 0:
                    q1 = series.quantile(0.25)
                    q3 = series.quantile(0.75)
                    iqr = q3 - q1
                    low = q1 - 1.5 * iqr
                    high = q3 + 1.5 * iqr
                    cnt = int(((series < low) | (series > high)).sum())
                    if cnt > 0:
                        outlier_cols.append((str(col), cnt, (cnt / len(series)) * 100))

        if outlier_cols:
            outlier_cols.sort(key=lambda x: x[1], reverse=True)
            sev = "warning" if any(c[2] > 5 for c in outlier_cols) else "info"
            top_outliers = [f"{c[0]} ({c[1]:,} outlier, {c[2]:.1f}%)" for c in outlier_cols[:3]]

            diagnoses.append(
                QualityDiagnosisItem(
                    category="outliers",
                    title="Potensi Outlier Numerik (Metode IQR)",
                    severity=sev,
                    problem=f"Ditemukan observasi ekstrem di luar batas 1.5×IQR pada kolom: {', '.join(top_outliers)}.",
                    magnitude=f"{len(outlier_cols)} kolom numerik memiliki data ekstrem.",
                    impact=(
                        "Mendistorsi estimasi mean dan standard deviation; menarik garis regresi pada OLS Linear Regression; "
                        "memperbesar loss pada metrik berbasis kuadrat (MSE/RMSE). Model sensitif jarak (KNN, SVM) dapat terganggu."
                    ),
                    options=[
                        "Gunakan RobustScaler (berbasis median dan IQR) saat tahap Scaling, bukan StandardScaler.",
                        "Terapkan transformasi log atau winsorizing (clipping pada persentil 1% dan 99%).",
                        "Gunakan model berbasis Decision Tree / Random Forest yang secara inheren kebal terhadap nilai outlier monotonik.",
                    ],
                    affected_columns=[c[0] for c in outlier_cols],
                    metrics={"columns_count": len(outlier_cols)},
                )
            )

        # =========================================================
        # 4. IDENTIFIER / LEAKAGE RISK COLUMNS
        # =========================================================
        potential_ids = []
        column_sigs = []
        if fingerprint and "representation" in fingerprint:
            column_sigs = fingerprint["representation"].get("column_signature", [])

        if column_sigs:
            for sig in column_sigs:
                if sig.get("potential_id"):
                    potential_ids.append(sig.get("name"))
        else:
            for col in dataframe.columns:
                series = dataframe[col].dropna()
                u_cnt = series.nunique()
                c_lower = str(col).lower().replace(" ", "_")
                if ("id" in c_lower or "kode" in c_lower or "index" in c_lower) and u_cnt > 0.8 * total_rows:
                    potential_ids.append(str(col))

        if potential_ids:
            diagnoses.append(
                QualityDiagnosisItem(
                    category="identifiers",
                    title="Kolom Identifier / Index Berisiko Data Leakage",
                    severity="warning",
                    problem=f"Kolom {', '.join(potential_ids)} terdeteksi sebagai kolom identitas unik atau indeks.",
                    magnitude=f"{len(potential_ids)} kolom memiliki rasio unik tinggi mendekati 100% baris.",
                    impact=(
                        "Menyertakan ID dalam training menyebabkan model menghafal record (overfitting) tanpa belajar pola prediktif; "
                        "jika dikonversi ke One-Hot Encoding akan mengakibatkan ledakan dimensi (Curse of Dimensionality)."
                    ),
                    options=[
                        "Kecualikan kolom identifier dari fitur pelatihan machine learning.",
                        "Gunakan kolom ID hanya sebagai metadata referensi penelitian.",
                    ],
                    affected_columns=potential_ids,
                    metrics={"id_columns": potential_ids},
                )
            )

        # =========================================================
        # 5. CONSTANT / ZERO-VARIANCE COLUMNS
        # =========================================================
        constant_cols = []
        for col in dataframe.columns:
            if dataframe[col].nunique(dropna=False) <= 1:
                constant_cols.append(str(col))

        if constant_cols:
            diagnoses.append(
                QualityDiagnosisItem(
                    category="constant_columns",
                    title="Kolom Konstan (Zero-Variance)",
                    severity="critical",
                    problem=f"Kolom {', '.join(constant_cols)} hanya memiliki 1 nilai unik untuk semua baris.",
                    magnitude=f"{len(constant_cols)} kolom dengan variansi nol.",
                    impact="Kolom konstan membawa zero information gain dan dapat menyebabkan matriks singular pada pemodelan.",
                    options=["Drop kolom konstan sebelum splitting dan training."],
                    affected_columns=constant_cols,
                )
            )

        # =========================================================
        # 6. CLASS IMBALANCE IN TARGET CANDIDATE
        # =========================================================
        target_candidates = []
        if fingerprint and "representation" in fingerprint:
            target_candidates = fingerprint["representation"].get("target_candidates", [])

        if target_candidates:
            target_col = target_candidates[0]
            if target_col in dataframe.columns:
                target_series = dataframe[target_col].dropna()
                u_cnt = target_series.nunique()
                if 2 <= u_cnt <= 10:
                    counts = target_series.value_counts()
                    maj_count = counts.iloc[0]
                    min_count = counts.iloc[-1]
                    ratio = maj_count / max(1, min_count)
                    if ratio >= 2.5:
                        maj_pct = (maj_count / len(target_series)) * 100
                        min_pct = (min_count / len(target_series)) * 100
                        diagnoses.append(
                            QualityDiagnosisItem(
                                category="class_imbalance",
                                title=f"Ketidakseimbangan Kelas pada Target '{target_col}'",
                                severity="warning" if ratio < 6 else "critical",
                                problem=(
                                    f"Distribusi kelas pada kandidat target '{target_col}' tidak seimbang "
                                    f"(rasio {ratio:.1f}:1)."
                                ),
                                magnitude=(
                                    f"Kelas mayoritas: {maj_pct:.1f}% ({maj_count:,} baris) vs "
                                    f"Kelas minoritas: {min_pct:.1f}% ({min_count:,} baris)."
                                ),
                                impact=(
                                    "Accuracy Paradox: Model dapat meraih akurasi tinggi hanya dengan selalu memprediksi kelas mayoritas, "
                                    "namun gagal mendeteksi kelas minoritas (Recall rendah)."
                                ),
                                options=[
                                    "Gunakan Stratified Split saat data splitting agar proporsi kelas terjaga.",
                                    "Terapkan class_weight='balanced' pada model klasifikasi.",
                                    "Evaluasi model menggunakan F1-Score (Macro), Precision-Recall AUC, dan Confusion Matrix, bukan sekadar Accuracy.",
                                ],
                                affected_columns=[target_col],
                                metrics={"ratio": ratio, "target": target_col},
                            )
                        )

        return diagnoses
