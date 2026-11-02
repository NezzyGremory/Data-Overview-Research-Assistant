from __future__ import annotations

import re
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd


class MLTargetDetector:
    """
    Evidence-based ML target detector.

    Analyzes columns to detect potential target candidates for supervised learning.
    Combines multiple evidence signals:
    - Column name semantic cues (target hints vs identifier hints)
    - Data types and semantic types
    - Cardinality and unique value ratio
    - Missing value proportions
    - Target suitability for classification vs regression
    - Structural signals (e.g. column position)

    Important Principle:
    Target detection is an ESTIMATION / RECOMMENDATION, never absolute truth.
    Users can confirm or override detected targets.
    """

    STATUS = "ESTIMATION"

    POSITIVE_TARGET_HINTS = {
        # General target keywords
        "target", "label", "class", "outcome", "result", "response",
        "output", "prediction", "dependent", "category", "status",
        "grade", "score", "value", "state", "group", "segment",
        # Common domain target names
        "survived", "survival", "churn", "default", "fraud", "is_fraud",
        "price", "saleprice", "sale_price", "salary", "revenue", "income",
        "cost", "profit", "rating", "sentiment", "diagnosis", "disease",
        "risk", "attrition", "converted", "conversion", "click", "bought",
        "subscribed", "purchased", "approved", "acceptance", "quality",
    }

    IDENTIFIER_HINTS = {
        "id", "uuid", "guid", "identifier", "index", "record", "record_id",
        "row_id", "entry_id", "pk", "key", "number", "no", "code",
        "passengerid", "customerid", "user_id", "userid", "client_id",
        "order_id", "transaction_id", "session_id", "ticket",
    }

    def __init__(self) -> None:
        pass

    def detect(
        self,
        dataframe: Optional[pd.DataFrame] = None,
        fingerprint: Optional[Dict[str, Any]] = None,
        columns_meta: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Detect candidate target columns ordered by confidence score (descending).
        """
        candidates: List[Dict[str, Any]] = []

        if dataframe is not None and not dataframe.empty:
            candidates = self._detect_from_dataframe(dataframe)
        elif columns_meta:
            candidates = self._detect_from_columns_meta(columns_meta)
        elif fingerprint:
            candidates = self._detect_from_fingerprint(fingerprint)

        # Sort descending by confidence score
        candidates.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
        return candidates

    # ------------------------------------------------------------------
    # DATAFRAME EVIDENCE EXTRACTION
    # ------------------------------------------------------------------

    def _detect_from_dataframe(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        candidates: List[Dict[str, Any]] = []
        n_rows = len(df)
        n_cols = len(df.columns)

        if n_rows == 0 or n_cols == 0:
            return []

        for idx, col in enumerate(df.columns):
            series = df[col]
            col_name = str(col).strip()
            if not col_name:
                continue

            unique_count = int(series.nunique(dropna=True))
            missing_count = int(series.isna().sum())
            missing_ratio = float(missing_count / n_rows) if n_rows > 0 else 0.0
            unique_ratio = float(unique_count / n_rows) if n_rows > 0 else 0.0

            normalized_name = self._normalize_name(col_name)
            tokens = set(re.split(r"[_\s\-]+", normalized_name))

            score = 0.0
            reasons: List[str] = []
            suggested_task: Optional[str] = None

            # 1. Constant or all-missing column (not a target)
            if unique_count <= 1:
                score -= 80.0
                reasons.append("Kolom konstan atau tidak memiliki variasi nilai.")

            # 2. Identifier detection
            is_id = self._is_identifier_signal(normalized_name, tokens, unique_ratio, n_rows)
            if is_id:
                score -= 60.0
                reasons.append("Kolom tampak seperti identifier / ID rekaman.")

            # 3. High missing penalty
            if missing_ratio > 0.5:
                score -= 40.0
                reasons.append(f"Persentase nilai hilang tinggi ({missing_ratio * 100:.1f}%).")
            elif missing_ratio > 0.2:
                score -= 15.0
                reasons.append(f"Memiliki missing value ({missing_ratio * 100:.1f}%).")

            # 4. Semantic name clues
            if normalized_name in self.POSITIVE_TARGET_HINTS:
                score += 45.0
                reasons.append(f"Nama kolom '{col_name}' merupakan sinyal kuat target.")
            elif tokens & self.POSITIVE_TARGET_HINTS:
                score += 25.0
                matched_token = list(tokens & self.POSITIVE_TARGET_HINTS)[0]
                reasons.append(f"Nama kolom mengandung kata kunci target '{matched_token}'.")

            # 5. Type and distribution suitability
            dtype_str = str(series.dtype).lower()
            is_bool = (
                dtype_str == "bool"
                or pd.api.types.is_bool_dtype(series)
                or (unique_count == 2 and set(series.dropna().unique()).issubset({0, 1, "0", "1", True, False}))
            )
            is_numeric = bool(pd.api.types.is_numeric_dtype(series))
            is_categorical = bool(
                isinstance(series.dtype, pd.CategoricalDtype)
                or pd.api.types.is_object_dtype(series)
                or pd.api.types.is_string_dtype(series)
            )

            if is_bool or unique_count == 2:
                score += 35.0
                suggested_task = "binary_classification"
                reasons.append("Kolom memiliki 2 kelas (sangat ideal untuk binary classification).")
            elif is_categorical:
                if 3 <= unique_count <= 25:
                    score += 26.0
                    suggested_task = "multiclass_classification"
                    reasons.append(f"Kolom kategorikal dengan {unique_count} kelas diskrit.")
                elif unique_count > 25 and unique_ratio > 0.4:
                    # High cardinality text / IDs
                    score -= 30.0
                    reasons.append("Kardinalitas teks terlalu tinggi untuk label target klasifikasi.")
            elif is_numeric:
                if 3 <= unique_count <= 15:
                    score += 22.0
                    suggested_task = "multiclass_classification"
                    reasons.append(f"Kolom numerik diskrit dengan {unique_count} nilai unik.")
                elif unique_count > 15:
                    # Continuous numerical target (Regression)
                    # Check if reasonable distribution
                    score += 20.0
                    suggested_task = "regression"
                    reasons.append("Kolom numerik kontinu dengan variasi nilai yang baik untuk regresi.")

            # 6. Positional structural signal (target often at the end or very first column)
            if n_cols > 2 and idx >= n_cols - 2:
                score += 8.0
                reasons.append("Posisi kolom berada di akhir struktur dataset.")

            final_score = round(float(np.clip(score, 0.0, 100.0)), 2)

            if final_score >= 15.0:
                candidates.append({
                    "name": col_name,
                    "score": final_score,
                    "confidence": final_score,
                    "suggested_task": suggested_task,
                    "unique_count": unique_count,
                    "missing_ratio": round(missing_ratio, 4),
                    "dtype": dtype_str,
                    "reasons": reasons,
                })

        return candidates

    # ------------------------------------------------------------------
    # FINGERPRINT / METADATA EVIDENCE
    # ------------------------------------------------------------------

    def _detect_from_fingerprint(self, fingerprint: Dict[str, Any]) -> List[Dict[str, Any]]:
        representation = fingerprint.get("representation", fingerprint)
        if not isinstance(representation, dict):
            representation = fingerprint

        # Check existing target_candidates in fingerprint
        fp_candidates = representation.get("target_candidates")
        columns_meta = representation.get("column_signature") or representation.get("columns")

        if columns_meta and isinstance(columns_meta, list):
            inferred = self._detect_from_columns_meta(columns_meta)
            if fp_candidates and isinstance(fp_candidates, list):
                return self._merge_candidates(fp_candidates, inferred)
            return inferred

        if fp_candidates and isinstance(fp_candidates, list):
            return self._normalize_candidate_list(fp_candidates)

        return []

    def _detect_from_columns_meta(self, columns_meta: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        candidates: List[Dict[str, Any]] = []
        n_cols = len(columns_meta)

        for idx, col in enumerate(columns_meta):
            if not isinstance(col, dict):
                continue

            name = str(col.get("name") or col.get("normalized_name") or "").strip()
            if not name:
                continue

            normalized_name = self._normalize_name(name)
            tokens = set(re.split(r"[_\s\-]+", normalized_name))

            unique_count = self._safe_int(col.get("unique_count", col.get("cardinality", 0)))
            semantic_type = str(col.get("semantic_type", "")).lower()
            dtype = str(col.get("dtype", "")).lower()
            potential_id = bool(col.get("potential_id", False))

            score = 0.0
            reasons: List[str] = []
            suggested_task: Optional[str] = None

            if unique_count <= 1 and "unique_count" in col:
                score -= 80.0

            if potential_id or self._is_identifier_name(normalized_name, tokens):
                score -= 60.0
                reasons.append("Kolom tampak seperti identifier.")

            if normalized_name in self.POSITIVE_TARGET_HINTS:
                score += 45.0
                reasons.append("Nama kolom merupakan sinyal umum target.")
            elif tokens & self.POSITIVE_TARGET_HINTS:
                score += 25.0
                reasons.append("Nama kolom mengandung istilah target.")

            if semantic_type == "boolean" or dtype == "bool" or unique_count == 2:
                score += 35.0
                suggested_task = "binary_classification"
                reasons.append("Kolom boolean/biner cocok untuk klasifikasi biner.")
            elif semantic_type == "categorical":
                if 2 <= unique_count <= 25:
                    score += 24.0
                    suggested_task = "multiclass_classification"
                    reasons.append(f"Kolom kategorikal memiliki {unique_count} kelas.")
            elif semantic_type == "numeric" or "int" in dtype or "float" in dtype:
                if unique_count == 2:
                    score += 30.0
                    suggested_task = "binary_classification"
                    reasons.append("Kolom numerik biner cocok untuk target klasifikasi.")
                elif 3 <= unique_count <= 15:
                    score += 20.0
                    suggested_task = "multiclass_classification"
                    reasons.append(f"Kolom numerik memiliki {unique_count} nilai diskrit.")
                elif unique_count > 15:
                    score += 18.0
                    suggested_task = "regression"
                    reasons.append("Kolom numerik kontinu berpotensi menjadi target regresi.")

            if n_cols > 2 and idx >= n_cols - 2:
                score += 8.0
                reasons.append("Kolom berada dekat akhir dataset.")

            final_score = round(float(np.clip(score, 0.0, 100.0)), 2)

            if final_score >= 15.0:
                candidates.append({
                    "name": name,
                    "score": final_score,
                    "confidence": final_score,
                    "suggested_task": suggested_task,
                    "unique_count": unique_count,
                    "dtype": dtype,
                    "reasons": reasons,
                })

        return candidates

    # ------------------------------------------------------------------
    # HELPERS
    # ------------------------------------------------------------------

    def _is_identifier_signal(
        self,
        normalized_name: str,
        tokens: set[str],
        unique_ratio: float,
        n_rows: int,
    ) -> bool:
        if self._is_identifier_name(normalized_name, tokens):
            return True
        # If unique ratio is near 100% on a large dataset and integer/string, likely an ID
        if unique_ratio > 0.98 and n_rows > 50:
            return True
        return False

    def _is_identifier_name(self, normalized_name: str, tokens: set[str]) -> bool:
        if normalized_name in self.IDENTIFIER_HINTS:
            return True
        if normalized_name.endswith("_id") or normalized_name.startswith("id_"):
            return True
        if tokens & self.IDENTIFIER_HINTS:
            return True
        return False

    def _merge_candidates(
        self,
        primary: List[Dict[str, Any]],
        secondary: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        merged: Dict[str, Dict[str, Any]] = {}

        for item in list(primary) + list(secondary):
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or item.get("column") or "").strip()
            if not name:
                continue
            key = name.lower()
            prev = merged.get(key)
            score = float(item.get("score") or item.get("confidence") or 0.0)
            if prev is None or score > float(prev.get("score", 0.0)):
                merged[key] = dict(item)
                merged[key]["name"] = name
                merged[key]["score"] = score
                merged[key]["confidence"] = score

        return sorted(merged.values(), key=lambda x: float(x.get("score", 0.0)), reverse=True)

    def _normalize_candidate_list(self, candidates: List[Any]) -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []
        for item in candidates:
            if isinstance(item, dict):
                name = str(item.get("name") or item.get("column") or "").strip()
                if name:
                    score = float(item.get("score", item.get("confidence", 50.0)))
                    normalized.append({
                        "name": name,
                        "score": score,
                        "confidence": score,
                        "suggested_task": item.get("suggested_task"),
                        "reasons": item.get("reasons", ["Berdasarkan metadata target."]),
                    })
            elif isinstance(item, str) and item.strip():
                normalized.append({
                    "name": item.strip(),
                    "score": 50.0,
                    "confidence": 50.0,
                    "reasons": ["Nama target yang ditentukan."],
                })
        return normalized

    @staticmethod
    def _normalize_name(value: Any) -> str:
        text = str(value or "").strip().lower()
        return re.sub(r"[\s\-]+", "_", text)

    @staticmethod
    def _safe_int(value: Any) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0
