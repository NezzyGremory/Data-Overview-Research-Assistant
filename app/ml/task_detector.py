from __future__ import annotations

from typing import Any, Optional

from app.ml.target_detector import MLTargetDetector


class MLTaskDetector:
    """
    Generic, dataset-agnostic ML task detector.

    The detector combines two evidence sources:
    1. Dataset fingerprint metadata.
    2. The raw pandas DataFrame, when available.

    It never branches on a dataset name. Target discovery is based on column
    semantics, cardinality, dtype, identifier likelihood, and weak positional
    signals. Results are estimates, not guarantees.
    """

    STATUS = "ESTIMATION"

    TASK_LABELS = {
        "binary_classification": "Binary Classification",
        "multiclass_classification": "Multiclass Classification",
        "regression": "Regression",
        "clustering": "Clustering",
        "anomaly_detection": "Anomaly Detection",
        "time_series_forecasting": "Time-Series Forecasting",
    }

    GENERIC_TARGET_HINTS = {
        "target", "label", "class", "outcome", "result", "response",
        "output", "prediction", "dependent", "category", "status",
        "grade", "score", "value", "state", "group", "segment",
    }

    GENERIC_ID_HINTS = {
        "id", "uuid", "identifier", "index", "record", "number", "no",
    }

    def __init__(self, target_detector: Optional[MLTargetDetector] = None) -> None:
        self.target_detector = target_detector or MLTargetDetector()

    def detect(
        self,
        fingerprint: dict[str, Any] | None,
        dataframe=None,
    ) -> dict[str, Any]:
        fingerprint = fingerprint if isinstance(fingerprint, dict) else {}

        representation = fingerprint.get("representation")
        if not isinstance(representation, dict):
            representation = fingerprint

        # The loaded frame is authoritative. A persisted fingerprint may have
        # been generated before dtype normalization or from another version.
        columns = self._columns_from_dataframe(dataframe) if dataframe is not None else []
        if not columns:
            columns = self._normalize_columns(representation.get("column_signature"))
        if not columns:
            columns = self._normalize_columns(representation.get("columns"))

        characteristics = representation.get("characteristics")
        if not isinstance(characteristics, dict):
            characteristics = {}

        target_candidates = self._normalize_target_candidates(
            representation.get("target_candidates")
        )

        # Delegate target inference to MLTargetDetector
        inferred = self.target_detector.detect(
            dataframe=dataframe,
            columns_meta=columns,
            fingerprint=fingerprint,
        )
        if inferred:
            if target_candidates:
                target_candidates = self._merge_target_candidates(
                    target_candidates, inferred
                )
            else:
                target_candidates = inferred

        tasks: list[dict[str, Any]] = []

        supervised = self._detect_supervised_task(columns, target_candidates)
        if supervised:
            tasks.append(supervised)

        numeric_count = self._get_numeric_count(columns, characteristics, dataframe)
        feature_count = self._feature_count(columns, characteristics, dataframe)
        datetime_count = self._get_datetime_count(columns, characteristics, dataframe)

        if numeric_count >= 2:
            tasks.append(
                self._make_task(
                    "clustering",
                    self._clustering_score(numeric_count, feature_count),
                    None,
                    [
                        f"Dataset memiliki {numeric_count} fitur numerik.",
                        "Clustering dapat dilakukan tanpa target.",
                        "Struktur fitur memungkinkan pencarian kelompok alami.",
                    ],
                )
            )
            tasks.append(
                self._make_task(
                    "anomaly_detection",
                    self._anomaly_score(numeric_count, feature_count),
                    None,
                    [
                        f"Dataset memiliki {numeric_count} fitur numerik.",
                        "Deteksi anomali dapat dilakukan tanpa target.",
                    ],
                )
            )

        if datetime_count >= 1:
            tasks.append(
                self._make_task(
                    "time_series_forecasting",
                    self._time_series_score(feature_count, datetime_count),
                    None,
                    [
                        "Dataset memiliki kolom bertipe waktu.",
                        "Struktur dataset berpotensi digunakan untuk analisis berbasis waktu.",
                    ],
                )
            )

        tasks.sort(key=lambda item: self._safe_score(item), reverse=True)
        primary_task = self._select_primary_task(tasks)

        if primary_task and primary_task.get("task") in {"clustering", "anomaly_detection"}:
            message = (
                "Tidak ada target yang cukup pasti untuk tugas supervised. "
                "Pilih task dan kolom target sesuai tujuan analisis sebelum melatih model."
            )
        else:
            message = (
                "Jenis machine learning merupakan estimasi berdasarkan "
                "struktur dataset, tipe fitur, cardinality, dan kandidat target."
            )

        return {
            "status": self.STATUS,
            "primary_task": primary_task,
            "tasks": tasks,
            "task_count": len(tasks),
            "target_candidates": target_candidates[:10],
            "message": message,
        }

    # ------------------------------------------------------------------
    # DATAFRAME -> COLUMN METADATA
    # ------------------------------------------------------------------

    def _columns_from_dataframe(self, dataframe) -> list[dict[str, Any]]:
        columns: list[dict[str, Any]] = []
        try:
            for name in dataframe.columns:
                series = dataframe[name]
                dtype = str(series.dtype).lower()
                unique_count = int(series.nunique(dropna=True))

                if self._is_datetime_series(series):
                    semantic_type = "datetime"
                elif dtype == "bool" or str(series.dropna().map(type).drop_duplicates().tolist()[:1]) == "[<class 'bool'>]":
                    semantic_type = "boolean"
                elif self._pandas_is_numeric(series):
                    semantic_type = "numeric"
                elif self._looks_like_text_series(series):
                    semantic_type = "text"
                else:
                    semantic_type = "categorical"

                item: dict[str, Any] = {
                    "name": str(name),
                    "normalized_name": self._normalize_name(name),
                    "dtype": dtype,
                    "semantic_type": semantic_type,
                    "unique_count": unique_count,
                }

                if self._pandas_is_numeric(series) and unique_count:
                    try:
                        item["min"] = float(series.min())
                        item["max"] = float(series.max())
                    except Exception:
                        pass

                columns.append(item)
        except Exception:
            return []

        return columns

    @staticmethod
    def _pandas_is_numeric(series) -> bool:
        try:
            import pandas as pd
            return bool(pd.api.types.is_numeric_dtype(series))
        except Exception:
            return False

    @staticmethod
    def _is_datetime_series(series) -> bool:
        try:
            import pandas as pd
            if pd.api.types.is_datetime64_any_dtype(series):
                return True
            if pd.api.types.is_object_dtype(series):
                sample = series.dropna().head(50)
                if sample.empty:
                    return False
                parsed = pd.to_datetime(sample, errors="coerce")
                return float(parsed.notna().mean()) >= 0.8
        except Exception:
            pass
        return False

    @staticmethod
    def _looks_like_text_series(series) -> bool:
        try:
            import pandas as pd
            if not pd.api.types.is_object_dtype(series) and not pd.api.types.is_string_dtype(series):
                return False
            sample = series.dropna().astype(str)
            if sample.empty:
                return False
            avg_len = float(sample.head(500).str.len().mean())
            unique_ratio = float(sample.nunique() / max(len(sample), 1))
            return avg_len > 24 and unique_ratio > 0.35
        except Exception:
            return False

    # ------------------------------------------------------------------
    # TARGET DISCOVERY
    # ------------------------------------------------------------------

    def _infer_target_candidates(self, columns, dataframe=None):
        candidates: list[dict[str, Any]] = []
        n = len(columns)

        for index, column in enumerate(columns):
            if not isinstance(column, dict):
                continue
            name = str(column.get("name") or column.get("normalized_name") or "").strip()
            if not name:
                continue

            normalized = self._normalize_name(name)
            tokens = set(normalized.split("_"))
            score = 0.0
            reasons: list[str] = []

            # Strong semantic clues, but never mandatory.
            if normalized in self.GENERIC_TARGET_HINTS:
                score += 42
                reasons.append("Nama kolom merupakan sinyal umum target.")
            elif tokens & self.GENERIC_TARGET_HINTS:
                score += 24
                reasons.append("Nama kolom mengandung istilah yang umum terkait target.")

            # Identifier penalty.
            if normalized in self.GENERIC_ID_HINTS or normalized.endswith("_id") or normalized.startswith("id_"):
                score -= 55
                reasons.append("Kolom terlihat seperti identifier.")

            unique_count = self._get_unique_count(column)
            semantic_type = str(column.get("semantic_type", "")).lower()

            if semantic_type == "boolean":
                score += 34
                reasons.append("Kolom boolean cocok untuk target klasifikasi.")
            elif semantic_type == "categorical":
                if 2 <= unique_count <= 20:
                    score += 22
                    reasons.append(f"Kolom kategorikal memiliki {unique_count} kelas.")
            elif self._is_numeric_column(column):
                if unique_count == 2:
                    score += 28
                    reasons.append("Kolom numerik biner berpotensi menjadi target klasifikasi.")
                elif 3 <= unique_count <= 20:
                    score += 16
                    reasons.append(f"Kolom numerik memiliki {unique_count} nilai unik.")
                elif unique_count > 20:
                    score += 12
                    reasons.append("Kolom numerik memiliki banyak nilai unik dan berpotensi menjadi target regresi.")

            # Weak structural signal: target is often near the end, but this
            # contributes only a small amount.
            if n and index >= n - 2:
                score += 10
                reasons.append("Kolom berada dekat akhir struktur dataset.")

            # A target candidate should usually have more than one value.
            if unique_count <= 1:
                score -= 40

            if score >= 20:
                candidates.append({
                    "name": name,
                    "score": round(max(0.0, min(score, 100.0)), 2),
                    "confidence": round(max(0.0, min(score, 100.0)), 2),
                    "reasons": reasons,
                })

        candidates.sort(key=lambda item: item.get("score", 0), reverse=True)
        return candidates

    @staticmethod
    def _merge_target_candidates(primary, inferred):
        merged: dict[str, dict[str, Any]] = {}
        for item in list(primary) + list(inferred):
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or item.get("column") or "").strip()
            if not name:
                continue
            key = name.lower()
            previous = merged.get(key)
            if previous is None or float(item.get("score", 0) or 0) > float(previous.get("score", 0) or 0):
                merged[key] = dict(item)
        return sorted(merged.values(), key=lambda x: float(x.get("score", 0) or 0), reverse=True)

    # ------------------------------------------------------------------
    # TASK DETECTION
    # ------------------------------------------------------------------

    def _detect_supervised_task(self, columns, candidates):
        for candidate in sorted(candidates, key=self._candidate_score, reverse=True):
            target_name = self._candidate_name(candidate)
            if not target_name:
                continue
            column = self._find_column(columns, target_name)
            if not column:
                continue
            task = self._detect_task_from_column(column)
            if task:
                candidate_score = self._candidate_score(candidate)
                task["target_confidence"] = round(candidate_score, 2)
                task["confidence"] = int(round(min(task["score"], candidate_score or task["score"]))) if candidate_score else task["score"]
                task["score"] = task["confidence"]
                task["reasons"].append(f"Keyakinan kandidat target: {candidate_score:.0f}%.")
                return task
        return None

    def _detect_task_from_column(self, column):
        name = column.get("name") or column.get("normalized_name") or ""
        semantic_type = str(column.get("semantic_type", "")).lower()
        dtype = str(column.get("dtype", "")).lower()
        unique_count = self._get_unique_count(column)

        if semantic_type == "boolean" or dtype == "bool":
            return self._make_task(
                "binary_classification", 98, name,
                ["Target bertipe boolean.", "Target memiliki dua kemungkinan kelas."],
            )

        if semantic_type == "categorical":
            if unique_count == 2:
                return self._make_task(
                    "binary_classification", 96, name,
                    ["Target bersifat kategorikal.", "Target memiliki dua kelas."],
                )
            if 3 <= unique_count <= 20:
                return self._make_task(
                    "multiclass_classification", 95, name,
                    ["Target bersifat kategorikal.", f"Target memiliki {unique_count} kelas."],
                )

        if self._is_numeric_column(column):
            if unique_count == 2:
                return self._make_task(
                    "binary_classification", 96, name,
                    ["Target numerik memiliki dua nilai unik."],
                )
            if 3 <= unique_count <= 20 and self._numeric_target_looks_discrete(column):
                return self._make_task(
                    "multiclass_classification", 90, name,
                    ["Target numerik memiliki sedikit nilai unik dan tampak diskrit."],
                )
            if unique_count > 20:
                return self._make_task(
                    "regression", 90, name,
                    ["Target numerik memiliki banyak nilai unik dan lebih konsisten dengan regresi."],
                )

        if semantic_type in {"text", "string"}:
            if unique_count == 2:
                return self._make_task("binary_classification", 78, name, ["Target teks memiliki dua nilai unik."])
            if 3 <= unique_count <= 20:
                return self._make_task("multiclass_classification", 74, name, ["Target teks memiliki jumlah kelas terbatas."])

        return None

    def _select_primary_task(self, tasks):
        if not tasks:
            return None
        supervised = [
            t for t in tasks
            if t.get("target") is not None
            and t.get("task") in {"binary_classification", "multiclass_classification", "regression"}
        ]
        if supervised:
            top_supervised = max(supervised, key=self._safe_score)
            if self._safe_score(top_supervised) >= 40:
                return top_supervised
            return max(tasks, key=self._safe_score)
        time_series = [t for t in tasks if t.get("task") == "time_series_forecasting"]
        if time_series:
            return max(time_series, key=self._safe_score)
        return max(tasks, key=self._safe_score)

    # ------------------------------------------------------------------
    # NORMALIZATION / HELPERS
    # ------------------------------------------------------------------

    def _normalize_columns(self, columns):
        if isinstance(columns, list):
            return [dict(c) for c in columns if isinstance(c, dict)]
        if isinstance(columns, dict):
            out = []
            for name, info in columns.items():
                if isinstance(info, dict):
                    item = dict(info)
                    item.setdefault("name", name)
                    out.append(item)
            return out
        return []

    def _normalize_target_candidates(self, candidates):
        if isinstance(candidates, dict):
            out = []
            for name, value in candidates.items():
                if isinstance(value, dict):
                    item = dict(value)
                    item.setdefault("name", name)
                else:
                    item = {"name": name, "score": value or 0}
                out.append(item)
            return out
        if isinstance(candidates, list):
            out = []
            for candidate in candidates:
                if isinstance(candidate, dict):
                    out.append(dict(candidate))
                elif isinstance(candidate, str):
                    out.append({"name": candidate, "score": 0})
            return out
        return []

    def _find_column(self, columns, target_name):
        target = str(target_name).strip().lower()
        for column in columns:
            name = str(column.get("name", "")).strip().lower()
            normalized = str(column.get("normalized_name", "")).strip().lower()
            if target in {name, normalized}:
                return column
        return None

    @staticmethod
    def _candidate_name(candidate):
        return candidate.get("name") or candidate.get("column") or candidate.get("column_name") or candidate.get("target")

    @staticmethod
    def _candidate_score(candidate):
        for key in ("score", "target_score", "confidence"):
            value = candidate.get(key)
            if isinstance(value, (int, float)):
                return float(value)
        return 0.0

    @staticmethod
    def _get_unique_count(column):
        for key in ("unique_count", "nunique", "cardinality"):
            try:
                value = column.get(key)
                if value is not None:
                    return max(0, int(value))
            except (TypeError, ValueError):
                pass
        return 0

    @staticmethod
    def _normalize_name(value):
        return str(value).strip().lower().replace("-", "_").replace(" ", "_")

    @staticmethod
    def _is_numeric_column(column):
        semantic = str(column.get("semantic_type", "")).lower()
        dtype = str(column.get("dtype", "")).lower()
        if semantic in {"numeric", "number"}:
            return True
        return any(token in dtype for token in ("int", "float", "double", "decimal", "number"))

    def _numeric_target_looks_discrete(self, column):
        name = self._normalize_name(column.get("name") or column.get("normalized_name") or "")
        discrete = {"class", "category", "type", "status", "group", "level", "rank", "grade", "segment", "tier", "label", "code", "state", "stage"}
        if set(name.split("_")) & discrete:
            return True
        try:
            minimum = float(column.get("min"))
            maximum = float(column.get("max"))
            return minimum >= 0 and maximum <= 20
        except (TypeError, ValueError):
            return False

    def _get_numeric_count(self, columns, characteristics, dataframe=None):
        if dataframe is not None:
            try:
                return int(dataframe.select_dtypes(include="number").shape[1])
            except Exception:
                pass
        value = characteristics.get("numeric_columns")
        if isinstance(value, int):
            return max(value, 0)
        if isinstance(value, list):
            return len(value)
        return sum(1 for c in columns if self._is_numeric_column(c))

    def _feature_count(self, columns, characteristics, dataframe=None):
        if dataframe is not None:
            try:
                return int(dataframe.shape[1])
            except Exception:
                pass
        value = characteristics.get("feature_count")
        if isinstance(value, int) and value >= 0:
            return value
        return len(columns)

    def _get_datetime_count(self, columns, characteristics, dataframe=None):
        if dataframe is not None:
            try:
                return int(sum(self._is_datetime_series(dataframe[c]) for c in dataframe.columns))
            except Exception:
                pass
        value = characteristics.get("datetime_columns")
        if isinstance(value, int):
            return max(value, 0)
        if isinstance(value, list):
            return len(value)
        return sum(1 for c in columns if str(c.get("semantic_type", "")).lower() in {"datetime", "date", "timestamp"})

    @staticmethod
    def _clustering_score(numeric_count, feature_count):
        score = 55
        if numeric_count >= 5:
            score += 10
        if feature_count >= 5:
            score += 5
        return min(score, 85)

    @staticmethod
    def _anomaly_score(numeric_count, feature_count):
        score = 50
        if numeric_count >= 5:
            score += 10
        if feature_count >= 10:
            score += 5
        return min(score, 80)

    @staticmethod
    def _time_series_score(feature_count, datetime_count):
        score = 65 + (5 if datetime_count else 0) + (5 if feature_count >= 2 else 0)
        return min(score, 80)

    def _make_task(self, task, score, target, reasons):
        score = int(round(max(0, min(score, 100))))
        return {
            "task": task,
            "label": self.TASK_LABELS.get(task, task),
            "score": score,
            "confidence": score,
            "target": target,
            "status": self.STATUS,
            "reasons": list(reasons),
        }

    @staticmethod
    def _safe_score(item):
        if not isinstance(item, dict):
            return 0.0
        value = item.get("score", 0)
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0
