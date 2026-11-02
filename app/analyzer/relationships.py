"""Bounded descriptive relationships for mixed-type tabular data."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


class RelationshipAnalyzer:
    """Compute covariance and categorical association matrices for EDA.

    These summaries are descriptive. Cramer's V is not a significance test,
    and neither matrix supports causal conclusions.
    """

    MAX_NUMERIC_COLUMNS = 30
    MAX_CATEGORICAL_COLUMNS = 20
    MAX_CATEGORY_LEVELS = 20
    MAX_ASSOCIATION_ROWS = 20_000

    def covariance(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Return sample covariance for finite, non-boolean numeric columns."""
        if dataframe is None or dataframe.empty:
            return pd.DataFrame()
        # Use the same deterministic, bounded row sample as categorical
        # associations so a covariance chart remains responsive on large data.
        numeric = self._numeric_frame(self._sample(dataframe))
        if numeric.shape[1] < 2:
            return pd.DataFrame()
        return numeric.cov(min_periods=2)

    def categorical_associations(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Return bias-corrected Cramer's V for suitable categorical columns.

        Numeric columns are deliberately excluded because their integer values
        may represent measurements or identifiers. At most 20,000 evenly
        spaced rows and 20 levels per column are considered to bound work.
        """
        if dataframe is None or dataframe.empty:
            return pd.DataFrame()

        sample = self._sample(dataframe)
        candidates = []
        for column in dataframe.columns:
            series = sample[column]
            dtype = series.dtype
            is_category = (
                pd.api.types.is_object_dtype(dtype)
                or pd.api.types.is_string_dtype(dtype)
                or isinstance(dtype, pd.CategoricalDtype)
                or pd.api.types.is_bool_dtype(dtype)
            )
            if not is_category:
                continue
            try:
                levels = int(series.nunique(dropna=True))
            except (TypeError, ValueError):
                continue
            if 2 <= levels <= self.MAX_CATEGORY_LEVELS:
                candidates.append(column)
            if len(candidates) >= self.MAX_CATEGORICAL_COLUMNS:
                break

        if len(candidates) < 2:
            return pd.DataFrame()

        result = pd.DataFrame(np.nan, index=candidates, columns=candidates, dtype=float)
        for index, left in enumerate(candidates):
            result.loc[left, left] = 1.0
            for right in candidates[index + 1 :]:
                pair = sample[[left, right]].dropna()
                try:
                    table = pd.crosstab(pair[left], pair[right])
                except (TypeError, ValueError):
                    continue
                association = self._cramers_v(table.to_numpy(dtype=float))
                result.loc[left, right] = association
                result.loc[right, left] = association
        return result

    @classmethod
    def _numeric_frame(cls, dataframe: pd.DataFrame) -> pd.DataFrame:
        if dataframe is None or dataframe.empty:
            return pd.DataFrame()
        columns = [
            column
            for column in dataframe.columns
            if pd.api.types.is_numeric_dtype(dataframe[column])
            and not pd.api.types.is_bool_dtype(dataframe[column])
            and not pd.api.types.is_complex_dtype(dataframe[column])
        ][: cls.MAX_NUMERIC_COLUMNS]
        if len(columns) < 2:
            return pd.DataFrame()
        numeric = dataframe.loc[:, columns].copy()
        for column in columns:
            numeric[column] = pd.to_numeric(numeric[column], errors="coerce")
            numeric[column] = numeric[column].where(np.isfinite(numeric[column]))
        return numeric

    @classmethod
    def _sample(cls, dataframe: pd.DataFrame) -> pd.DataFrame:
        if len(dataframe) <= cls.MAX_ASSOCIATION_ROWS:
            return dataframe
        step = math.ceil(len(dataframe) / cls.MAX_ASSOCIATION_ROWS)
        return dataframe.iloc[::step].head(cls.MAX_ASSOCIATION_ROWS)

    @staticmethod
    def _cramers_v(table: np.ndarray) -> float:
        """Calculate finite, bias-corrected Cramer's V from a contingency table."""
        if table.ndim != 2 or min(table.shape) < 2:
            return float("nan")
        n = float(table.sum())
        if n <= 1 or not np.isfinite(table).all():
            return float("nan")

        try:
            from scipy.stats import chi2_contingency

            chi_squared = float(chi2_contingency(table, correction=False)[0])
        except (ValueError, FloatingPointError):
            return float("nan")

        rows, columns = table.shape
        phi_squared = chi_squared / n
        corrected_phi_squared = max(
            0.0,
            phi_squared - ((columns - 1) * (rows - 1)) / (n - 1),
        )
        corrected_rows = rows - ((rows - 1) ** 2) / (n - 1)
        corrected_columns = columns - ((columns - 1) ** 2) / (n - 1)
        denominator = min(corrected_rows - 1, corrected_columns - 1)
        if denominator <= 0:
            return float("nan")
        value = math.sqrt(corrected_phi_squared / denominator)
        return float(value) if math.isfinite(value) else float("nan")
