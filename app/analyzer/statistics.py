import numpy as np
import pandas as pd


class DatasetStatistics:
    """Calculate statistics for dataset columns."""

    def analyze(self, dataframe: pd.DataFrame) -> list[dict]:
        results = []

        total_rows = len(dataframe)

        for column in dataframe.columns:
            series = dataframe[column]

            missing_count = int(series.isna().sum())
            unique_count = int(series.nunique(dropna=True))

            result = {
                "column": str(column),
                "dtype": str(series.dtype),
                "missing_count": missing_count,
                "missing_percentage": (
                    missing_count / total_rows * 100
                    if total_rows > 0
                    else 0
                ),
                "unique_count": unique_count,
                "unique_percentage": (
                    unique_count / total_rows * 100
                    if total_rows > 0
                    else 0
                ),
            }

            # Boolean values are categories, not continuous measurements.
            # Treating them as numeric can route them through reducers such as
            # skewness that may subtract boolean arrays on some NumPy versions.
            if pd.api.types.is_bool_dtype(series):
                value_counts = series.value_counts(dropna=True)
                result.update(
                    {
                        "category_count": unique_count,
                        "top_values": [
                            {"value": str(value), "frequency": int(frequency)}
                            for value, frequency in value_counts.head(5).items()
                        ],
                    }
                )

            elif pd.api.types.is_numeric_dtype(series):
                non_missing = series.dropna()
                # Infinity is not a valid observation for finite descriptive
                # statistics. Keep its count separate from actual missing data.
                if np.iscomplexobj(non_missing.to_numpy()):
                    finite = non_missing.iloc[0:0]
                    result["numeric_summary_status"] = "unsupported_complex"
                else:
                    try:
                        values = non_missing.to_numpy(dtype=float, na_value=np.nan)
                        finite = non_missing.iloc[np.flatnonzero(np.isfinite(values))]
                    except (TypeError, ValueError, OverflowError):
                        finite = non_missing.iloc[0:0]
                        result["numeric_summary_status"] = "conversion_failed"
                result["non_finite_count"] = int(len(non_missing) - len(finite)) if "numeric_summary_status" not in result else None
                result["finite_count"] = int(len(finite))
                result.update(
                    {
                        "mean": float(finite.mean())
                        if not finite.empty
                        else None,
                        "median": float(finite.median())
                        if not finite.empty
                        else None,
                        "std": float(finite.std())
                        if not finite.empty
                        else None,
                        "min": float(finite.min())
                        if not finite.empty
                        else None,
                        "max": float(finite.max())
                        if not finite.empty
                        else None,
                        "q1": float(finite.quantile(0.25))
                        if not finite.empty
                        else None,
                        "q3": float(finite.quantile(0.75))
                        if not finite.empty
                        else None,
                        "skewness": float(finite.skew())
                        if not finite.empty
                        else None,
                    }
                )
                for metric in ("mean", "median", "std", "min", "max", "q1", "q3", "skewness"):
                    value = result[metric]
                    if value is not None and not np.isfinite(value):
                        result[metric] = None

            elif (
                pd.api.types.is_object_dtype(series)
                or pd.api.types.is_string_dtype(series)
                or isinstance(series.dtype, pd.CategoricalDtype)
            ):
                value_counts = series.value_counts(dropna=True)

                result.update(
                    {
                        "category_count": unique_count,
                        "top_values": [
                            {
                                "value": str(value),
                                "frequency": int(frequency),
                            }
                            for value, frequency in value_counts.head(5).items()
                        ],
                    }
                )

            results.append(result)

        return results
