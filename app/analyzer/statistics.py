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
                # Reuse the filtered series for every statistic. Repeated
                # dropna() calls each allocate and scan a copy of this column.
                non_missing = series.dropna()
                result.update(
                    {
                        "mean": float(non_missing.mean())
                        if not non_missing.empty
                        else None,
                        "median": float(non_missing.median())
                        if not non_missing.empty
                        else None,
                        "std": float(non_missing.std())
                        if not non_missing.empty
                        else None,
                        "min": float(non_missing.min())
                        if not non_missing.empty
                        else None,
                        "max": float(non_missing.max())
                        if not non_missing.empty
                        else None,
                        "q1": float(non_missing.quantile(0.25))
                        if not non_missing.empty
                        else None,
                        "q3": float(non_missing.quantile(0.75))
                        if not non_missing.empty
                        else None,
                        "skewness": float(non_missing.skew())
                        if not non_missing.empty
                        else None,
                    }
                )

            elif pd.api.types.is_object_dtype(series) or pd.api.types.is_categorical_dtype(series):
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
