from __future__ import annotations

import re
from collections import Counter

import pandas as pd


class KeywordExtractor:
    """
    Extract research-related keywords from a dataset.

    Sumber keyword:
    - Nama kolom
    - Nilai kategorikal
    - Nama file jika tersedia
    - Target candidate
    """

    DEFAULT_STOP_WORDS = {
        "id",
        "no",
        "num",
        "number",
        "value",
        "data",
        "column",
        "field",
        "the",
        "of",
        "and",
        "or",
        "to",
        "for",
        "with",
        "from",
        "date",
        "time",
    }
    MAX_VALUE_SAMPLE_ROWS = 10_000

    def __init__(
        self,
        stop_words: set[str] | None = None,
    ):
        self.stop_words = (
            stop_words
            if stop_words is not None
            else self.DEFAULT_STOP_WORDS
        )

    def extract(
        self,
        dataframe: pd.DataFrame,
        fingerprint: dict | None = None,
        filename: str | None = None,
    ) -> dict:
        """
        Extract keywords from dataset.
        """

        column_keywords = self._extract_column_keywords(
            dataframe
        )

        value_keywords = self._extract_value_keywords(
            dataframe
        )

        filename_keywords = self._extract_filename_keywords(
            filename
        )

        target_keywords = self._extract_target_keywords(
            fingerprint
        )

        all_keywords = (
            column_keywords
            + value_keywords
            + filename_keywords
            + target_keywords
        )

        frequencies = Counter(all_keywords)

        ranked_keywords = [
            {
                "keyword": keyword,
                "frequency": frequency,
            }
            for keyword, frequency in frequencies.most_common()
        ]

        return {
            "keywords": [
                item["keyword"]
                for item in ranked_keywords
            ],
            "ranked_keywords": ranked_keywords,
            "sources": {
                "columns": sorted(
                    set(column_keywords)
                ),
                "values": sorted(
                    set(value_keywords)
                ),
                "filename": sorted(
                    set(filename_keywords)
                ),
                "target": sorted(
                    set(target_keywords)
                ),
            },
        }

    def _extract_column_keywords(
        self,
        dataframe: pd.DataFrame,
    ) -> list[str]:

        keywords = []

        for column in dataframe.columns:

            normalized = self._normalize_text(
                str(column)
            )

            keywords.extend(
                self._tokenize(normalized)
            )

        return keywords

    def _extract_value_keywords(
        self,
        dataframe: pd.DataFrame,
    ) -> list[str]:

        keywords = []

        categorical_columns = dataframe.select_dtypes(
            include=["object", "category", "string"]
        ).columns

        for column in categorical_columns:

            series = dataframe[column]

            # Value discovery only needs common labels. Sampling large columns
            # at a fixed stride avoids repeated full-column scans while still
            # observing values from across the dataset.
            if len(series) > self.MAX_VALUE_SAMPLE_ROWS:
                stride = max(
                    1,
                    (len(series) + self.MAX_VALUE_SAMPLE_ROWS - 1)
                    // self.MAX_VALUE_SAMPLE_ROWS,
                )
                series = series.iloc[::stride].head(self.MAX_VALUE_SAMPLE_ROWS)

            series = series.dropna()

            if series.empty:
                continue

            unique_count = series.nunique()

            # Hindari mengambil nilai dari kolom
            # dengan cardinality terlalu tinggi.
            if unique_count > 20:
                continue

            values = (
                series.astype(str)
                .value_counts()
                .head(10)
                .index
            )

            for value in values:

                normalized = self._normalize_text(
                    value
                )

                keywords.extend(
                    self._tokenize(normalized)
                )

        return keywords

    def _extract_filename_keywords(
        self,
        filename: str | None,
    ) -> list[str]:

        if not filename:
            return []

        filename = re.sub(
            r"\.[^.]+$",
            "",
            filename,
        )

        normalized = self._normalize_text(
            filename
        )

        return self._tokenize(normalized)

    def _extract_target_keywords(
        self,
        fingerprint: dict | None,
    ) -> list[str]:

        if not fingerprint:
            return []

        representation = fingerprint.get(
            "representation",
            fingerprint,
        )

        targets = representation.get(
            "target_candidates",
            [],
        )

        keywords = []

        for target in targets:

            normalized = self._normalize_text(
                str(target)
            )

            keywords.extend(
                self._tokenize(normalized)
            )

        return keywords

    def _normalize_text(
        self,
        text: str,
    ) -> str:

        text = text.strip().lower()

        text = re.sub(
            r"[^a-z0-9]+",
            " ",
            text,
        )

        return text

    def _tokenize(
        self,
        text: str,
    ) -> list[str]:

        tokens = text.split()

        result = []

        for token in tokens:

            if token in self.stop_words:
                continue

            if len(token) < 2:
                continue

            if token.isdigit():
                continue

            result.append(token)

        return result
