from __future__ import annotations

import csv
from collections import Counter
from io import StringIO
from pathlib import Path

import pandas as pd

from app.core.constants import MAX_DATASET_SIZE_MB
from app.core.exceptions import DatasetLoadError
from app.analyzer.cleaner import MISSING_TOKENS
from app.analyzer.missing_values import normalize_missing_values


class DatasetLoader:
    """
    Load and validate CSV-like tabular datasets.

    The loader automatically detects common delimiters instead of assuming
    commas. This allows datasets exported by different tools to work without
    dataset-specific rules.
    """

    COMMON_DELIMITERS = [",", ";", "\t", "|", ":", "^", "~"]
    ENCODINGS = ["utf-8-sig", "utf-8", "cp1252", "latin1"]
    CSV_ENGINES = ("c", "python")

    SUPPORTED_SUFFIXES = {".csv", ".tsv", ".xlsx", ".xls", ".json", ".parquet"}

    def load(self, file_path: str, chunksize: int = 100_000) -> pd.DataFrame:
        """Baca CSV, Excel, JSON, atau Parquet dan normalkan placeholder."""
        path = Path(file_path)
        if not path.exists():
            raise DatasetLoadError("File dataset tidak ditemukan.")
        file_size = path.stat().st_size
        file_size_mb = file_size / (1024 * 1024)
        if file_size_mb > MAX_DATASET_SIZE_MB:
            raise DatasetLoadError(
                f"Ukuran dataset terlalu besar. "
                f"Maksimal {MAX_DATASET_SIZE_MB} MB."
            )

        if file_size == 0:
            raise DatasetLoadError("File dataset kosong.")
        suffix = path.suffix.lower()
        if suffix not in self.SUPPORTED_SUFFIXES:
            raise DatasetLoadError(
                "Format tidak didukung. Gunakan CSV, TSV, XLSX, XLS, JSON, atau Parquet."
            )
        try:
            if suffix in {".csv", ".tsv"}:
                sample_text = self._read_sample(path)
                delimiter, skiprows, sample_text = self._csv_read_options(
                    sample_text,
                    preferred="\t" if suffix == ".tsv" else None,
                )
                dataframe = self._read_dataframe(
                    path,
                    delimiter,
                    chunksize=chunksize,
                    sample_text=sample_text,
                    skiprows=skiprows,
                )
            elif suffix in {".xlsx", ".xls"}:
                dataframe = pd.read_excel(path)
            elif suffix == ".json":
                dataframe = pd.read_json(path)
            else:
                dataframe = pd.read_parquet(path)
        except UnicodeDecodeError as error:
            raise DatasetLoadError(
                "Encoding CSV tidak dapat dibaca."
            ) from error
        except pd.errors.ParserError as error:
            raise DatasetLoadError(
                f"CSV tidak dapat dibaca setelah mencoba delimiter dan engine umum: {error}"
            ) from error
        except DatasetLoadError:
            raise
        except Exception as error:
            raise DatasetLoadError(
                f"Gagal membaca dataset: {error}"
            ) from error

        return self._finalize_dataframe(dataframe)

    def load_csv(self, file_path: str, chunksize: int = 100_000) -> pd.DataFrame:
        """Wrapper kompatibel untuk pemanggil lama yang khusus membaca CSV."""
        if Path(file_path).suffix.lower() != ".csv":
            raise DatasetLoadError("Format dataset harus CSV.")
        return self.load(file_path, chunksize=chunksize)

    def _finalize_dataframe(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(dataframe, pd.DataFrame):
            raise DatasetLoadError("File tidak berisi tabel DataFrame yang didukung.")
        if dataframe.empty:
            raise DatasetLoadError("Dataset tidak memiliki data.")

        if len(dataframe.columns) == 0:
            raise DatasetLoadError("Dataset tidak memiliki kolom.")

        # Clean up accidental whitespace/BOM in column names while preserving
        # the actual data values.
        dataframe.columns = [
            str(column).replace("\ufeff", "").strip()
            for column in dataframe.columns
        ]

        return normalize_missing_values(dataframe)

    # =============================================================
    # SAMPLE / ENCODING
    # =============================================================

    def _read_sample(self, path: Path, sample_size: int = 64 * 1024) -> str:
        """
        Read a small text sample using common CSV encodings.
        """

        with path.open("rb") as source:
            raw = source.read(sample_size)

        last_error: Exception | None = None

        for encoding in self._encoding_candidates(raw):
            # A bounded sample can end halfway through a multibyte character.
            for trim in range(4):
                sample = raw[:-trim] if trim else raw
                try:
                    text = sample.decode(encoding)
                    if len(raw) == sample_size:
                        last_newline = text.rfind("\n")
                        if last_newline >= 0:
                            text = text[:last_newline + 1]
                    return text
                except UnicodeDecodeError as error:
                    last_error = error

        if last_error is not None:
            raise last_error

        raise DatasetLoadError(
            "Encoding CSV tidak dapat dibaca."
        )

    def _read_dataframe(
        self,
        path: Path,
        delimiter: str | None,
        chunksize: int | None = None,
        sample_text: str | None = None,
        skiprows: int = 0,
    ) -> pd.DataFrame:
        """
        Read the dataframe using a detected delimiter.

        The encoding loop makes the loader tolerant of common exported CSV
        encodings without changing the dataset content.
        """

        delimiters = (
            self._delimiter_candidates(sample_text, preferred=delimiter)
            if sample_text is not None
            else [delimiter or ",", *[d for d in self.COMMON_DELIMITERS if d != (delimiter or ",")]]
        )
        last_decode_error: Exception | None = None
        last_parser_error: Exception | None = None
        single_column_dataframe: pd.DataFrame | None = None

        with path.open("rb") as source:
            prefix = source.read(4096)
        encodings = self._encoding_candidates(prefix)

        for encoding in encodings:
            decode_failed = False
            for candidate in delimiters:
                engines = (
                    ("python",)
                    if len(candidate.encode("utf-8")) > 1
                    else self.CSV_ENGINES
                )
                for engine in engines:
                    quote_options = [
                        {"quoting": csv.QUOTE_MINIMAL, "quotechar": '"', "doublequote": True},
                        {
                            "quoting": csv.QUOTE_MINIMAL,
                            "quotechar": '"',
                            "doublequote": True,
                            "escapechar": "\\",
                        },
                    ]
                    if sample_text is not None and self._quote_none_is_consistent(
                        sample_text, candidate
                    ):
                        # Some exports contain literal, unescaped quote marks.
                        # Only disable quote handling when that preserves the
                        # same field count throughout the sample.
                        quote_options.append(
                            {"quoting": csv.QUOTE_NONE, "escapechar": "\\"}
                        )

                    for quote_option in quote_options:
                        try:
                            parsed = pd.read_csv(
                                path,
                                sep=candidate,
                                engine=engine,
                                encoding=encoding,
                                na_values=sorted(MISSING_TOKENS),
                                skipinitialspace=True,
                                skiprows=skiprows,
                                on_bad_lines="error",
                                chunksize=chunksize,
                                **quote_option,
                            )
                            if chunksize is not None:
                                chunks = list(parsed)
                                dataframe = (
                                    pd.concat(chunks, ignore_index=True)
                                    if chunks
                                    else pd.DataFrame()
                                )
                            else:
                                dataframe = parsed
                            if sample_text is not None and len(dataframe.columns) <= 1:
                                if single_column_dataframe is None:
                                    single_column_dataframe = dataframe
                                # A wrong separator often parses successfully
                                # as one column. Keep looking for a better one.
                                continue
                            return dataframe
                        except UnicodeDecodeError as error:
                            last_decode_error = error
                            decode_failed = True
                            break
                        except (
                            csv.Error,
                            pd.errors.ParserError,
                            pd.errors.EmptyDataError,
                            ValueError,
                        ) as error:
                            # The Python engine may raise _csv.Error directly
                            # (for example: "; expected after \""). Treat it
                            # as a failed attempt so other dialects can run.
                            last_parser_error = error
                    if decode_failed:
                        break
                if decode_failed:
                    break

        fallback_dataframe = self._read_csv_fallback(
            path,
            delimiters=delimiters,
            encodings=encodings,
            skiprows=skiprows,
        )
        if fallback_dataframe is not None:
            return fallback_dataframe
        if single_column_dataframe is not None:
            return single_column_dataframe

        if last_parser_error is not None:
            raise last_parser_error
        if last_decode_error is not None:
            raise last_decode_error
        raise DatasetLoadError("CSV tidak memiliki isi tabel yang dapat dibaca.")

    @staticmethod
    def _read_csv_fallback(
        path: Path,
        delimiters: list[str],
        encodings: list[str],
        skiprows: int = 0,
    ) -> pd.DataFrame | None:
        """Recover CSVs rejected by pandas using Python's more permissive reader."""
        quote_options = [
            {"quoting": csv.QUOTE_MINIMAL, "quotechar": '"', "doublequote": True},
            {
                "quoting": csv.QUOTE_MINIMAL,
                "quotechar": '"',
                "doublequote": True,
                "escapechar": "\\",
            },
            {"quoting": csv.QUOTE_MINIMAL, "quotechar": "'", "doublequote": True},
            {"quoting": csv.QUOTE_NONE, "escapechar": "\\"},
        ]
        best_frame: pd.DataFrame | None = None
        best_score: tuple[float, int, int, int] | None = None

        for encoding in encodings:
            try:
                text = path.read_text(encoding=encoding)
            except UnicodeDecodeError:
                continue

            for delimiter_index, delimiter in enumerate(delimiters):
                for quote_index, options in enumerate(quote_options):
                    try:
                        reader = csv.reader(
                            StringIO(text, newline=""),
                            delimiter=delimiter,
                            skipinitialspace=True,
                            strict=False,
                            **options,
                        )
                        rows = []
                        for row_index, row in enumerate(reader):
                            if row_index < skiprows:
                                continue
                            if row and any(value.strip() for value in row):
                                rows.append(row)
                    except (csv.Error, TypeError, ValueError):
                        continue

                    if len(rows) < 2 or not rows[0]:
                        continue
                    column_count = len(rows[0])
                    data_rows = rows[1:]
                    if any(len(row) > column_count for row in data_rows):
                        continue

                    exact_width_rows = sum(
                        len(row) == column_count for row in data_rows
                    )
                    consistency = exact_width_rows / max(len(data_rows), 1)
                    # The fallback can fill absent trailing cells, but a
                    # delimiter that invents extra fields is rejected above.
                    if consistency < 0.8:
                        continue
                    score = (
                        consistency,
                        column_count,
                        -delimiter_index,
                        -quote_index,
                    )
                    if best_score is not None and score <= best_score:
                        continue

                    headers = DatasetLoader._unique_headers(rows[0])
                    normalized_rows = []
                    for row in data_rows:
                        values = list(row) + [""] * (column_count - len(row))
                        normalized_rows.append(values)
                    frame = pd.DataFrame(normalized_rows, columns=headers)
                    DatasetLoader._infer_fallback_column_types(frame)
                    best_frame = frame
                    best_score = score

        return best_frame

    @staticmethod
    def _unique_headers(raw_headers: list[str]) -> list[str]:
        """Match pandas' duplicate-header behavior for the fallback dataframe."""
        headers: list[str] = []
        used: set[str] = set()
        occurrences: dict[str, int] = {}
        for index, raw_header in enumerate(raw_headers, start=1):
            base = str(raw_header).replace("\ufeff", "").strip() or f"column_{index}"
            count = occurrences.get(base, 0)
            header = base if count == 0 else f"{base}.{count}"
            while header in used:
                count += 1
                header = f"{base}.{count}"
            occurrences[base] = count + 1
            used.add(header)
            headers.append(header)
        return headers

    @staticmethod
    def _infer_fallback_column_types(dataframe: pd.DataFrame) -> None:
        """Retain basic pandas-like numeric and boolean inference on fallback."""
        missing_values = {value.casefold() for value in MISSING_TOKENS}
        for column in dataframe.columns:
            series = dataframe[column].map(
                lambda value: (
                    pd.NA
                    if isinstance(value, str) and value.strip().casefold() in missing_values
                    else value
                )
            )
            observed = series.dropna()
            if observed.empty:
                dataframe[column] = series
                continue

            lowered = observed.map(lambda value: str(value).strip().casefold())
            if lowered.isin({"true", "false"}).all():
                boolean_values = series.map(
                    lambda value: (
                        pd.NA
                        if pd.isna(value)
                        else str(value).strip().casefold() == "true"
                    )
                )
                dataframe[column] = boolean_values.astype("boolean")
                continue

            numeric = pd.to_numeric(series, errors="coerce")
            if numeric.loc[observed.index].notna().all():
                dataframe[column] = numeric
            else:
                dataframe[column] = series

    @staticmethod
    def _quote_none_is_consistent(sample_text: str, delimiter: str) -> bool:
        """Allow literal-quote fallback only when it keeps a stable row width."""
        try:
            rows = [
                row
                for row in csv.reader(
                    StringIO(sample_text),
                    delimiter=delimiter,
                    quoting=csv.QUOTE_NONE,
                    escapechar="\\",
                )
                if row and any(value.strip() for value in row)
            ][:60]
        except csv.Error:
            return False

        if len(rows) < 2:
            return False
        expected_width = len(rows[0])
        if expected_width <= 1:
            return False
        matching_rows = sum(len(row) == expected_width for row in rows[1:])
        return matching_rows / (len(rows) - 1) >= 0.95

    @classmethod
    def _encoding_candidates(cls, raw: bytes) -> list[str]:
        """Prefer BOM/byte-pattern encoding detection, then common CSV encodings."""
        if raw.startswith((b"\xff\xfe\x00\x00", b"\x00\x00\xfe\xff")):
            return ["utf-32"]
        elif raw.startswith((b"\xff\xfe", b"\xfe\xff")):
            return ["utf-16"]
        elif raw.startswith(b"\xef\xbb\xbf"):
            return ["utf-8-sig"]
        elif len(raw) >= 8:
            even_nuls = sum(raw[index] == 0 for index in range(0, len(raw), 2))
            odd_nuls = sum(raw[index] == 0 for index in range(1, len(raw), 2))
            pairs = max(1, len(raw) // 2)
            if even_nuls / pairs > 0.3:
                return ["utf-16-be"]
            elif odd_nuls / pairs > 0.3:
                return ["utf-16-le"]

        return list(cls.ENCODINGS)

    # =============================================================
    # DELIMITER DETECTION
    # =============================================================

    def _detect_delimiter(self, sample_text: str) -> str:
        """
        Detect a delimiter by comparing quote-aware field counts first.
        """

        if not sample_text.strip():
            raise DatasetLoadError("CSV tidak memiliki isi yang dapat dibaca.")
        return self._delimiter_candidates(sample_text)[0]

    def _csv_read_options(
        self,
        sample_text: str,
        preferred: str | None = None,
    ) -> tuple[str, int, str]:
        """Handle Excel's optional ``sep=;`` directive before delimiter sniffing."""
        lines = sample_text.splitlines()
        skiprows = 0
        directive = None
        for index, line in enumerate(lines):
            value = line.strip()
            if not value:
                continue
            if value.casefold().startswith("sep=") and len(value) == 5:
                directive = value[-1]
                skiprows = index + 1
                sample_text = "\n".join(lines[index + 1:])
            break

        delimiter = directive or preferred or self._detect_delimiter(sample_text)
        return delimiter, skiprows, sample_text

    def _delimiter_candidates(
        self,
        sample_text: str,
        preferred: str | None = None,
    ) -> list[str]:
        """Rank delimiters by quote-aware consistency across CSV records."""
        scored: list[tuple[float, float, int, str]] = []
        for delimiter in self.COMMON_DELIMITERS:
            widths: list[int] = []
            try:
                reader = csv.reader(StringIO(sample_text), delimiter=delimiter)
                for row in reader:
                    if row and any(value.strip() for value in row):
                        widths.append(len(row))
                    if len(widths) >= 60:
                        break
            except csv.Error:
                continue

            if not widths:
                continue
            common_width, common_rows = Counter(widths).most_common(1)[0]
            consistency = common_rows / len(widths)
            if common_width <= 1 or consistency < 0.5:
                continue
            score = common_width * consistency
            scored.append((score, consistency, common_width, delimiter))

        scored.sort(key=lambda entry: entry[:3], reverse=True)
        candidates = (
            [preferred]
            if preferred and len(preferred) == 1 and preferred not in {"\r", "\n", '"', "'"}
            else []
        )
        candidates.extend(delimiter for _, _, _, delimiter in scored)
        try:
            dialect = csv.Sniffer().sniff(sample_text[:65536])
            if (
                len(dialect.delimiter) == 1
                and not dialect.delimiter.isspace()
                and not dialect.delimiter.isalnum()
                and dialect.delimiter not in {'"', "'"}
            ):
                widths: list[int] = []
                reader = csv.reader(
                    StringIO(sample_text), delimiter=dialect.delimiter
                )
                for row in reader:
                    if row and any(value.strip() for value in row):
                        widths.append(len(row))
                    if len(widths) >= 60:
                        break
                if widths:
                    common_width, common_rows = Counter(widths).most_common(1)[0]
                    if common_width > 1 and common_rows / len(widths) >= 0.5:
                        candidates.append(dialect.delimiter)
        except (csv.Error, TypeError, ValueError):
            pass
        candidates.extend(self.COMMON_DELIMITERS)
        return list(dict.fromkeys(candidates))
