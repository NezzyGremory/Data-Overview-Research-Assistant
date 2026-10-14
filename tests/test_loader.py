import pandas as pd
import pytest

from app.analyzer.loader import DatasetLoader
from app.core.exceptions import DatasetLoadError


@pytest.mark.parametrize("suffix", [".json", ".parquet"])
def test_loader_supports_json_and_parquet(tmp_path, suffix):
    source = tmp_path / f"dataset{suffix}"
    dataframe = pd.DataFrame({"amount": [1, 2], "label": ["ok", "NA"]})
    if suffix == ".json":
        dataframe.to_json(source, orient="records")
    else:
        dataframe.to_parquet(source, index=False)

    loaded = DatasetLoader().load(str(source))

    assert loaded.columns.tolist() == ["amount", "label"]
    assert loaded["label"].iloc[0] == "ok"
    assert pd.isna(loaded["label"].iloc[1])


def test_loader_supports_xlsx_when_excel_engine_is_installed(tmp_path):
    pytest.importorskip("openpyxl")
    source = tmp_path / "dataset.xlsx"
    pd.DataFrame({"value": [1, 2]}).to_excel(source, index=False)

    loaded = DatasetLoader().load(str(source))

    assert loaded["value"].tolist() == [1, 2]


def test_loader_detects_semicolon_csv_and_indonesian_decimal(tmp_path):
    source = tmp_path / "padi.csv"
    source.write_text("Jumlah;Keterangan\n2.286,09;tersedia\n-;kosong\n", encoding="utf-8")

    loaded = DatasetLoader().load_csv(str(source))

    assert loaded.columns.tolist() == ["Jumlah", "Keterangan"]
    assert loaded["Jumlah"].iloc[0] == pytest.approx(2286.09)
    assert pd.isna(loaded["Jumlah"].iloc[1])


def test_loader_reports_unsupported_extension(tmp_path):
    source = tmp_path / "dataset.txt"
    source.write_text("value\n1\n", encoding="utf-8")

    with pytest.raises(DatasetLoadError, match="Format tidak didukung"):
        DatasetLoader().load(str(source))
