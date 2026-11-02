import pandas as pd
import pytest

from app.analyzer.loader import DatasetLoader
from app.core.exceptions import DatasetLoadError


@pytest.mark.parametrize("suffix", [".json", ".jsonl", ".parquet"])
def test_loader_supports_json_jsonl_and_parquet(tmp_path, suffix):
    source = tmp_path / f"dataset{suffix}"
    dataframe = pd.DataFrame({"amount": [1, 2], "label": ["ok", "NA"]})
    if suffix == ".json":
        dataframe.to_json(source, orient="records")
    elif suffix == ".jsonl":
        dataframe.to_json(source, orient="records", lines=True)
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


def test_loader_reads_tab_separated_file(tmp_path):
    source = tmp_path / "dataset.tsv"
    source.write_text("value\tlabel\n1\talpha\n2\tbeta\n", encoding="utf-8")

    loaded = DatasetLoader().load(str(source))

    assert loaded.columns.tolist() == ["value", "label"]
    assert loaded["value"].tolist() == [1, 2]
    assert loaded["label"].tolist() == ["alpha", "beta"]


def test_csv_default_and_explicit_chunked_load_return_same_dataframe(tmp_path):
    source = tmp_path / "chunked.csv"
    source.write_text(
        "value,label\n1,alpha\n2,beta\n3,gamma\n4,delta\n5,epsilon\n",
        encoding="utf-8",
    )

    loader = DatasetLoader()
    default_loaded = loader.load_csv(str(source))
    chunked_loaded = loader.load_csv(str(source), chunksize=2)

    pd.testing.assert_frame_equal(default_loaded, chunked_loaded)


def test_loader_reports_unsupported_extension(tmp_path):
    source = tmp_path / "dataset.txt"
    source.write_text("value\n1\n", encoding="utf-8")

    with pytest.raises(DatasetLoadError, match="Format tidak didukung"):
        DatasetLoader().load(str(source))


def test_loader_serializes_nested_json_cells_for_tabular_analysis(tmp_path):
    source = tmp_path / "nested.jsonl"
    source.write_text(
        '{"amount": 1, "meta": {"region": "A"}, "tags": ["x", "y"]}\n'
        '{"amount": 2, "meta": {"region": "B"}, "tags": ["z"]}\n',
        encoding="utf-8",
    )

    loaded = DatasetLoader().load(str(source))

    assert loaded["meta"].tolist() == ['{"region": "A"}', '{"region": "B"}']
    assert loaded["tags"].tolist() == ['["x", "y"]', '["z"]']


def test_loader_rejects_column_collision_after_normalizing_names():
    dataframe = pd.DataFrame([[1, 2]], columns=["feature", " feature "])

    with pytest.raises(DatasetLoadError, match="duplikat"):
        DatasetLoader()._finalize_dataframe(dataframe)
