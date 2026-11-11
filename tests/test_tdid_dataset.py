from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from app.ai.tdid_dataset import (
    TDIDDatasetBuilder,
    TDIDError,
    TDIDVersionStore,
)


def _record(index: int, *, source_group: str | None = None, verified: bool = True):
    return {
        "record_id": f"tdid-{index}",
        "messages": [
            {"role": "user", "content": f"Explain concept {index}."},
            {"role": "assistant", "content": f"A grounded explanation of concept {index}."},
        ],
        "metadata": {
            "category": "data_science",
            "difficulty": "beginner",
            "instruction_type": "knowledge",
            "verified": verified,
            "source": f"Reference {index}" if verified else "",
            "source_group": source_group or f"group-{index}",
        },
    }


def test_jsonl_import_reports_invalid_lines_and_retains_valid_records(tmp_path: Path):
    path = tmp_path / "examples.jsonl"
    path.write_text("not json\n" + json.dumps(_record(1)) + "\n", encoding="utf-8")

    builder = TDIDDatasetBuilder.from_jsonl(path)

    assert len(builder.records) == 1
    assert len(builder.import_issues) == 1
    assert builder.import_issues[0].record_index == 1
    assert builder.validate()[0].field == "json"


def test_csv_import_line_errors_do_not_exclude_valid_records_from_split(tmp_path: Path):
    path = tmp_path / "examples.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["messages", "category", "source", "verified"])
        writer.writeheader()
        writer.writerow({"messages": "{broken", "category": "data_science", "source": "bad", "verified": "yes"})
        for index in range(3):
            writer.writerow(
                {
                    "messages": json.dumps(_record(index)["messages"]),
                    "category": "data_science",
                    "source": f"reference-{index}",
                    "verified": "yes",
                }
            )

    builder = TDIDDatasetBuilder.from_csv(
        path,
        field_map={"messages": "messages", "category": "category", "source": "source", "verified": "verified"},
    )
    split = builder.split()

    assert len(builder.records) == 3
    assert len(builder.import_issues) == 1
    assert sum(map(len, (split.train, split.validation, split.test))) == 3


def test_csv_import_maps_user_assistant_columns(tmp_path: Path):
    path = tmp_path / "simple.csv"
    path.write_text(
        "question,answer,category,source\nWhat is a mean?,Arithmetic average.,statistics,NIST\n",
        encoding="utf-8",
    )

    builder = TDIDDatasetBuilder.from_csv(
        path,
        field_map={
            "user": "question",
            "assistant": "answer",
            "category": "category",
            "source": "source",
        },
    )

    assert builder.validate() == ()
    assert builder.records[0]["messages"][0]["content"] == "What is a mean?"


def test_records_without_ids_receive_stable_ids_for_split_manifests():
    builder = TDIDDatasetBuilder([_record(index) for index in range(3)])
    for record in builder.records:
        record.pop("record_id")
    # Reconstructing a builder is the supported import/normalization boundary.
    normalized = TDIDDatasetBuilder(builder.records)
    split = normalized.split()
    ids = [record_id for part in ("train", "validation", "test") for record_id in split.manifest()["record_ids"][part]]
    assert all(ids)
    assert len(set(ids)) == 3


def test_validation_requires_verified_source_and_valid_categories():
    record = _record(0)
    record["metadata"].pop("source")
    record["metadata"]["category"] = "not-a-category"

    issues = TDIDDatasetBuilder([record]).validate()

    assert {issue.field for issue in issues} >= {"metadata.category", "metadata.source"}


def test_duplicate_detection_and_grouped_split_exclusions():
    records = [_record(i) for i in range(6)]
    records[1] = {**records[0], "record_id": "duplicate-copy"}
    records.append(_record(7, verified=False))
    builder = TDIDDatasetBuilder(records)

    split = builder.split(seed=7)
    split_ids = [item["record_id"] for part in (split.train, split.validation, split.test) for item in part]

    assert builder.duplicate_groups() == ((0, 1),)
    assert "duplicate-copy" in split.excluded_duplicates
    assert "tdid-7" in split.excluded_unverified
    assert len(split_ids) == len(set(split_ids)) == 5
    assert split.manifest()["seed"] == 7


def test_source_groups_never_cross_splits_and_insufficient_groups_fail():
    builder = TDIDDatasetBuilder(
        [_record(0, source_group="shared"), _record(1, source_group="shared")]
        + [_record(i, source_group=f"unique-{i}") for i in range(2, 7)]
    )

    split = builder.split(seed=11)
    split_groups = [
        {record["metadata"]["source_group"] for record in records}
        for records in (split.train, split.validation, split.test)
    ]
    assert not (split_groups[0] & split_groups[1])
    assert not (split_groups[0] & split_groups[2])
    assert not (split_groups[1] & split_groups[2])

    with pytest.raises(ValueError, match="three independent source groups"):
        TDIDDatasetBuilder([_record(0, source_group="same"), _record(1, source_group="same"), _record(2, source_group="same")]).split()


def test_repeated_prompts_stay_in_one_split_even_across_source_groups():
    records = [_record(index) for index in range(6)]
    records[1]["messages"][0]["content"] = records[0]["messages"][0]["content"]
    builder = TDIDDatasetBuilder(records)

    split = builder.split(seed=4)
    locations = {
        record["record_id"]: split_name
        for split_name, values in (
            ("train", split.train),
            ("validation", split.validation),
            ("test", split.test),
        )
        for record in values
    }

    assert locations["tdid-0"] == locations["tdid-1"]


def test_versions_are_immutable_hash_verified_and_final_requires_verified_data(tmp_path: Path):
    store = TDIDVersionStore(tmp_path / "store")
    builder = TDIDDatasetBuilder([_record(i) for i in range(3)])
    split = builder.split(seed=3)

    version = store.save_version("verified-v1", builder, status="final", split=split)
    loaded = store.load_version(version.version_id)
    assert loaded.records == builder.records
    assert version.split_manifest["counts"] == {
        "train": len(split.train), "validation": len(split.validation), "test": len(split.test)
    }

    Path(version.snapshot_path if Path(version.snapshot_path).is_absolute() else store.storage_dir / version.snapshot_path).write_text("tampered\n", encoding="utf-8")
    with pytest.raises(TDIDError, match="hash"):
        store.load_version(version.version_id)

    with pytest.raises(TDIDError, match="only verified"):
        store.save_version("draft-with-unverified", TDIDDatasetBuilder([_record(9, verified=False)]), status="final")

