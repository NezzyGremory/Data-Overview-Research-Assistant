from types import SimpleNamespace
from unittest.mock import patch

from app.research.sinta_verifier import SintaVerificationResult
from app.ui.main_window import ResearchPage, SintaBatchLookupWorker


def test_batch_deduplicates_journal_lookup_normalizes_year_and_continues_after_failure():
    tasks = [
        {
            "paper_index": 4,
            "identity": "paper-a",
            "issn": ["1234-5678", "2222-3333"],
            "journal": "Journal A",
            "year": 2024.0,
        },
        {
            "paper_index": 9,
            "identity": "paper-b",
            "issn": ["2222-3333"],
            "journal": "Journal A",
            "year": "2023",
        },
        {
            "paper_index": 12,
            "identity": "paper-c",
            "issn": [],
            "journal": "Journal B",
            "year": "2022.0",
        },
    ]
    base_result = SintaVerificationResult(
        found=True,
        rank="S2",
        status="UNVERIFIED",
        lookup_status="PERIOD_UNVERIFIED",
        journal_name="Journal A",
        period_start=2020,
        period_end=2025,
        period_status="PERIOD_CHECK_REQUIRED",
        match_method="ISSN",
    )
    progress_events = []
    finished_events = []

    worker = SintaBatchLookupWorker(tasks)
    worker.progress.connect(lambda *values: progress_events.append(values))
    worker.finished.connect(finished_events.append)

    def verify_by_journal(*, issn_list, journal_name, article_year):
        if journal_name == "Journal A":
            return base_result
        raise RuntimeError("offline")

    with patch(
        "app.research.sinta_verifier.SintaVerifier.verify",
        side_effect=verify_by_journal,
    ) as verify:
        worker.run()

    assert len(verify.call_args_list) == 2
    assert len(progress_events) == 3
    assert sorted(event[2] for event in progress_events) == [4, 9, 12]
    assert sorted(event[0] for event in progress_events) == [1, 2, 3]
    progress_by_index = {event[2]: event[4] for event in progress_events}
    assert progress_by_index[4]["checked_for_year"] == 2024
    assert progress_by_index[4]["status"] == "S2"
    assert progress_by_index[9]["checked_for_year"] == 2023
    assert progress_by_index[9]["status"] == "S2"
    assert progress_by_index[12]["lookup_status"] == "FAILED"
    assert "offline" in progress_by_index[12]["notes"]
    calls_by_name = {call.kwargs["journal_name"]: call.kwargs for call in verify.call_args_list}
    assert calls_by_name["Journal A"]["article_year"] is None
    assert set(calls_by_name["Journal A"]["issn_list"]) == {"12345678", "22223333"}
    assert finished_events == [3]


def test_sinta_ui_status_labels_are_clear_and_indonesian():
    assert ResearchPage._sinta_lookup_label("FAILED") == "Gagal diperiksa; dapat dicoba ulang"
    assert ResearchPage._sinta_lookup_label("NOT_FOUND") == "Tidak ada hasil pada direktori resmi"
    assert ResearchPage._sinta_lookup_label("PERIOD_UNVERIFIED") == "Periode SK belum terverifikasi"
    assert ResearchPage._sinta_lookup_label("AMBIGUOUS") == "Kandidat jurnal perlu ditinjau"

    period_only = {
        "sinta_screening": {
            "status": "UNVERIFIED",
            "rank": "S4",
            "lookup_status": "PERIOD_UNVERIFIED",
        }
    }
    assert ResearchPage._sinta_status_label(period_only) == "SINTA 4*"
    assert ResearchPage._sinta_status_label({
        "sinta_screening": {"status": "UNVERIFIED", "lookup_status": "NO_MATCH"}
    }) == "Jurnal Belum Cocok"


def test_batch_progress_applies_rank_to_the_correct_paper_and_refreshes_ui():
    paper = {"title": "Paper test", "doi": "10.1234/test", "year": 2026}
    refreshed = []

    class Widget:
        def __init__(self):
            self.value = None
            self.text = ""

        def setValue(self, value):
            self.value = value

        def setText(self, value):
            self.text = value

    fake_page = SimpleNamespace(
        research_result={"papers": [paper]},
        _paper_identity=ResearchPage._paper_identity,
        _apply_sinta_result=ResearchPage._apply_sinta_result,
        _sinta_batch_found=0,
        _sinta_batch_manual=0,
        sinta_lookup_progress=Widget(),
        sinta_lookup_message=Widget(),
        _sinta_lookup_cache={},
        _screening_metadata_key=ResearchPage._screening_metadata_key,
        _persist_sinta_screening=lambda _paper: None,
        _render_filtered_papers=lambda: refreshed.append(True),
    )
    result = {
        "found": True,
        "status": "UNVERIFIED",
        "lookup_status": "PERIOD_UNVERIFIED",
        "rank": "S4",
        "checked_for_year": 2026,
    }

    ResearchPage._on_sinta_batch_progress(
        fake_page, 1, 1, 0, ResearchPage._paper_identity(paper), result
    )

    assert paper["sinta_screening"]["rank"] == "S4"
    assert ResearchPage._sinta_status_label(paper) == "SINTA 4*"
    assert fake_page.sinta_lookup_progress.value == 1
    assert refreshed


def test_batch_does_not_reuse_an_issn_result_for_same_name_without_issn():
    tasks = [
        {
            "paper_index": 0,
            "identity": "paper-with-issn",
            "issn": ["1111-2222"],
            "journal": "Shared Journal Name",
            "year": 2024,
        },
        {
            "paper_index": 1,
            "identity": "paper-without-issn",
            "issn": [],
            "journal": "Shared Journal Name",
            "year": 2024,
        },
    ]
    worker = SintaBatchLookupWorker(tasks)
    progress_events = []
    worker.progress.connect(lambda *values: progress_events.append(values))

    with patch(
        "app.research.sinta_verifier.SintaVerifier.verify",
        return_value=SintaVerificationResult(
            found=True,
            rank="S4",
            status="UNVERIFIED",
            lookup_status="PERIOD_UNVERIFIED",
        ),
    ) as verify:
        worker.run()

    assert verify.call_count == 2
    assert {tuple(call.kwargs["issn_list"]) for call in verify.call_args_list} == {
        ("11112222",),
        (),
    }
    assert {event[2] for event in progress_events} == {0, 1}


def test_batch_checks_article_doi_separately_for_papers_in_the_same_journal():
    tasks = [
        {
            "paper_index": 0,
            "identity": "paper-8104",
            "issn": ["2828-4984"],
            "journal": "INSOLOGI: Jurnal Sains dan Teknologi",
            "year": 2026,
            "doi": "10.55123/insologi.v5i3.8104",
        },
        {
            "paper_index": 1,
            "identity": "paper-8110",
            "issn": ["2828-4984"],
            "journal": "INSOLOGI: Jurnal Sains dan Teknologi",
            "year": 2026,
            "doi": "10.55123/insologi.v5i3.8110",
        },
    ]
    worker = SintaBatchLookupWorker(tasks)
    progress_events = []
    worker.progress.connect(lambda *values: progress_events.append(values))
    base_result_object = SintaVerificationResult(
        found=True,
        rank="S4",
        status="UNVERIFIED",
        lookup_status="PERIOD_UNVERIFIED",
        source_url="https://sinta.kemdiktisaintek.go.id/journals/profile/13574",
        all_issns=["2828-4984", "2828-4992"],
    )
    base_result = base_result_object.to_dict()

    def article_result(_verifier, _base, doi, year):
        rank = "S4" if doi.endswith("8104") else "S2"
        return SintaVerificationResult(
            found=True,
            rank=rank,
            status=rank,
            lookup_status="VERIFIED",
            match_method="DOI",
            period_status="ARTICLE_RECORD",
            period_confirmed=True,
            checked_for_year=year,
        )

    with patch(
        "app.research.sinta_verifier.SintaVerifier.verify",
        return_value=base_result_object,
    ) as verify, patch(
        "app.research.sinta_verifier.SintaVerifier.evaluate_article_doi",
        autospec=True,
        side_effect=article_result,
    ) as evaluate_article:
        worker.run()

    assert verify.call_count == 1
    assert {call.args[2] for call in evaluate_article.call_args_list} == {
        "10.55123/insologi.v5i3.8104",
        "10.55123/insologi.v5i3.8110",
    }
    results_by_index = {event[2]: event[4] for event in progress_events}
    assert results_by_index[0]["rank"] == "S4"
    assert results_by_index[1]["rank"] == "S2"


def test_screening_cache_key_is_specific_to_each_article_doi():
    base = {
        "issn": ["2828-4984"],
        "venue": "INSOLOGI: Jurnal Sains dan Teknologi",
        "year": 2026,
    }
    first = {**base, "doi": "10.55123/insologi.v5i3.8104"}
    second = {**base, "doi": "https://doi.org/10.55123/insologi.v5i3.8110"}

    first_key = ResearchPage._screening_metadata_key(first)
    second_key = ResearchPage._screening_metadata_key(second)

    assert first_key.startswith("sinta-parser:v4|")
    assert first_key != second_key
    assert "doi:10.55123/insologi.v5i3.8104" in first_key


def test_sinta_filters_include_current_rank_when_article_period_needs_confirmation():
    paper = {
        "sinta_screening": {
            "status": "UNVERIFIED",
            "rank": "S4",
            "lookup_status": "PERIOD_UNVERIFIED",
        }
    }

    assert ResearchPage._matches_sinta_filter(paper, "S4")
    assert ResearchPage._matches_sinta_filter(paper, "PERIOD_UNVERIFIED")
    assert not ResearchPage._matches_sinta_filter(paper, "S2")
    assert not ResearchPage._matches_sinta_filter(paper, "UNVERIFIED")


def test_sinta_status_labels_distinguish_lookup_outcomes():
    assert ResearchPage._sinta_status_label({
        "sinta_screening": {"status": "UNVERIFIED", "lookup_status": "NO_METADATA"}
    }) == "ISSN/Jurnal Tidak Ada"
    assert ResearchPage._sinta_status_label({
        "sinta_screening": {"status": "UNVERIFIED", "lookup_status": "RANK_UNAVAILABLE"}
    }) == "Peringkat Tidak Terbaca"
    assert ResearchPage._sinta_status_label({
        "sinta_screening": {
            "status": "S2",
            "lookup_status": "MANUALLY_CONFIRMED",
            "rank": "S2",
        }
    }) == "SINTA 2"


def test_legacy_rank_without_verification_evidence_is_scheduled_for_recheck():
    screening = ResearchPage._sinta_screening({"sinta_screening": {"status": "S4"}})

    assert screening["rank"] == "S4"
    assert screening["status"] == "UNVERIFIED"
    assert screening["lookup_status"] == "LEGACY_UNVERIFIED"
    assert ResearchPage._sinta_status_label({"sinta_screening": screening}) == "Perlu Diperiksa Ulang"
