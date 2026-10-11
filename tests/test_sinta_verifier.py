import unittest
from unittest.mock import patch

import httpx

from app.research.sinta_verifier import SintaVerifier


def journal_card(name, issns, rank="SINTA 2", period="Masa berlaku SK akreditasi: 2021 - 2025"):
    issn_text = " | ".join(f"E-ISSN: {value}" for value in issns)
    period_text = f"<p>{period}</p>" if period else ""
    rank_text = f"<span class='badge'>{rank}</span>" if rank else ""
    return (
        "<div class='journal-card'>"
        f"<h3>{name}</h3><p>{issn_text}</p>{rank_text}{period_text}"
        "</div>"
    )


class TestSintaVerifier(unittest.TestCase):
    def setUp(self):
        self.verifier = SintaVerifier()

    def test_normalize_issn(self):
        self.assertEqual(self.verifier.normalize_issn("1234-567X"), "1234567X")
        self.assertEqual(self.verifier.normalize_issn("1234-567x"), "1234567X")
        self.assertEqual(self.verifier.normalize_issn(" 1234 5678 "), "12345678")
        self.assertEqual(self.verifier.normalize_issn(""), "")
        self.assertEqual(self.verifier.normalize_issn(None), "")

    def test_format_issn(self):
        self.assertEqual(self.verifier.format_issn("1234567X"), "1234-567X")

    def test_both_issns_are_checked_and_exact_match_beats_name_match(self):
        name_only = journal_card(
            "Jurnal Sains dan Teknologi",
            ["1111-2222"],
            "SINTA 5",
        )
        exact_second_issn = journal_card(
            "Jurnal Sains dan Teknologi",
            ["2828-4992", "2828-4984"],
            "SINTA 3",
        ).replace(
            "</h3>",
            "</h3><a href='/journals/profile/2468'>Profile</a>",
        )
        profile = (
            "<h1>Jurnal Sains dan Teknologi</h1>"
            "<p>P-ISSN: 28284992 E-ISSN: 28284984</p>"
            "<p>Sinta 3 Current Accreditation</p>"
            "<p>Masa berlaku SK akreditasi: 2021 - 2025</p>"
        )
        responses = {
            "28284984": name_only,
            "28284992": exact_second_issn,
            "Jurnal Sains dan Teknologi": name_only,
        }
        seen = []

        def lookup(query):
            seen.append(query)
            return responses[query]

        with patch.object(self.verifier, "search_sinta_directory", side_effect=lookup), patch.object(
            self.verifier, "fetch_sinta_profile", return_value=profile
        ):
            result = self.verifier.verify(
                issn_list=["2828-4984", "2828-4992"],
                journal_name="Jurnal Sains dan Teknologi",
                article_year="2023",
            )

        self.assertEqual(seen, ["28284984", "28284992"])
        self.assertEqual(result.match_method, "ISSN")
        self.assertIn(result.issn_matched, {"2828-4984", "2828-4992"})
        self.assertEqual(result.rank, "S3")
        self.assertEqual(result.status, "S3")
        self.assertEqual(result.period_status, "VALID")

    def test_rank_is_taken_only_from_the_matching_journal_record(self):
        html = journal_card("Journal One", ["1111-2222"], "SINTA 1") + journal_card(
            "Journal Two", ["3333-4444"], "SINTA 4"
        )
        result = self.verifier.parse_and_evaluate(
            html, ["3333-4444"], "Journal Two", article_year=2023
        )
        self.assertEqual(result.journal_name, "Journal Two")
        self.assertEqual(result.rank, "S4")
        self.assertEqual(result.status, "S4")

    def test_parser_recognizes_alternate_journal_result_card_markup(self):
        html = (
            "<div class='result journal-list-item'>"
            "<h3>Jurnal Rekayasa Sistem</h3>"
            "<p>P-ISSN: 1111-2222</p><span>S4 Accredited</span>"
            "</div>"
        )
        result = self.verifier.parse_and_evaluate(
            html, ["1111-2222"], "Jurnal Rekayasa Sistem", article_year=2023
        )
        self.assertTrue(result.found)
        self.assertEqual(result.rank, "S4")
        self.assertEqual(result.status, "UNVERIFIED")
        self.assertEqual(result.lookup_status, "PERIOD_UNVERIFIED")

    def test_parser_uses_exact_issn_when_directory_card_markup_is_missing(self):
        html = (
            "<div><h3>Jurnal Pertama</h3><p>P-ISSN: 1111-2222</p>"
            "<span>S2 Accredited</span></div>"
            "<div><h3>INSOLOGI: Jurnal Sains dan Teknologi</h3>"
            "<a href='/journals/profile/13574'>Lihat profil jurnal</a>"
            "<p>P-ISSN: 2828-4992 | E-ISSN: 2828-4984</p>"
            "<span>S4 Accredited</span></div>"
            "<div><h3>Jurnal Ketiga</h3><p>ISSN: 3333-4444</p>"
            "<span>S1 Accredited</span></div>"
        )
        result = self.verifier.parse_and_evaluate(
            html, ["2828-4984"], "INSOLOGI: Jurnal Sains dan Teknologi", 2026
        )
        self.assertTrue(result.found)
        self.assertEqual(result.issn_matched, "2828-4984")
        self.assertEqual(result.rank, "S4")
        self.assertEqual(result.status, "UNVERIFIED")
        self.assertEqual(result.lookup_status, "PERIOD_UNVERIFIED")
        self.assertEqual(
            result.source_url,
            "https://sinta.kemdiktisaintek.go.id/journals/profile/13574",
        )

    def test_current_accreditation_overrides_unrelated_historical_rank_mentions(self):
        html = (
            "<div class='journal-card'>"
            "<h3>INSOLOGI: Jurnal Sains dan Teknologi</h3>"
            "<p>P-ISSN: 2828-4992</p><p>E-ISSN: 2828-4984</p>"
            "<section><span>SINTA 1</span><p>History Accreditation</p></section>"
            "<section><span>SINTA 4</span><p>Current Acreditation</p></section>"
            "</div>"
        )

        result = self.verifier.parse_and_evaluate(
            html,
            ["2828-4984", "2828-4992"],
            "INSOLOGI: Jurnal Sains dan Teknologi",
            article_year=2026,
        )

        self.assertTrue(result.found)
        self.assertEqual(result.issn_matched, "2828-4984")
        self.assertEqual(result.rank, "S4")
        self.assertEqual(result.lookup_status, "PERIOD_UNVERIFIED")
        self.assertFalse(result.is_ambiguous)

    def test_conflicting_unlabeled_ranks_are_not_guessed(self):
        html = (
            "<div class='journal-card'>"
            "<h3>General Research Journal</h3><p>E-ISSN: 1234-5678</p>"
            "<span>SINTA 1</span><span>SINTA 4</span>"
            "</div>"
        )

        result = self.verifier.parse_and_evaluate(
            html, ["1234-5678"], "General Research Journal", article_year=2024
        )

        self.assertTrue(result.found)
        self.assertIsNone(result.rank)
        self.assertEqual(result.lookup_status, "AMBIGUOUS")
        self.assertTrue(result.is_ambiguous)

    def test_exact_article_doi_uses_accreditation_shown_on_official_profile(self):
        doi = "10.55123/insologi.v5i3.8104"
        search_html = (
            "<div class='journal-card'>"
            "<h3>INSOLOGI: Jurnal Sains dan Teknologi</h3>"
            "<a href='/journals/profile/13574'>Lihat profil jurnal</a>"
            "<p>P-ISSN: 2828-4992 | E-ISSN: 2828-4984</p>"
            "<span>SINTA 1</span><span>SINTA 4</span>"
            "</div>"
        )
        profile_html = (
            "<html><body><h1>Jurnal Sains dan Teknologi</h1>"
            "<p>P-ISSN : 28284992 E-ISSN : 28284984</p>"
            "<p>Sinta 4 Current Acreditation</p>"
            "<div class='publication'>Klasifikasi Kelayakan Bantuan Pendidikan "
            f"2026 DOI: {doi} Accred : Sinta 4</div>"
            "</body></html>"
        )

        with patch.object(self.verifier, "search_sinta_directory", return_value=search_html), patch.object(
            self.verifier, "fetch_sinta_profile", return_value=profile_html
        ):
            result = self.verifier.verify(
                issn_list=["2828-4984", "2828-4992"],
                journal_name="INSOLOGI Jurnal Sains dan Teknologi",
                article_year=2026,
                article_doi=doi,
            )

        self.assertEqual(result.rank, "S4")
        self.assertEqual(result.status, "S4")
        self.assertEqual(result.lookup_status, "VERIFIED")
        self.assertEqual(result.match_method, "DOI")
        self.assertEqual(result.period_status, "ARTICLE_RECORD")
        self.assertTrue(result.period_confirmed)

    def test_profile_current_rank_is_used_but_not_claimed_for_unlisted_article(self):
        doi = "10.9999/unlisted.article"
        search_html = (
            "<div class='journal-card'>"
            "<h3>INSOLOGI: Jurnal Sains dan Teknologi</h3>"
            "<a href='/journals/profile/13574'>Lihat profil jurnal</a>"
            "<p>P-ISSN: 2828-4992 | E-ISSN: 2828-4984</p>"
            "<span>SINTA 1</span><span>SINTA 4</span></div>"
        )
        profile_html = (
            "<h1>Jurnal Sains dan Teknologi</h1>"
            "<p>P-ISSN : 28284992 E-ISSN : 28284984</p>"
            "<p>Sinta 1 History Accreditation</p>"
            "<p>Sinta 4 Current Acreditation</p>"
        )

        with patch.object(self.verifier, "search_sinta_directory", return_value=search_html), patch.object(
            self.verifier, "fetch_sinta_profile", return_value=profile_html
        ):
            result = self.verifier.verify(
                issn_list=["2828-4984"],
                journal_name="INSOLOGI Jurnal Sains dan Teknologi",
                article_year=2026,
                article_doi=doi,
            )

        self.assertEqual(result.rank, "S4")
        self.assertEqual(result.status, "UNVERIFIED")
        self.assertEqual(result.lookup_status, "PERIOD_UNVERIFIED")
        self.assertNotEqual(result.match_method, "DOI")

    def test_exact_article_with_unknown_accreditation_does_not_inherit_journal_rank(self):
        doi = "10.9999/article-with-unknown-rank"
        search_html = (
            "<div class='journal-card'><h3>Journal Example</h3>"
            "<a href='/journals/profile/2468'>Profile</a>"
            "<p>P-ISSN: 1234-5678</p><span>SINTA 4</span></div>"
        )
        profile_html = (
            "<h1>Journal Example</h1><p>P-ISSN: 1234-5678</p>"
            f"<div>2025 DOI: {doi} Accred: Unknown</div>"
            "<p>Sinta 4 Current Acreditation</p>"
        )

        with patch.object(self.verifier, "search_sinta_directory", return_value=search_html), patch.object(
            self.verifier, "fetch_sinta_profile", return_value=profile_html
        ):
            result = self.verifier.verify(
                issn_list=["1234-5678"],
                journal_name="Journal Example",
                article_year=2025,
                article_doi=doi,
            )

        self.assertIsNone(result.rank)
        self.assertEqual(result.lookup_status, "RANK_UNAVAILABLE")
        self.assertEqual(result.match_method, "DOI")

    def test_search_snippet_rank_is_never_used_without_a_verified_profile_link(self):
        search_html = journal_card(
            "Journal Example", ["1234-5678"], "SINTA 1", period=""
        )
        with patch.object(
            self.verifier, "search_sinta_directory", return_value=search_html
        ):
            result = self.verifier.verify(
                issn_list=["1234-5678"],
                journal_name="Journal Example",
                article_year=2025,
                article_doi="10.9999/example.1",
            )

        self.assertIsNone(result.rank)
        self.assertEqual(result.lookup_status, "RANK_UNAVAILABLE")
        self.assertIn("tidak ditemukan", result.notes.lower())

    def test_directory_request_retries_alternate_official_route(self):
        request = httpx.Request("GET", "https://sinta.kemdiktisaintek.go.id/journals?q=1111-2222")
        blocked = httpx.Response(403, request=request)
        ok = httpx.Response(200, text="directory html", request=request)
        with patch("app.research.sinta_verifier.httpx.Client") as client_factory:
            client = client_factory.return_value
            client.get.side_effect = [blocked, ok]
            response = self.verifier.search_sinta_directory("1111-2222")

        self.assertEqual(response, "directory html")
        self.assertEqual(client.get.call_count, 2)
        self.assertIn("/journals/index/?q=1111-2222", client.get.call_args_list[1].args[0])

    def test_partial_and_multiple_name_matches_are_ambiguous(self):
        partial = self.verifier.parse_and_evaluate(
            journal_card("Jurnal Sains dan Teknologi Terapan", ["1111-2222"]),
            [],
            "Jurnal Sains dan Teknologi",
            article_year=2023,
        )
        self.assertTrue(partial.is_ambiguous)
        self.assertEqual(partial.lookup_status, "AMBIGUOUS")
        self.assertEqual(partial.status, "UNVERIFIED")

        multiple = self.verifier.parse_and_evaluate(
            journal_card("Jurnal Sains", ["1111-2222"])
            + journal_card("Jurnal Sains", ["3333-4444"], "SINTA 4"),
            [],
            "Jurnal Sains",
            article_year=2023,
        )
        self.assertTrue(multiple.is_ambiguous)
        self.assertEqual(multiple.status, "UNVERIFIED")

    def test_valid_accreditation_period_is_evaluated_for_article_year(self):
        result = self.verifier.parse_and_evaluate(
            journal_card("Jurnal Rekayasa Sistem", ["1111-2222"], "SINTA 2", "Akreditasi berlaku: 2021 - 2025"),
            ["1111-2222"],
            "Jurnal Rekayasa Sistem",
            article_year=2023.0,
        )
        self.assertEqual(result.rank, "S2")
        self.assertEqual(result.status, "S2")
        self.assertEqual(result.period_status, "VALID")
        self.assertEqual(result.checked_for_year, 2023)

    def test_unknown_or_noncovering_period_never_proves_no_accreditation(self):
        no_period = self.verifier.parse_and_evaluate(
            journal_card("Jurnal Teknologi Informasi", ["9999-8888"], period=""),
            ["9999-8888"],
            "Jurnal Teknologi Informasi",
            article_year="2023",
        )
        self.assertEqual(no_period.rank, "S2")
        self.assertEqual(no_period.status, "UNVERIFIED")
        self.assertEqual(no_period.lookup_status, "PERIOD_UNVERIFIED")
        self.assertEqual(no_period.period_status, "PERIOD_CHECK_REQUIRED")

        outside_current_period = self.verifier.parse_and_evaluate(
            journal_card("Jurnal Rekayasa Sistem", ["1111-2222"], "SINTA 2", "Akreditasi berlaku: 2021 - 2025"),
            ["1111-2222"],
            "Jurnal Rekayasa Sistem",
            article_year=2018,
        )
        self.assertEqual(outside_current_period.rank, "S2")
        self.assertEqual(outside_current_period.status, "UNVERIFIED")
        self.assertEqual(outside_current_period.period_status, "EXPIRED")
        self.assertIn("tidak membuktikan", outside_current_period.notes.lower())

    def test_network_timeout_and_unrecognized_html_are_failures(self):
        with patch.object(
            self.verifier,
            "search_sinta_directory",
            side_effect=httpx.TimeoutException("timeout"),
        ):
            timeout_result = self.verifier.verify(["1234-5678"], "Journal", 2023)
        self.assertEqual(timeout_result.lookup_status, "FAILED")
        self.assertNotEqual(timeout_result.status, "NOT_IN_SINTA")

        with patch.object(self.verifier, "search_sinta_directory", return_value="<html><body>maintenance</body></html>"):
            html_result = self.verifier.verify(["1234-5678"], "Journal", 2023)
        self.assertEqual(html_result.lookup_status, "FAILED")
        self.assertNotEqual(html_result.status, "NOT_IN_SINTA")

        with patch.object(self.verifier, "search_sinta_directory", return_value=""):
            empty_result = self.verifier.verify(["1234-5678"], "Journal", 2023)
        self.assertEqual(empty_result.lookup_status, "FAILED")

    def test_unmatched_candidate_is_not_claimed_absent_but_explicit_empty_is(self):
        with patch.object(
            self.verifier,
            "search_sinta_directory",
            return_value=journal_card("Other Journal", ["9999-8888"]),
        ):
            unmatched = self.verifier.verify(["1234-5678"], "Wanted Journal", 2023)
        self.assertEqual(unmatched.status, "UNVERIFIED")
        self.assertEqual(unmatched.lookup_status, "NO_MATCH")

        no_results = "<div class='empty'>No journals found matching your search.</div>"
        with patch.object(self.verifier, "search_sinta_directory", return_value=no_results):
            absent = self.verifier.verify(["1234-5678"], "Wanted Journal", 2023)
        self.assertEqual(absent.status, "NOT_IN_SINTA")
        self.assertEqual(absent.lookup_status, "NOT_FOUND")

    def test_year_normalization_handles_string_float_and_invalid_values(self):
        self.assertEqual(self.verifier.normalize_year("2023"), 2023)
        self.assertEqual(self.verifier.normalize_year(2023.0), 2023)
        self.assertIsNone(self.verifier.normalize_year(float("nan")))
        self.assertIsNone(self.verifier.normalize_year("unknown"))


if __name__ == "__main__":
    unittest.main()
