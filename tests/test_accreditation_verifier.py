import sqlite3
import unittest
from pathlib import Path

from app.research.accreditation_verifier import (
    AccreditationVerificationResult,
    AccreditationVerifier,
)
from app.research.deduplication import PaperDeduplicator
from app.research.paper import Paper
from app.storage.database import Database
from app.storage.project_repository import ProjectRepository


class TestAccreditationVerifier(unittest.TestCase):
    """Unit tests untuk AccreditationVerifier non-SINTA."""

    def setUp(self):
        self.verifier = AccreditationVerifier()

    def test_normalize_and_format_issn(self):
        """Uji normalisasi dan pemformatan ISSN."""
        self.assertEqual(AccreditationVerifier.normalize_issn("1532-4435"), "15324435")
        self.assertEqual(AccreditationVerifier.normalize_issn(" 0304-405x "), "0304405X")
        self.assertEqual(AccreditationVerifier.normalize_issn("invalid"), "")
        self.assertEqual(AccreditationVerifier.normalize_issn(None), "")
        self.assertEqual(AccreditationVerifier.format_issn("15324435"), "1532-4435")
        self.assertEqual(AccreditationVerifier.format_issn("0304405X"), "0304-405X")

    def test_exact_issn_matching_hkdir_level2(self):
        """Pencocokan persis ISSN pada direktori resmi HK-dir Level 2."""
        result = self.verifier.verify(
            issn_list=["1532-4435"],  # JMLR
            journal_name="Journal of Machine Learning Research",
            article_year=2021,
        )
        self.assertTrue(result.found)
        self.assertEqual(result.system, "HKDIR")
        self.assertEqual(result.rank, "Level 2")
        self.assertEqual(result.status, "ACCREDITED")
        self.assertEqual(result.lookup_status, "VERIFIED")
        self.assertEqual(result.period_status, "VALID")
        self.assertEqual(result.match_method, "ISSN")
        self.assertFalse(result.is_ambiguous)
        self.assertTrue("HK-dir" in result.verification_source)

    def test_exact_issn_matching_abdc(self):
        """Pencocokan persis ISSN pada direktori resmi ABDC."""
        result = self.verifier.verify(
            issn_list=["0304-405X"],  # Journal of Financial Economics
            journal_name="Journal of Financial Economics",
            article_year=2020,
        )
        self.assertTrue(result.found)
        self.assertEqual(result.system, "ABDC")
        self.assertEqual(result.rank, "A*")
        self.assertEqual(result.status, "ACCREDITED")
        self.assertEqual(result.lookup_status, "VERIFIED")
        self.assertEqual(result.period_status, "VALID")
        self.assertEqual(result.match_method, "ISSN")
        self.assertFalse(result.is_ambiguous)

    def test_expired_accreditation_period(self):
        """Akreditasi yang sudah berakhir tidak boleh diklaim valid untuk artikel baru."""
        result = self.verifier.verify(
            issn_list=["1234-9999"],  # Period 2010 - 2021
            journal_name="Nordic Journal of Historical Studies",
            article_year=2024,  # Terbit setelah masa berlaku berakhir
        )
        self.assertTrue(result.found)
        self.assertEqual(result.period_status, "EXPIRED")
        self.assertEqual(result.status, "UNVERIFIED")
        self.assertEqual(result.lookup_status, "PERIOD_UNVERIFIED")
        self.assertFalse(result.period_confirmed)
        self.assertIn("di luar periode masa berlaku", result.notes)

    def test_missing_year_requires_confirmation(self):
        """Jika tahun artikel tidak diketahui, status harus unverified dengan konfirmasi periode."""
        result = self.verifier.verify(
            issn_list=["0028-0836"],  # Nature
            journal_name="Nature",
            article_year=None,
        )
        self.assertTrue(result.found)
        self.assertEqual(result.period_status, "PERIOD_CHECK_REQUIRED")
        self.assertEqual(result.status, "UNVERIFIED")
        self.assertEqual(result.lookup_status, "PERIOD_UNVERIFIED")
        self.assertFalse(result.period_confirmed)
        self.assertIn("Tahun artikel tidak tersedia", result.notes)

    def test_ambiguous_journal_name_fallback_without_issn(self):
        """Pencocokan nama jurnal tanpa ISSN harus ditandai ambigu."""
        result = self.verifier.verify(
            issn_list=[],
            journal_name="Nature",
            article_year=2023,
        )
        self.assertTrue(result.found)
        self.assertTrue(result.is_ambiguous)
        self.assertEqual(result.status, "UNVERIFIED")
        self.assertEqual(result.lookup_status, "AMBIGUOUS")
        self.assertEqual(result.match_method, "JOURNAL_NAME")
        self.assertIn("tanpa ISSN", result.ambiguity_reason)

    def test_journal_name_mismatch_with_issn_is_ambiguous(self):
        """Jika ISSN cocok tetapi nama jurnal pada artikel bertentangan, tandai ambigu."""
        result = self.verifier.verify(
            issn_list=["0028-0836"],  # Nature ISSN
            journal_name="Completely Unrelated Agriculture Journal",
            article_year=2022,
        )
        self.assertTrue(result.found)
        self.assertTrue(result.is_ambiguous)
        self.assertEqual(result.status, "UNVERIFIED")
        self.assertEqual(result.lookup_status, "AMBIGUOUS")
        self.assertIn("berbeda dengan catatan resmi", result.ambiguity_reason)

    def test_not_found_in_official_registries(self):
        """Jurnal yang dicari dan tidak ada di direktori yang didukung harus ditandai NOT_FOUND jujur."""
        result = self.verifier.verify(
            issn_list=["9999-8888"],
            journal_name="Unknown Local Gazette",
            article_year=2023,
        )
        self.assertFalse(result.found)
        self.assertEqual(result.status, "NOT_FOUND")
        self.assertEqual(result.lookup_status, "NOT_FOUND")
        self.assertIn("bukan bukti bahwa jurnal tidak terakreditasi oleh otoritas negara lain", result.notes)

    def test_no_metadata_handling(self):
        """Jika tidak ada ISSN dan tidak ada nama jurnal, tandai NO_METADATA."""
        result = self.verifier.verify(
            issn_list=[],
            journal_name="",
            article_year=2023,
        )
        self.assertFalse(result.found)
        self.assertEqual(result.status, "UNVERIFIED")
        self.assertEqual(result.lookup_status, "NO_METADATA")

    def test_level_0_not_accredited(self):
        """Jurnal Level 0 resmi dari HK-dir harus berstatus NOT_ACCREDITED."""
        result = self.verifier.verify(
            issn_list=["0000-1111"],
            journal_name="International Journal of Unverified Scientific Claims",
            article_year=2020,
        )
        self.assertTrue(result.found)
        self.assertEqual(result.status, "NOT_ACCREDITED")
        self.assertEqual(result.lookup_status, "VERIFIED")
        self.assertEqual(result.rank, "Level 0")

    def test_evaluate_for_year_reuses_journal_record(self):
        """Fungsi evaluate_for_year mengevaluasi tahun baru tanpa kueri ulang."""
        initial = self.verifier.verify(
            issn_list=["1234-9999"],  # Period 2010 - 2021
            journal_name="Nordic Journal of Historical Studies",
            article_year=2020,  # Valid
        )
        self.assertEqual(initial.status, "ACCREDITED")

        # Evaluasi untuk tahun di luar periode
        reevaluated = AccreditationVerifier.evaluate_for_year(initial, 2024)
        self.assertEqual(reevaluated.status, "UNVERIFIED")
        self.assertEqual(reevaluated.lookup_status, "PERIOD_UNVERIFIED")
        self.assertEqual(reevaluated.period_status, "EXPIRED")

    def test_paper_model_and_deduplication_preserves_accreditation(self):
        """Model Paper dan PaperDeduplicator mempertahankan data accreditation_screening."""
        p1 = Paper(
            title="Deep Learning Study",
            authors=["Alice"],
            doi="10.1000/1",
            issn=["1532-4435"],
            accreditation_screening={"status": "ACCREDITED", "rank": "Level 2"},
        )
        p2 = Paper(
            title="Deep Learning Study",
            authors=["Alice", "Bob"],
            doi="10.1000/1",
            issn=["1532-4435", "1533-7928"],
        )
        dedup = PaperDeduplicator()
        merged = dedup.deduplicate([p1, p2])
        self.assertEqual(len(merged), 1)
        self.assertEqual(
            merged[0].accreditation_screening.get("status"), "ACCREDITED"
        )
        self.assertEqual(merged[0].accreditation_screening.get("rank"), "Level 2")

    def test_sqlite_persistence_round_trip(self):
        """Penyimpanan dan pemuatan hasil screening non-SINTA di SQLite lokal."""
        import tempfile
        import shutil

        temp_dir = tempfile.mkdtemp()
        try:
            db_path = Path(temp_dir) / "test_accred.db"
            db = Database(db_path)
            repo = ProjectRepository(db)

            project_id = repo.create_project(
                name="Test Project",
                description="Testing non-SINTA persistence",
            )

            screening_data = {
                "system": "HKDIR",
                "system_name": "Norwegian Register (HK-dir)",
                "rank": "Level 2",
                "status": "ACCREDITED",
                "lookup_status": "VERIFIED",
                "journal_name": "Journal of Machine Learning Research",
                "issn_matched": "1532-4435",
                "period_start": 2005,
                "period_end": None,
                "period_confirmed": True,
                "checked_for_year": 2022,
                "verification_source": "HK-dir",
                "source_url": "https://kanalregisteret.hkdir.no/",
                "is_ambiguous": False,
                "notes": "Tercakup dalam masa berlaku akreditasi.",
            }

            paper_data = {
                "title": "Machine Learning Paper",
                "authors": ["Researcher A"],
                "year": 2022,
                "doi": "10.1234/test.accred",
                "venue": "Journal of Machine Learning Research",
                "issn": ["1532-4435"],
                "accreditation_screening": screening_data,
            }

            paper_id = repo.save_paper(project_id, paper_data)
            self.assertGreater(paper_id, 0)

            # Simpan juga ke tabel non_sinta_screening
            repo.save_non_sinta_screening(
                paper_key="key-test-accred",
                screening_data=screening_data,
                doi="10.1234/test.accred",
            )

            # Muat kembali dari paper
            loaded_papers = repo.get_papers(project_id)
            self.assertEqual(len(loaded_papers), 1)
            loaded_accred = loaded_papers[0].get("accreditation_screening", {})
            self.assertEqual(loaded_accred.get("status"), "ACCREDITED")
            self.assertEqual(loaded_accred.get("rank"), "Level 2")

            # Muat kembali dari non_sinta_screening
            loaded_screening = repo.get_non_sinta_screening(doi="10.1234/test.accred")
            self.assertIsNotNone(loaded_screening)
            self.assertEqual(loaded_screening.get("system"), "HKDIR")
            self.assertEqual(loaded_screening.get("rank"), "Level 2")

            # List screenings
            all_screenings = repo.list_non_sinta_screenings()
            self.assertIn("doi:10.1234/test.accred", all_screenings)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()

