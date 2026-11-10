from app.ui.main_window import (
    PapersToolPage,
    _clean_visible_ui_copy,
    _normalize_visible_ui_punctuation,
)


def test_visible_ui_uses_comma_character_for_decimal_values():
    assert _normalize_visible_ui_punctuation("Rasio 1.25; rata-rata 1.5.") == (
        "Rasio 1,25; rata-rata 1,5."
    )
    assert _clean_visible_ui_copy("Nilai 1.25, kelas A; selesai.") == (
        "Nilai 1,25, kelas A; selesai."
    )


def test_research_relevance_displays_decimal_comma():
    assert PapersToolPage._format_relevance(82.5) == "82,5%"
    assert PapersToolPage._clean_ui_text("Skor 82.5%, DOI 10.1234/xyz.") == (
        "Skor 82,5%, DOI 10.1234/xyz."
    )
