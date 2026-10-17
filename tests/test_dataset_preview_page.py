import pandas as pd
from types import SimpleNamespace
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from app.ui.dataset_preview_page import DataFrameTableModel, DatasetPreviewPage
from app.ui.upload_loaded_page import LoadedUploadPage


def _app():
    return QApplication.instance() or QApplication([])


def test_dataframe_model_shows_row_numbers_and_numeric_header_details():
    model = DataFrameTableModel(
        pd.DataFrame({"value": [2.5]}),
        include_index=True,
        row_offset=10,
    )

    assert model.rowCount() == 1
    assert model.columnCount() == 2
    assert model.data(model.index(0, 0), Qt.DisplayRole) == "11"
    assert model.data(model.index(0, 1), Qt.DisplayRole) == "2.5"
    assert model.headerData(0, Qt.Horizontal) == "#"
    assert model.headerData(1, Qt.Horizontal) == "value\n# numeric"


def test_dataframe_model_can_use_compact_single_line_headers():
    model = DataFrameTableModel(
        pd.DataFrame({"PassengerId": [1]}),
        include_index=True,
        detailed_headers=False,
    )

    assert model.headerData(0, Qt.Horizontal) == "#"
    assert model.headerData(1, Qt.Horizontal) == "PassengerId"


def test_preview_search_pagination_and_tabs_use_active_dataframe():
    _app()
    page = DatasetPreviewPage()
    dataframe = pd.DataFrame({
        "score": range(31),
        "label": ["needle" if index in (4, 28) else "other" for index in range(31)],
    })

    page.update_dataset(dataframe, "sample.csv")
    assert page.model.rowCount() == 10
    assert page.row_count_label.text() == (
        "Menampilkan 1–10 dari total 31 baris dan 2 kolom."
    )

    page.table_search.setText("needle")
    assert page.model.rowCount() == 2
    assert page.pagination_label.text() == "Menampilkan 1–2 dari 2 baris"

    page.table_search.clear()
    page.page_size_combo.setCurrentText("25")
    page._change_page(1)
    assert page.model.rowCount() == 6
    assert page.model.data(page.model.index(0, 0), Qt.DisplayRole) == "26"

    for index in range(6):
        page._select_tab(index)
        assert page.content_stack.currentIndex() == index
        assert page.tips_card.isHidden() is (index != 0)
        assert page.columns_card.isHidden() is (index != 0)


def test_correlation_table_model_marks_diagonal_and_signed_values():
    dataframe = pd.DataFrame({
        "Column": ["x", "y"],
        "x": [1.0, -0.5],
        "y": [-0.5, 1.0],
    })
    model = DataFrameTableModel(dataframe, correlation_heatmap=True)

    diagonal = model.data(model.index(0, 1), Qt.BackgroundRole)
    negative = model.data(model.index(1, 1), Qt.BackgroundRole)

    assert diagonal.color().name() == "#155eef"
    assert diagonal.color().alpha() == 255
    assert negative.color().red() > negative.color().blue()


def test_loaded_upload_page_paginates_real_dataframe_and_routes_tabs():
    _app()
    destinations = []

    class PreviewPage:
        def update_dataset(self, dataframe, filename):
            self.dataframe = dataframe
            self.filename = filename

        def _select_tab(self, index):
            self.selected_tab = index

    preview = PreviewPage()
    main_window = SimpleNamespace(
        current_file_path=None,
        dataset_preview_page=preview,
        datasets_tabs=SimpleNamespace(setCurrentWidget=lambda page: destinations.append(page)),
        _show_page=lambda key: destinations.append(key),
        open_help=lambda: None,
        open_dataset_search_page=lambda: None,
    )
    upload_page = SimpleNamespace(
        main_window=main_window,
        select_dataset=lambda *_args: None,
        _csv_paths_from_drop=lambda _event: [],
    )
    dataframe = pd.DataFrame({"id": range(26), "name": [f"row-{i}" for i in range(26)]})
    page = LoadedUploadPage(upload_page)

    page.update_dataset(dataframe, "sample.csv")
    assert page.filename_label.text() == "sample.csv"
    assert page.file_details_label.text().startswith("26 baris")
    assert page.model.rowCount() == 10
    page.page_size_combo.setCurrentText("25")
    assert page.model.rowCount() == 25
    page._go_to_page(1)
    assert page.model.rowCount() == 1
    assert page.model.data(page.model.index(0, 0), Qt.DisplayRole) == "26"

    page._open_tab(2)
    assert preview.dataframe is dataframe
    assert preview.filename == "sample.csv"
    assert preview.selected_tab == 1
    assert destinations == [preview, "dataset"]
