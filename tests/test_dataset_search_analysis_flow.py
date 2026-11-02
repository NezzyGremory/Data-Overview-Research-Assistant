from types import SimpleNamespace

import app.ui.main_window as main_window_module
from app.ui.dataset_search_page import DatasetSearchPage


def test_downloaded_dataset_analysis_waits_for_lazy_page(monkeypatch):
    scheduled = []
    shown_pages = []
    analysis_calls = []

    class TimerStub:
        @staticmethod
        def singleShot(delay, callback):
            scheduled.append((delay, callback))

    monkeypatch.setattr(main_window_module, "QTimer", TimerStub)
    main_window = SimpleNamespace(
        analysis_page=None,
        _analysis_ready_callbacks=[],
        _analysis_build_scheduled=False,
        current_dataset=None,
        _show_page=shown_pages.append,
    )

    def create_analysis_page():
        main_window._analysis_build_scheduled = False
        main_window.analysis_page = SimpleNamespace(
            run_analysis=lambda: analysis_calls.append("run")
        )
        main_window_module.MainWindow._run_analysis_ready_callbacks(main_window)

    main_window._create_analysis_page = create_analysis_page
    callback = lambda page: DatasetSearchPage._run_imported_analysis(
        SimpleNamespace(), page
    )

    main_window_module.MainWindow.open_analysis_page(
        main_window, on_ready=callback
    )

    assert shown_pages == ["analysis"]
    assert len(scheduled) == 1
    build_delay, build_page = scheduled.pop()
    assert build_delay == 40
    build_page()

    assert len(scheduled) == 1
    callback_delay, run_callback = scheduled.pop()
    assert callback_delay == 0
    run_callback()
    assert analysis_calls == ["run"]
