from __future__ import annotations

from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from app.ml.task_detector import MLTaskDetector
from app.ml.method_recommender import MethodRecommender


# ==========================================================
# INFO CARD
# ==========================================================

class InfoCard(QFrame):

    def __init__(
        self,
        title: str,
        value: str = "-",
        subtitle: str = "",
        parent: QWidget | None = None,
    ):
        super().__init__(parent)

        self.setObjectName("contentCard")

        self.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Fixed,
        )

        layout = QVBoxLayout(self)

        layout.setContentsMargins(
            20,
            18,
            20,
            18,
        )

        layout.setSpacing(6)

        title_label = QLabel(
            title.upper()
        )

        title_label.setObjectName(
            "cardLabel"
        )

        self.value_label = QLabel(
            value
        )

        self.value_label.setObjectName(
            "cardValue"
        )

        self.value_label.setWordWrap(
            True
        )

        self.subtitle_label = QLabel(
            subtitle
        )

        self.subtitle_label.setObjectName(
            "cardDescription"
        )

        self.subtitle_label.setWordWrap(
            True
        )

        layout.addWidget(
            title_label
        )

        layout.addWidget(
            self.value_label
        )

        layout.addWidget(
            self.subtitle_label
        )

    def set_value(
        self,
        value: str,
    ):
        self.value_label.setText(
            value
        )

    def set_subtitle(
        self,
        text: str,
    ):
        self.subtitle_label.setText(
            text
        )


# ==========================================================
# TASK CARD
# ==========================================================

class TaskCard(QFrame):

    def __init__(
        self,
        task: dict[str, Any],
        parent: QWidget | None = None,
    ):
        super().__init__(parent)

        self.setObjectName(
            "contentCard"
        )

        layout = QVBoxLayout(
            self
        )

        layout.setContentsMargins(
            20,
            18,
            20,
            18,
        )

        layout.setSpacing(
            10
        )

        header = QHBoxLayout()

        label = (
            task.get("label")
            or task.get("task")
            or "Unknown Task"
        )

        title = QLabel(
            str(label)
        )

        title.setObjectName(
            "sectionTitle"
        )

        score = task.get(
            "confidence",
            task.get("score", 0),
        )

        confidence = QLabel(
            f"{score}% confidence"
        )

        confidence.setObjectName(
            "scoreBadge"
        )

        header.addWidget(
            title
        )

        header.addStretch()

        header.addWidget(
            confidence
        )

        layout.addLayout(
            header
        )

        target = task.get(
            "target"
        )

        if target:

            target_label = QLabel(
                f"Target: {target}"
            )

            target_label.setObjectName(
                "cardDescription"
            )

            layout.addWidget(
                target_label
            )

        status = task.get(
            "status",
            "ESTIMATION",
        )

        status_label = QLabel(
            f"Status: {status}"
        )

        status_label.setObjectName(
            "statusLabel"
        )

        layout.addWidget(
            status_label
        )

        reasons = task.get(
            "reasons",
            [],
        )

        if reasons:

            reasons_title = QLabel(
                "Why this task?"
            )

            reasons_title.setObjectName(
                "cardLabel"
            )

            layout.addWidget(
                reasons_title
            )

            for reason in reasons:

                reason_label = QLabel(
                    f"• {reason}"
                )

                reason_label.setObjectName(
                    "cardDescription"
                )

                reason_label.setWordWrap(
                    True
                )

                layout.addWidget(
                    reason_label
                )


# ==========================================================
# METHOD CARD
# ==========================================================

class MethodCard(QFrame):

    def __init__(
        self,
        method: dict[str, Any],
        parent: QWidget | None = None,
    ):
        super().__init__(parent)

        self.method = method

        self.setObjectName(
            "contentCard"
        )

        layout = QVBoxLayout(
            self
        )

        layout.setContentsMargins(
            20,
            18,
            20,
            18,
        )

        layout.setSpacing(
            8
        )

        # --------------------------------------------------
        # HEADER
        # --------------------------------------------------

        header = QHBoxLayout()

        title = QLabel(
            str(
                method.get(
                    "method",
                    "Unknown Method",
                )
            )
        )

        title.setObjectName(
            "sectionTitle"
        )

        score = method.get(
            "score",
            0,
        )

        score_label = QLabel(
            f"{score}% match"
        )

        score_label.setObjectName(
            "scoreBadge"
        )

        header.addWidget(
            title
        )

        header.addStretch()

        header.addWidget(
            score_label
        )

        layout.addLayout(
            header
        )

        # --------------------------------------------------
        # CATEGORY
        # --------------------------------------------------

        category = QLabel(
            f"Category: "
            f"{method.get('category', '-')}"
        )

        category.setObjectName(
            "cardDescription"
        )

        layout.addWidget(
            category
        )

        # --------------------------------------------------
        # DESCRIPTION
        # --------------------------------------------------

        description = QLabel(
            str(
                method.get(
                    "description",
                    "",
                )
            )
        )

        description.setObjectName(
            "cardDescription"
        )

        description.setWordWrap(
            True
        )

        layout.addWidget(
            description
        )

        # --------------------------------------------------
        # REASONS
        # --------------------------------------------------

        reasons = method.get(
            "reasons",
            [],
        )

        if reasons:

            reasons_title = QLabel(
                "Why recommended?"
            )

            reasons_title.setObjectName(
                "cardLabel"
            )

            layout.addWidget(
                reasons_title
            )

            for reason in reasons[:5]:

                reason_label = QLabel(
                    f"• {reason}"
                )

                reason_label.setObjectName(
                    "cardDescription"
                )

                reason_label.setWordWrap(
                    True
                )

                layout.addWidget(
                    reason_label
                )

        # --------------------------------------------------
        # QUICK INFO
        # --------------------------------------------------

        info_layout = QHBoxLayout()

        interpretability = method.get(
            "interpretability",
            "-",
        )

        scaling = method.get(
            "scaling_required",
            False,
        )

        nonlinear = method.get(
            "nonlinear",
            False,
        )

        info_layout.addWidget(
            QLabel(
                f"Interpretability: "
                f"{interpretability}"
            )
        )

        info_layout.addWidget(
            QLabel(
                f"Scaling: "
                f"{'Required' if scaling else 'Not required'}"
            )
        )

        info_layout.addWidget(
            QLabel(
                f"Nonlinear: "
                f"{'Yes' if nonlinear else 'No'}"
            )
        )

        info_layout.addStretch()

        layout.addLayout(
            info_layout
        )


# ==========================================================
# METHOD DETAIL CARD
# ==========================================================

class MethodDetailCard(QFrame):

    def __init__(
        self,
        method: dict[str, Any],
        parent: QWidget | None = None,
    ):
        super().__init__(parent)

        self.setObjectName(
            "contentCard"
        )

        layout = QVBoxLayout(
            self
        )

        layout.setContentsMargins(
            24,
            22,
            24,
            22,
        )

        layout.setSpacing(
            12
        )

        # --------------------------------------------------
        # TITLE
        # --------------------------------------------------

        title = QLabel(
            method.get(
                "method",
                "Method Details",
            )
        )

        title.setObjectName(
            "sectionTitle"
        )

        layout.addWidget(
            title
        )

        # --------------------------------------------------
        # DESCRIPTION
        # --------------------------------------------------

        description_title = QLabel(
            "Description"
        )

        description_title.setObjectName(
            "cardLabel"
        )

        layout.addWidget(
            description_title
        )

        description = QLabel(
            method.get(
                "description",
                "-",
            )
        )

        description.setObjectName(
            "cardDescription"
        )

        description.setWordWrap(
            True
        )

        layout.addWidget(
            description
        )

        # --------------------------------------------------
        # STRENGTHS
        # --------------------------------------------------

        strengths_title = QLabel(
            "Strengths"
        )

        strengths_title.setObjectName(
            "cardLabel"
        )

        layout.addWidget(
            strengths_title
        )

        for item in method.get(
            "strengths",
            [],
        ):

            label = QLabel(
                f"• {item}"
            )

            label.setObjectName(
                "cardDescription"
            )

            label.setWordWrap(
                True
            )

            layout.addWidget(
                label
            )

        # --------------------------------------------------
        # LIMITATIONS
        # --------------------------------------------------

        limitations_title = QLabel(
            "Limitations"
        )

        limitations_title.setObjectName(
            "cardLabel"
        )

        layout.addWidget(
            limitations_title
        )

        for item in method.get(
            "limitations",
            [],
        ):

            label = QLabel(
                f"• {item}"
            )

            label.setObjectName(
                "cardDescription"
            )

            label.setWordWrap(
                True
            )

            layout.addWidget(
                label
            )

        # --------------------------------------------------
        # PREPROCESSING
        # --------------------------------------------------

        preprocessing_title = QLabel(
            "Preprocessing"
        )

        preprocessing_title.setObjectName(
            "cardLabel"
        )

        layout.addWidget(
            preprocessing_title
        )

        for item in method.get(
            "preprocessing",
            [],
        ):

            label = QLabel(
                f"• {item}"
            )

            label.setObjectName(
                "cardDescription"
            )

            label.setWordWrap(
                True
            )

            layout.addWidget(
                label
            )

        # --------------------------------------------------
        # META
        # --------------------------------------------------

        meta = QLabel(
            "Interpretability: "
            f"{method.get('interpretability', '-')}"
            "\n"
            "Feature Scaling: "
            f"{'Required' if method.get('scaling_required') else 'Not required'}"
            "\n"
            "Nonlinear: "
            f"{'Yes' if method.get('nonlinear') else 'No'}"
            "\n"
            "Small Data: "
            f"{'Suitable' if method.get('small_data') else 'Limited'}"
            "\n"
            "Large Data: "
            f"{'Suitable' if method.get('large_data') else 'Limited'}"
        )

        meta.setObjectName(
            "cardDescription"
        )

        meta.setWordWrap(
            True
        )

        layout.addWidget(
            meta
        )


# ==========================================================
# ML INTELLIGENCE PAGE
# ==========================================================

class MLIntelligencePage(QWidget):

    """
    Halaman ML Intelligence.

    Pipeline:

        Dataset
            ↓
        Fingerprint
            ↓
        ML Task Detection
            ↓
        Method Recommendation

    Semua hasil tetap bersifat estimasi/rekomendasi.
    """

    def __init__(
        self,
        main_window,
    ):
        super().__init__()

        self.main_window = main_window

        self.detector = (
            MLTaskDetector()
        )

        self.recommender = (
            MethodRecommender()
        )

        self.last_task_result = {}
        self.last_method_result = {}

        self.build_ui()

    # ======================================================
    # UI
    # ======================================================

    def build_ui(self):

        root_layout = QVBoxLayout(
            self
        )

        root_layout.setContentsMargins(
            30,
            25,
            30,
            25,
        )

        root_layout.setSpacing(
            18
        )

        # --------------------------------------------------
        # HEADER
        # --------------------------------------------------

        header = QHBoxLayout()

        title_layout = QVBoxLayout()

        title_layout.setSpacing(
            4
        )

        title = QLabel(
            "ML Intelligence"
        )

        title.setObjectName(
            "pageTitle"
        )

        description = QLabel(
            "Deteksi task machine learning "
            "dan rekomendasi metode berdasarkan "
            "struktur serta karakteristik dataset."
        )

        description.setObjectName(
            "pageDescription"
        )

        description.setWordWrap(
            True
        )

        title_layout.addWidget(
            title
        )

        title_layout.addWidget(
            description
        )

        header.addLayout(
            title_layout
        )

        header.addStretch()

        self.refresh_button = QPushButton(
            "Analyze ML"
        )

        self.refresh_button.setObjectName(
            "primaryButton"
        )

        self.refresh_button.clicked.connect(
            self.run_detection
        )

        header.addWidget(
            self.refresh_button
        )

        root_layout.addLayout(
            header
        )

        # --------------------------------------------------
        # SCROLL
        # --------------------------------------------------

        scroll = QScrollArea()

        scroll.setWidgetResizable(
            True
        )

        scroll.setFrameShape(
            QFrame.NoFrame
        )

        self.content = QWidget()

        self.content_layout = QVBoxLayout(
            self.content
        )

        self.content_layout.setContentsMargins(
            0,
            5,
            10,
            20,
        )

        self.content_layout.setSpacing(
            16
        )

        scroll.setWidget(
            self.content
        )

        root_layout.addWidget(
            scroll
        )

        self.show_empty_state()

    # ======================================================
    # EMPTY STATE
    # ======================================================

    def show_empty_state(self):

        self.clear_content()

        card = QFrame()

        card.setObjectName(
            "contentCard"
        )

        layout = QVBoxLayout(
            card
        )

        layout.setContentsMargins(
            30,
            40,
            30,
            40,
        )

        layout.setSpacing(
            10
        )

        title = QLabel(
            "ML Intelligence belum tersedia"
        )

        title.setObjectName(
            "sectionTitle"
        )

        title.setAlignment(
            Qt.AlignCenter
        )

        description = QLabel(
            "Upload dan analisis dataset terlebih dahulu "
            "untuk mendapatkan estimasi task machine learning "
            "dan rekomendasi metode."
        )

        description.setObjectName(
            "cardDescription"
        )

        description.setAlignment(
            Qt.AlignCenter
        )

        description.setWordWrap(
            True
        )

        layout.addWidget(
            title
        )

        layout.addWidget(
            description
        )

        self.content_layout.addWidget(
            card
        )

        self.content_layout.addStretch()

    # ======================================================
    # DETECTION
    # ======================================================

    def run_detection(self):

        analysis_result = getattr(
            self.main_window,
            "analysis_result",
            {},
        )

        if not analysis_result:

            self.show_empty_state()

            return

        fingerprint = analysis_result.get(
            "fingerprint"
        )

        if not fingerprint:

            self.show_empty_state()

            return

        try:

            # ------------------------------------------------
            # ML TASK DETECTION
            # ------------------------------------------------

            result = self.detector.detect(
                fingerprint
            )

            self.last_task_result = result

            # ------------------------------------------------
            # DATASET METADATA
            # ------------------------------------------------

            dataset_info = self.extract_dataset_info(
                analysis_result,
                fingerprint,
            )

            # ------------------------------------------------
            # ADD DATASET INFORMATION
            # ------------------------------------------------

            result["dataset"] = dataset_info

            # ------------------------------------------------
            # METHOD RECOMMENDATION
            # ------------------------------------------------

            method_result = (
                self.recommender.recommend(
                    result
                )
            )

            self.last_method_result = (
                method_result
            )

            # ------------------------------------------------
            # DISPLAY
            # ------------------------------------------------

            self.display_result(
                result,
                method_result,
            )

        except Exception as exc:

            self.show_error(
                "Gagal melakukan ML Intelligence:\n"
                f"{exc}"
            )

    # ======================================================
    # DATASET INFORMATION
    # ======================================================

    def extract_dataset_info(
        self,
        analysis_result: dict[str, Any],
        fingerprint: dict[str, Any],
    ) -> dict[str, Any]:

        dataset = {}

        dataframe = getattr(
            self.main_window,
            "current_dataset",
            None,
        )

        # --------------------------------------------------
        # PRIMARY SOURCE: DATAFRAME
        # --------------------------------------------------

        if dataframe is not None:

            try:

                dataset["rows"] = int(
                    dataframe.shape[0]
                )

                dataset["columns"] = int(
                    dataframe.shape[1]
                )

                numeric_features = len(
                    dataframe.select_dtypes(
                        include="number"
                    ).columns
                )

                dataset[
                    "numeric_features"
                ] = numeric_features

            except Exception:
                pass

        # --------------------------------------------------
        # FALLBACK: FINGERPRINT
        # --------------------------------------------------

        representation = fingerprint.get(
            "representation",
            fingerprint,
        )

        characteristics = (
            representation.get(
                "characteristics",
                {},
            )
        )

        if not dataset.get("rows"):

            dataset["rows"] = int(
                representation.get(
                    "rows",
                    fingerprint.get(
                        "rows",
                        0,
                    ),
                )
                or 0
            )

        if not dataset.get("columns"):

            dataset["columns"] = int(
                representation.get(
                    "columns",
                    fingerprint.get(
                        "columns",
                        0,
                    ),
                )
                or 0
            )

        if not dataset.get(
            "numeric_features"
        ):

            dataset[
                "numeric_features"
            ] = int(
                characteristics.get(
                    "numeric_count",
                    0,
                )
                or 0
            )

        return dataset

    # ======================================================
    # DISPLAY RESULT
    # ======================================================

    def display_result(
        self,
        result: dict[str, Any],
        method_result: dict[str, Any],
    ):

        self.clear_content()

        status = result.get(
            "status",
            "ESTIMATION",
        )

        primary = result.get(
            "primary_task"
        )

        tasks = result.get(
            "tasks",
            [],
        )

        dataset = result.get(
            "dataset",
            {},
        )

        recommendations = method_result.get(
            "recommendations",
            [],
        )

        # ==================================================
        # DATASET SUMMARY
        # ==================================================

        dataset_title = QLabel(
            "Dataset Intelligence"
        )

        dataset_title.setObjectName(
            "sectionTitle"
        )

        self.content_layout.addWidget(
            dataset_title
        )

        dataset_cards = QHBoxLayout()

        dataset_cards.setSpacing(
            14
        )

        dataset_cards.addWidget(
            InfoCard(
                "Rows",
                str(
                    dataset.get(
                        "rows",
                        0,
                    )
                ),
                "Jumlah observasi dalam dataset.",
            )
        )

        dataset_cards.addWidget(
            InfoCard(
                "Columns",
                str(
                    dataset.get(
                        "columns",
                        0,
                    )
                ),
                "Jumlah kolom yang dianalisis.",
            )
        )

        dataset_cards.addWidget(
            InfoCard(
                "Numeric Features",
                str(
                    dataset.get(
                        "numeric_features",
                        0,
                    )
                ),
                "Jumlah fitur numerik.",
            )
        )

        self.content_layout.addLayout(
            dataset_cards
        )

        # ==================================================
        # PRIMARY TASK
        # ==================================================

        if primary:

            primary_label = primary.get(
                "label",
                primary.get(
                    "task",
                    "-",
                ),
            )

            confidence = primary.get(
                "confidence",
                primary.get(
                    "score",
                    0,
                ),
            )

            target = primary.get(
                "target"
            )

            cards_layout = QHBoxLayout()

            cards_layout.setSpacing(
                14
            )

            cards_layout.addWidget(
                InfoCard(
                    "Primary ML Task",
                    str(
                        primary_label
                    ),
                    f"Confidence: {confidence}%",
                )
            )

            cards_layout.addWidget(
                InfoCard(
                    "Target",
                    str(target)
                    if target
                    else "None",
                    "Target candidate yang digunakan detector.",
                )
            )

            cards_layout.addWidget(
                InfoCard(
                    "Confidence",
                    f"{confidence}%",
                    "Nilai ini merupakan estimasi heuristik.",
                )
            )

            self.content_layout.addLayout(
                cards_layout
            )

        # ==================================================
        # STATUS
        # ==================================================

        status_card = QFrame()

        status_card.setObjectName(
            "contentCard"
        )

        status_layout = QVBoxLayout(
            status_card
        )

        status_layout.setContentsMargins(
            20,
            18,
            20,
            18,
        )

        status_layout.setSpacing(
            8
        )

        status_title = QLabel(
            "Detection Status"
        )

        status_title.setObjectName(
            "cardLabel"
        )

        status_value = QLabel(
            str(status)
        )

        status_value.setObjectName(
            "statusLabel"
        )

        message = QLabel(
            result.get(
                "message",
                "Task merupakan estimasi berdasarkan dataset.",
            )
        )

        message.setObjectName(
            "cardDescription"
        )

        message.setWordWrap(
            True
        )

        status_layout.addWidget(
            status_title
        )

        status_layout.addWidget(
            status_value
        )

        status_layout.addWidget(
            message
        )

        self.content_layout.addWidget(
            status_card
        )

        # ==================================================
        # PRIMARY TASK ANALYSIS
        # ==================================================

        if primary:

            primary_section = QLabel(
                "Primary Task Analysis"
            )

            primary_section.setObjectName(
                "sectionTitle"
            )

            self.content_layout.addWidget(
                primary_section
            )

            self.content_layout.addWidget(
                TaskCard(
                    primary
                )
            )

        # ==================================================
        # METHOD RECOMMENDATION
        # ==================================================

        recommendation_title = QLabel(
            "Recommended ML Methods"
        )

        recommendation_title.setObjectName(
            "sectionTitle"
        )

        self.content_layout.addWidget(
            recommendation_title
        )

        recommendation_description = QLabel(
            "Ranking berikut merupakan recommendation score "
            "berdasarkan ML task dan karakteristik dataset. "
            "Score ini bukan accuracy atau hasil training model."
        )

        recommendation_description.setObjectName(
            "cardDescription"
        )

        recommendation_description.setWordWrap(
            True
        )

        self.content_layout.addWidget(
            recommendation_description
        )

        # --------------------------------------------------
        # METHOD CARDS
        # --------------------------------------------------

        if recommendations:

            for index, method in enumerate(
                recommendations,
                start=1,
            ):

                rank_label = QLabel(
                    f"#{index}"
                )

                rank_label.setObjectName(
                    "cardLabel"
                )

                self.content_layout.addWidget(
                    rank_label
                )

                self.content_layout.addWidget(
                    MethodCard(
                        method
                    )
                )

            # ------------------------------------------------
            # TOP METHOD DETAIL
            # ------------------------------------------------

            best_method = recommendations[0]

            detail_title = QLabel(
                "Top Recommended Method"
            )

            detail_title.setObjectName(
                "sectionTitle"
            )

            self.content_layout.addWidget(
                detail_title
            )

            self.content_layout.addWidget(
                MethodDetailCard(
                    best_method
                )
            )

        else:

            empty_methods = QLabel(
                "Tidak ada metode yang dapat direkomendasikan "
                "untuk task yang terdeteksi."
            )

            empty_methods.setObjectName(
                "cardDescription"
            )

            empty_methods.setWordWrap(
                True
            )

            self.content_layout.addWidget(
                empty_methods
            )

        # ==================================================
        # OTHER POSSIBLE TASKS
        # ==================================================

        tasks_section = QLabel(
            f"Possible ML Tasks ({len(tasks)})"
        )

        tasks_section.setObjectName(
            "sectionTitle"
        )

        self.content_layout.addWidget(
            tasks_section
        )

        if tasks:

            for task in tasks:

                self.content_layout.addWidget(
                    TaskCard(
                        task
                    )
                )

        else:

            empty = QLabel(
                "Tidak ada task ML yang berhasil dideteksi."
            )

            empty.setObjectName(
                "cardDescription"
            )

            self.content_layout.addWidget(
                empty
            )

        # ==================================================
        # DISCLAIMER
        # ==================================================

        disclaimer = QFrame()

        disclaimer.setObjectName(
            "contentCard"
        )

        disclaimer_layout = QVBoxLayout(
            disclaimer
        )

        disclaimer_layout.setContentsMargins(
            20,
            18,
            20,
            18,
        )

        disclaimer_title = QLabel(
            "Research Note"
        )

        disclaimer_title.setObjectName(
            "cardLabel"
        )

        disclaimer_text = QLabel(
            "ML task detection dan method recommendation "
            "merupakan estimasi berbasis heuristik serta "
            "knowledge base. Performa aktual metode harus "
            "divalidasi melalui eksperimen, cross-validation, "
            "dan metric evaluasi yang sesuai."
        )

        disclaimer_text.setObjectName(
            "cardDescription"
        )

        disclaimer_text.setWordWrap(
            True
        )

        disclaimer_layout.addWidget(
            disclaimer_title
        )

        disclaimer_layout.addWidget(
            disclaimer_text
        )

        self.content_layout.addWidget(
            disclaimer
        )

        self.content_layout.addStretch()

    # ======================================================
    # ERROR
    # ======================================================

    def show_error(
        self,
        message: str,
    ):

        self.clear_content()

        card = QFrame()

        card.setObjectName(
            "contentCard"
        )

        layout = QVBoxLayout(
            card
        )

        layout.setContentsMargins(
            25,
            25,
            25,
            25,
        )

        layout.setSpacing(
            10
        )

        title = QLabel(
            "ML Intelligence Error"
        )

        title.setObjectName(
            "sectionTitle"
        )

        error = QLabel(
            message
        )

        error.setObjectName(
            "cardDescription"
        )

        error.setWordWrap(
            True
        )

        layout.addWidget(
            title
        )

        layout.addWidget(
            error
        )

        self.content_layout.addWidget(
            card
        )

        self.content_layout.addStretch()

    # ======================================================
    # UPDATE DATASET
    # ======================================================

    def update_dataset(self):

        if getattr(
            self.main_window,
            "analysis_result",
            {},
        ):

            self.run_detection()

        else:

            self.show_empty_state()

    # ======================================================
    # CLEAR CONTENT
    # ======================================================

    def clear_content(self):

        while self.content_layout.count():

            item = (
                self.content_layout.takeAt(0)
            )

            widget = item.widget()

            if widget is not None:

                widget.setParent(None)
                widget.deleteLater()

            elif item.layout() is not None:

                self.clear_layout(
                    item.layout()
                )

    # ======================================================
    # CLEAR LAYOUT
    # ======================================================

    def clear_layout(
        self,
        layout,
    ):

        while layout.count():

            item = layout.takeAt(0)

            widget = item.widget()

            if widget is not None:

                widget.setParent(None)
                widget.deleteLater()

            elif item.layout() is not None:

                self.clear_layout(
                    item.layout()
                )