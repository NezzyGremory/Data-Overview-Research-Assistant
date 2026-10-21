from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QFrame,
    QScrollArea,
)


class AnalysisPage(QWidget):

    def __init__(self, main_window):
        super().__init__()

        self.main_window = main_window

        self.build_ui()

    def build_ui(self):

        root_layout = QVBoxLayout(self)

        root_layout.setContentsMargins(
            35, 30, 35, 30
        )

        root_layout.setSpacing(20)

        # =================================================
        # HEADER
        # =================================================

        header_layout = QHBoxLayout()

        title_layout = QVBoxLayout()

        title = QLabel(
            "Dataset Analysis"
        )

        title.setObjectName(
            "pageTitle"
        )

        subtitle = QLabel(
            "Understand the structure, quality, "
            "statistics, and relationships within your dataset."
        )

        subtitle.setObjectName(
            "pageSubtitle"
        )

        subtitle.setWordWrap(True)

        title_layout.addWidget(title)
        title_layout.addWidget(subtitle)

        header_layout.addLayout(
            title_layout
        )

        header_layout.addStretch()

        self.analyze_button = QPushButton(
            "Run Analysis"
        )

        self.analyze_button.setObjectName(
            "primaryButton"
        )

        self.analyze_button.setMinimumHeight(
            42
        )

        self.analyze_button.clicked.connect(
            self.run_analysis
        )

        header_layout.addWidget(
            self.analyze_button
        )

        root_layout.addLayout(
            header_layout
        )

        # =================================================
        # DATASET STATUS
        # =================================================

        self.dataset_label = QLabel(
            "No dataset loaded."
        )

        self.dataset_label.setObjectName(
            "contentCard"
        )

        root_layout.addWidget(
            self.dataset_label
        )

        # =================================================
        # RESULT AREA
        # =================================================

        scroll = QScrollArea()

        scroll.setWidgetResizable(
            True
        )

        scroll.setFrameShape(
            QFrame.NoFrame
        )

        container = QWidget()

        self.result_layout = QVBoxLayout(
            container
        )

        self.result_layout.setSpacing(
            15
        )

        scroll.setWidget(
            container
        )

        root_layout.addWidget(
            scroll
        )

        self.show_empty_state()

    # =====================================================
    # EMPTY STATE
    # =====================================================

    def show_empty_state(self):

        self.clear_results()

        label = QLabel(
            "No analysis results yet.\n\n"
            "Upload a dataset and click "
            "\"Run Analysis\"."
        )

        label.setObjectName(
            "emptyState"
        )

        label.setWordWrap(True)

        self.result_layout.addWidget(
            label
        )

        self.result_layout.addStretch()

    # =====================================================
    # CLEAR
    # =====================================================

    def clear_results(self):

        while self.result_layout.count():

            item = self.result_layout.takeAt(0)

            widget = item.widget()

            if widget:

                widget.setParent(None)
                widget.deleteLater()

    # =====================================================
    # RUN ANALYSIS
    # =====================================================

    def run_analysis(self):

        dataframe = (
            self.main_window.current_dataset
        )

        if dataframe is None :

            self.show_empty_state()

            return

        self.analyze_button.setEnabled(
            False
        )

        self.analyze_button.setText(
            "Analyzing..."
        )

        try:

            result = (
                self.main_window.run_dataset_analysis()
            )

            self.display_results(
                result
            )

        except Exception as error:

            self.clear_results()

            error_label = QLabel(
                "Analysis failed.\n\n"
                f"{error}"
            )

            error_label.setObjectName(
                "errorState"
            )

            error_label.setWordWrap(
                True
            )

            self.result_layout.addWidget(
                error_label
            )

        finally:

            self.analyze_button.setEnabled(
                True
            )

            self.analyze_button.setText(
                "Run Analysis"
            )

    # =====================================================
    # DISPLAY RESULTS
    # =====================================================

    def display_results(self, result):

        self.clear_results()

        profile = result.get(
            "profile"
        )

        statistics = result.get(
            "statistics"
        )

        missing_values = result.get(
            "missing_values"
        )

        duplicates = result.get(
            "duplicates"
        )

        outliers = result.get(
            "outliers"
        )

        correlations = result.get(
            "correlations"
        )

        fingerprint = result.get(
            "fingerprint"
        )

        # =================================================
        # OVERVIEW
        # =================================================

        self.add_section(
            "Dataset Overview",
            str(profile)
        )

        # =================================================
        # STATISTICS
        # =================================================

        self.add_section(
            "Column Statistics",
            self.format_collection(
                statistics
            )
        )

        # =================================================
        # MISSING
        # =================================================

        self.add_section(
            "Missing Values",
            self.format_collection(
                missing_values
            )
        )

        # =================================================
        # DUPLICATES
        # =================================================

        self.add_section(
            "Duplicates",
            str(duplicates)
        )

        # =================================================
        # OUTLIERS
        # =================================================

        self.add_section(
            "Outliers",
            self.format_collection(
                outliers
            )
        )

        # =================================================
        # CORRELATIONS
        # =================================================

        self.add_section(
            "Correlation Analysis",
            str(correlations)
        )

        # =================================================
        # FINGERPRINT
        # =================================================

        if fingerprint:

            fingerprint_text = (
                f"SHA-256:\n"
                f"{fingerprint.get('fingerprint', '—')}\n\n"
                f"Representation:\n"
                f"{fingerprint.get('representation', '—')}"
            )

            self.add_section(
                "Dataset Fingerprint",
                fingerprint_text
            )

        self.result_layout.addStretch()

    # =====================================================
    # SECTION
    # =====================================================

    def add_section(
        self,
        title,
        content
    ):

        card = QFrame()

        card.setObjectName(
            "contentCard"
        )

        layout = QVBoxLayout(
            card
        )

        layout.setContentsMargins(
            22, 20, 22, 20
        )

        layout.setSpacing(
            10
        )

        title_label = QLabel(
            title
        )

        title_label.setObjectName(
            "sectionTitle"
        )

        content_label = QLabel(
            str(content)
        )

        content_label.setObjectName(
            "analysisContent"
        )

        content_label.setWordWrap(
            True
        )

        layout.addWidget(
            title_label
        )

        layout.addWidget(
            content_label
        )

        self.result_layout.addWidget(
            card
        )

    # =====================================================
    # FORMAT COLLECTION
    # =====================================================

    @staticmethod
    def format_collection(
        collection
    ):

        if collection is None:
            return "No data."

        if isinstance(
            collection,
            (list, tuple)
        ):

            if not collection:
                return "No data."

            return "\n\n".join(
                str(item)
                for item in collection
            )

        return str(collection)

    # =====================================================
    # UPDATE DATASET
    # =====================================================

    def update_dataset(
        self,
        dataframe,
        filename
    ):

        self.dataset_label.setText(
            f"<b>{filename}</b> — "
            f"{len(dataframe):,} rows × "
            f"{len(dataframe.columns):,} columns"
        )

        self.show_empty_state()