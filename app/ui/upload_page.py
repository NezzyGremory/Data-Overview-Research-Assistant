from PySide6.QtCore import Qt
from pathlib import Path

from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QFileDialog,
    QMessageBox,
    QFrame,
)


class UploadPage(QWidget):
    def __init__(self, main_window):
        super().__init__()

        self.main_window = main_window

        self.build_ui()

    def build_ui(self):

        layout = QVBoxLayout(self)

        layout.setContentsMargins(
            35, 30, 35, 30
        )

        layout.setSpacing(20)

        # =================================================
        # HEADER
        # =================================================

        title = QLabel(
            "Upload Dataset"
        )

        title.setObjectName(
            "pageTitle"
        )

        subtitle = QLabel(
            "Upload a CSV dataset to begin "
            "your academic research analysis."
        )

        subtitle.setObjectName(
            "pageSubtitle"
        )

        subtitle.setWordWrap(True)

        layout.addWidget(title)
        layout.addWidget(subtitle)

        # =================================================
        # UPLOAD CARD
        # =================================================

        upload_card = QFrame()

        upload_card.setObjectName(
            "uploadCard"
        )

        upload_layout = QVBoxLayout(
            upload_card
        )

        upload_layout.setContentsMargins(
            40, 40, 40, 40
        )

        upload_layout.setSpacing(15)

        icon_label = QLabel(
            "CSV DATASET"
        )

        icon_label.setObjectName(
            "uploadIcon"
        )

        icon_label.setAlignment(
            Qt.AlignCenter
        )

        upload_layout.addWidget(
            icon_label
        )

        upload_title = QLabel(
            "Choose your dataset"
        )

        upload_title.setObjectName(
            "cardTitle"
        )

        upload_title.setAlignment(
            Qt.AlignCenter
        )

        upload_layout.addWidget(
            upload_title
        )

        upload_description = QLabel(
            "Supported format: CSV\n"
            "The original dataset will never be modified."
        )

        upload_description.setObjectName(
            "cardDescription"
        )

        upload_description.setAlignment(
            Qt.AlignCenter
        )

        upload_layout.addWidget(
            upload_description
        )

        button_layout = QHBoxLayout()

        button_layout.addStretch()

        self.upload_button = QPushButton(
            "Select CSV Dataset"
        )

        self.upload_button.setObjectName(
            "primaryButton"
        )

        self.upload_button.setMinimumHeight(
            45
        )

        self.upload_button.setMinimumWidth(
            200
        )

        self.upload_button.clicked.connect(
            self.select_dataset
        )

        button_layout.addWidget(
            self.upload_button
        )

        button_layout.addStretch()

        upload_layout.addLayout(
            button_layout
        )

        layout.addWidget(
            upload_card
        )

        # =================================================
        # FILE INFORMATION
        # =================================================

        info_card = QFrame()

        info_card.setObjectName(
            "contentCard"
        )

        info_layout = QVBoxLayout(
            info_card
        )

        info_layout.setContentsMargins(
            22, 20, 22, 20
        )

        info_title = QLabel(
            "Dataset Information"
        )

        info_title.setObjectName(
            "sectionTitle"
        )

        self.info_label = QLabel(
            "No dataset selected."
        )

        self.info_label.setObjectName(
            "cardDescription"
        )

        self.info_label.setWordWrap(
            True
        )

        info_layout.addWidget(
            info_title
        )

        info_layout.addWidget(
            self.info_label
        )

        layout.addWidget(
            info_card
        )

        layout.addStretch()

    # =====================================================
    # SELECT DATASET
    # =====================================================

    def select_dataset(self):

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Dataset",
            "",
            "Datasets (*.csv *.xlsx *.xls *.json *.parquet);;CSV Files (*.csv);;Excel Files (*.xlsx *.xls);;JSON Files (*.json);;Parquet Files (*.parquet)"
        )

        if not file_path:
            return

        try:

            dataframe = (
                self.main_window.loader.load(
                    file_path
                )
            )

            self.main_window.current_dataset = (
                dataframe
            )

            self.main_window.current_file_path = (
                file_path
            )

            filename = Path(
                file_path
            ).name

            file_size = (
                Path(file_path).stat().st_size
                / 1024
            )

            self.info_label.setText(
                f"<b>File:</b> {filename}<br>"
                f"<b>Size:</b> {file_size:.2f} KB<br>"
                f"<b>Rows:</b> {len(dataframe):,}<br>"
                f"<b>Columns:</b> "
                f"{len(dataframe.columns):,}"
            )

            self.main_window.dashboard.update_dataset(
                dataframe,
                filename
            )

            self.main_window.update_dataset_state()

            self.main_window.open_analysis_page()

        except Exception as error:

            QMessageBox.critical(
                self,
                "Upload Error",
                "Failed to load dataset.\n\n"
                f"{error}"
            )
