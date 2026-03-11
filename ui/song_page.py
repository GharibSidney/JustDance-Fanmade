"""
Song Page module for Just Dance UI.
The main page displaying all available songs in a responsive grid.
"""

import logging
import math
from typing import Optional, List

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QScrollArea,
                             QGridLayout, QLabel, QPushButton, QFrame,
                             QGraphicsDropShadowEffect, QSizePolicy)
from PyQt6.QtCore import Qt, QSize, pyqtSignal, QEvent, QPointF, QTimer
from PyQt6.QtGui import QFont, QColor, QIcon, QPainter, QPixmap, QPainterPath

import constants
from media_manager import SongInfo, get_media_manager
from song_card import SongCard, AnimatedSongCard


class HeaderWidget(QWidget):
    """
    Custom header widget with title and navigation.
    """

    settings_clicked = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        """Initialize the header widget."""
        super().__init__(parent)
        self._logger = logging.getLogger(self.__class__.__name__)

        self._setup_ui()

    def _setup_ui(self) -> None:
        """Set up the header UI."""
        self.setFixedHeight(constants.HEADER_HEIGHT)
        self.setStyleSheet(f"""
            background-color: {constants.THEME_WHITE};
            border-bottom: 1px solid {constants.THEME_GRAY};
        """)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(
            constants.GRID_PADDING,
            0,
            constants.GRID_PADDING,
            0
        )
        layout.setSpacing(constants.SPACING_MD)

        # Logo/Title section
        self._title_container = QWidget()
        title_layout = QVBoxLayout(self._title_container)
        title_layout.setContentsMargins(0, 0, 0, 0)
        title_layout.setSpacing(0)

        # Main title
        self._title_label = QLabel(constants.APP_NAME)
        title_font = QFont(constants.FONT_FAMILY, constants.FONT_SIZE_HEADING,
                          constants.FONT_WEIGHT_BOLD)
        self._title_label.setFont(title_font)
        self._title_label.setStyleSheet(f"""
            color: {constants.ACCENT_PRIMARY};
        """)

        # Subtitle
        self._subtitle_label = QLabel("Select a song to play")
        subtitle_font = QFont(constants.FONT_FAMILY, constants.FONT_SIZE_CAPTION,
                              constants.FONT_WEIGHT_NORMAL)
        self._subtitle_label.setFont(subtitle_font)
        self._subtitle_label.setStyleSheet(f"""
            color: {constants.TEXT_SECONDARY};
        """)

        title_layout.addWidget(self._title_label)
        title_layout.addWidget(self._subtitle_label)

        layout.addWidget(self._title_container, alignment=Qt.AlignmentFlag.AlignLeft)

        # Spacer
        layout.addStretch()

        # Settings button
        self._settings_button = QPushButton()
        self._settings_button.setFixedSize(44, 44)
        self._settings_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._settings_button.setToolTip("Settings")

        # Create settings icon (gear symbol)
        settings_icon = self._create_settings_icon()
        self._settings_button.setIcon(settings_icon)
        self._settings_button.setIconSize(QSize(24, 24))

        self._settings_button.setStyleSheet(f"""
            QPushButton {{
                background-color: transparent;
                border: none;
                border-radius: 22px;
                padding: 8px;
            }}
            QPushButton:hover {{
                background-color: {constants.THEME_LIGHT_GRAY};
            }}
            QPushButton:pressed {{
                background-color: {constants.THEME_GRAY};
            }}
        """)

        self._settings_button.clicked.connect(self._on_settings_clicked)

        layout.addWidget(self._settings_button, alignment=Qt.AlignmentFlag.AlignRight)

        # Add shadow
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(8)
        shadow.setXOffset(0)
        shadow.setYOffset(2)
        shadow.setColor(QColor(0, 0, 0, 30))
        self.setGraphicsEffect(shadow)

    def _create_settings_icon(self) -> QIcon:
        pixmap = QPixmap(24, 24)
        pixmap.fill(Qt.GlobalColor.transparent)

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        painter.setPen(QColor(constants.TEXT_SECONDARY))
        painter.setBrush(QColor(constants.TEXT_SECONDARY))

        cx, cy = 12, 12
        outer_r = 7
        inner_r = 5
        teeth = 8
        tooth_depth = 2

        path = QPainterPath()

        for i in range(teeth * 2):
            angle = i * math.pi / teeth
            r = outer_r + tooth_depth if i % 2 == 0 else inner_r

            x = cx + r * math.cos(angle - math.pi / 2)
            y = cy + r * math.sin(angle - math.pi / 2)

            if i == 0:
                path.moveTo(x, y)
            else:
                path.lineTo(x, y)

        path.closeSubpath()

        painter.drawPath(path)

        # center circle
        painter.drawEllipse(QPointF(cx, cy), 3, 3)

        painter.end()

        return QIcon(pixmap)

    def _on_settings_clicked(self) -> None:
        """Handle settings button click."""
        self.settings_clicked.emit()


class EmptyStateWidget(QWidget):
    """
    Widget displayed when no songs are found.
    """

    def __init__(self, message: str = None, parent: Optional[QWidget] = None) -> None:
        """Initialize the empty state widget."""
        super().__init__(parent)
        self._message = message or constants.ERROR_NO_MUSICS_FOLDER
        self._setup_ui()

    def _setup_ui(self) -> None:
        """Set up the empty state UI."""
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(constants.SPACING_MD)

        # Icon
        icon_label = QLabel("🎵")
        icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        icon_label.setStyleSheet(f"""
            font-size: 64px;
        """)

        # Title
        title_label = QLabel("No Songs Found")
        title_font = QFont(constants.FONT_FAMILY, constants.FONT_SIZE_HEADING,
                          constants.FONT_WEIGHT_BOLD)
        title_label.setFont(title_font)
        title_label.setStyleSheet(f"""
            color: {constants.TEXT_PRIMARY};
        """)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Message
        message_label = QLabel(self._message)
        message_font = QFont(constants.FONT_FAMILY, constants.FONT_SIZE_BODY,
                            constants.FONT_WEIGHT_NORMAL)
        message_label.setFont(message_font)
        message_label.setStyleSheet(f"""
            color: {constants.TEXT_SECONDARY};
        """)
        message_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        message_label.setWordWrap(True)
        message_label.setMaximumWidth(400)

        layout.addWidget(icon_label)
        layout.addWidget(title_label)
        layout.addWidget(message_label)


class SongPage(QWidget):
    """
    Main song selection page.
    Displays all available songs in a responsive grid layout.
    """

    # Signals
    song_selected = pyqtSignal(str)  # Emits song name when a song is selected
    settings_requested = pyqtSignal()  # Emits when settings button is clicked

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        """Initialize the song page."""
        super().__init__(parent)
        self._logger = logging.getLogger(self.__class__.__name__)

        self._media_manager = get_media_manager()
        self._song_cards: List[SongCard] = []

        self._setup_ui()
        self._connect_signals()
        self._load_songs()

    def _setup_ui(self) -> None:
        """Set up the page UI."""
        self.setStyleSheet(f"""
            background-color: {constants.THEME_OFF_WHITE};
        """)

        # Main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Header
        self._header = HeaderWidget()
        main_layout.addWidget(self._header)

        # Scroll area for songs
        self._scroll_area = QScrollArea()
        self._scroll_area.setWidgetResizable(True)
        self._scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._scroll_area.setStyleSheet(f"""
            QScrollArea {{
                background-color: {constants.THEME_OFF_WHITE};
                border: none;
            }}
            QScrollBar:vertical {{
                background-color: {constants.THEME_LIGHT_GRAY};
                width: {constants.SCROLLBAR_WIDTH}px;
                border-radius: {constants.SCROLLBAR_RADIUS}px;
                margin: 0px;
            }}
            QScrollBar::handle:vertical {{
                background-color: {constants.THEME_MEDIUM_GRAY};
                border-radius: {constants.SCROLLBAR_RADIUS}px;
                min-height: 50px;
            }}
            QScrollBar::handle:vertical:hover {{
                background-color: {constants.TEXT_SECONDARY};
            }}
        """)

        # Container for grid
        self._scroll_content = QWidget()
        self._grid_layout = QGridLayout(self._scroll_content)
        self._grid_layout.setSpacing(constants.GRID_SPACING)
        self._grid_layout.setContentsMargins(
            constants.GRID_PADDING,
            constants.GRID_PADDING,
            constants.GRID_PADDING,
            constants.GRID_PADDING
        )

        self._scroll_area.setWidget(self._scroll_content)
        main_layout.addWidget(self._scroll_area)

        # Empty state (initially hidden)
        self._empty_state = EmptyStateWidget()
        self._empty_state.setVisible(False)
        main_layout.addWidget(self._empty_state)

    def _connect_signals(self) -> None:
        """Connect signals and slots."""
        # Header signals
        self._header.settings_clicked.connect(self.settings_requested)

        # Media manager signals
        self._media_manager.scan_completed.connect(self._on_scan_completed)
        self._media_manager.scan_error.connect(self._on_scan_error)

    def _load_songs(self) -> None:
        """Load songs from the musics folder."""
        self._media_manager.scan_music_folder()

    def _on_scan_completed(self, songs: List[SongInfo]) -> None:
        """Handle scan completion."""
        self._populate_grid(songs)

    def _on_scan_error(self, error: str) -> None:
        """Handle scan error."""
        self._logger.error(f"Scan error: {error}")
        self._show_empty_state(error)

    def _populate_grid(self, songs: List[SongInfo]) -> None:
        """Populate the grid with song cards."""
        # Clear existing cards
        self._clear_grid()

        if not songs:
            self._show_empty_state()
            return

        self._empty_state.setVisible(False)
        self._scroll_area.setVisible(True)

        # Calculate optimal number of columns based on width
        screen_width = self._scroll_area.viewport().width() if self._scroll_area.viewport() else 1200

        # Calculate available width for cards
        available_width = screen_width - (2 * constants.GRID_PADDING)
        card_with_spacing = constants.CARD_MIN_WIDTH + constants.GRID_SPACING

        num_columns = max(constants.GRID_MIN_COLUMNS,
                         min(constants.GRID_MAX_COLUMNS,
                             available_width // card_with_spacing))

        # Create song cards
        for index, song_info in enumerate(songs):
            row = index // num_columns
            column = index % num_columns

            # Create card
            card = AnimatedSongCard(song_info)
            card.clicked.connect(self._on_song_clicked)

            # Store reference
            self._song_cards.append(card)

            # Add to grid
            self._grid_layout.addWidget(card, row, column)

        # Update grid alignment
        self._grid_layout.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter)

    def _clear_grid(self) -> None:
        """Clear all song cards from the grid."""
        while self._grid_layout.count():
            item = self._grid_layout.takeAt(0)
            if item and item.widget():
                item.widget().deleteLater()

        self._song_cards.clear()

    def _show_empty_state(self, message: str = None) -> None:
        """Show the empty state widget."""
        self._scroll_area.setVisible(False)
        self._empty_state.setVisible(True)

        if message:
            # Update the message in empty state
            for child in self._empty_state.findChildren(QLabel):
                if child.wordWrap():
                    child.setText(message)

    def _on_song_clicked(self, song_name: str) -> None:
        """Handle song card click."""
        self._logger.info(f"Song selected: {song_name}")
        self.song_selected.emit(song_name)

    def refresh(self) -> None:
        """Refresh the song list."""
        self._load_songs()

    def resizeEvent(self, event) -> None:
        """Handle resize event for responsive grid."""
        super().resizeEvent(event)

        # Recalculate columns on resize
        if self._song_cards:
            QTimer.singleShot(50, self._recalculate_columns)

    def _recalculate_columns(self) -> None:
        """Recalculate and adjust grid columns."""
        if not self._song_cards:
            return

        songs = [card.get_song_info() for card in self._song_cards]
        self._populate_grid(songs)
