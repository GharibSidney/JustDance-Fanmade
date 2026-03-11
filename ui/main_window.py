"""
Main Window module for Just Dance UI.
Handles navigation between pages and application lifecycle.
"""

import logging
import sys
from typing import Optional

from PyQt6.QtWidgets import (QMainWindow, QStackedWidget, QApplication,
                             QMessageBox, QSplashScreen)
from PyQt6.QtCore import Qt, QPropertyAnimation, QEasingCurve, QTimer, QRect
from PyQt6.QtGui import QFont, QPixmap, QColor, QPainter, QGuiApplication

import constants
from song_page import SongPage
from settings_page import SettingsPage
from settings_manager import get_settings_manager
from media_manager import get_media_manager


class MainWindow(QMainWindow):
    """
    Main application window.
    Manages page navigation and overall application state.
    """

    # Page indices
    PAGE_SONG = 0
    PAGE_SETTINGS = 1

    def __init__(self) -> None:
        """Initialize the main window."""
        super().__init__()

        self._logger = logging.getLogger(self.__class__.__name__)
        self._logger.setLevel(getattr(logging, constants.LOG_LEVEL))

        # Initialize managers
        self._settings_manager = get_settings_manager()
        self._media_manager = get_media_manager()

        # Setup logging
        self._setup_exception_handler()

        # UI setup
        self._setup_window()
        self._setup_ui()
        self._connect_signals()

        self._logger.info("Main window initialized")

    def _setup_exception_handler(self) -> None:
        """Set up global exception handler."""
        def exception_hook(exc_type, exc_value, exc_traceback):
            if issubclass(exc_type, KeyboardInterrupt):
                sys.__excepthook__(exc_type, exc_value, exc_traceback)
                return

            self._logger.critical(
                f"Uncaught exception: {exc_type.__name__}: {exc_value}",
                exc_info=(exc_type, exc_value, exc_traceback)
            )

            # Show error dialog
            error_msg = QMessageBox(self)
            error_msg.setIcon(QMessageBox.Icon.Critical)
            error_msg.setWindowTitle("Application Error")
            error_msg.setText(f"An unexpected error occurred:\n{exc_type.__name__}: {exc_value}")
            error_msg.exec()

        sys.excepthook = exception_hook

    def _setup_window(self) -> None:
        """Configure the main window properties."""
        self.setWindowTitle(constants.APP_NAME)

        # Set minimum size
        self.setMinimumSize(
            constants.WINDOW_MIN_WIDTH,
            constants.WINDOW_MIN_HEIGHT
        )

        # Set default size
        self.resize(
            constants.WINDOW_DEFAULT_WIDTH,
            constants.WINDOW_DEFAULT_HEIGHT
        )

        # Center on screen
        self._center_on_screen()

        # Set style
        self.setStyleSheet(f"""
            QMainWindow {{
                background-color: {constants.THEME_OFF_WHITE};
            }}
            QMessageBox {{
                background-color: {constants.THEME_WHITE};
            }}
        """)

    def _center_on_screen(self) -> None:
        """Center the window on the screen."""
        screen = QGuiApplication.primaryScreen()
        if screen:
            screen_geometry = screen.availableGeometry()
            window_geometry = self.frameGeometry()

            x = (screen_geometry.width() - window_geometry.width()) // 2
            y = (screen_geometry.height() - window_geometry.height()) // 2

            self.move(x, y)

    def _setup_ui(self) -> None:
        """Set up the user interface."""
        # Create stacked widget for page navigation
        self._stacked_widget = QStackedWidget()
        self.setCentralWidget(self._stacked_widget)

        # Create pages
        self._song_page = SongPage()
        self._settings_page = SettingsPage()

        # Add pages to stacked widget
        self._stacked_widget.addWidget(self._song_page)
        self._stacked_widget.addWidget(self._settings_page)

        # Set initial page
        self._stacked_widget.setCurrentIndex(self.PAGE_SONG)

    def _connect_signals(self) -> None:
        """Connect signals between components."""
        # Song page signals
        self._song_page.settings_requested.connect(self._navigate_to_settings)
        self._song_page.song_selected.connect(self._on_song_selected)

        # Settings page signals
        self._settings_page.back_requested.connect(self._navigate_to_songs)

    def _navigate_to_settings(self) -> None:
        """Navigate to the settings page with animation."""
        self._logger.info("Navigating to settings page")

        # Animate transition
        self._animate_page_transition(
            self.PAGE_SONG,
            self.PAGE_SETTINGS,
            direction='left'
        )

    def _navigate_to_songs(self) -> None:
        """Navigate back to the song page with animation."""
        self._logger.info("Navigating to song page")

        # Animate transition
        self._animate_page_transition(
            self.PAGE_SETTINGS,
            self.PAGE_SONG,
            direction='right'
        )

    def _animate_page_transition(self, from_index: int, to_index: int,
                                 direction: str = 'left') -> None:
        """
        Animate page transition.

        Args:
            from_index: Source page index
            to_index: Destination page index
            direction: Animation direction ('left' or 'right')
        """
        # Get widget sizes
        from_widget = self._stacked_widget.widget(from_index)
        to_widget = self._stacked_widget.widget(to_index)

        if not from_widget or not to_widget:
            self._stacked_widget.setCurrentIndex(to_index)
            return

        # Set end state immediately
        self._stacked_widget.setCurrentIndex(to_index)

        # Create fade animation
        opacity_animation = QPropertyAnimation(self._stacked_widget, b"windowOpacity")
        opacity_animation.setDuration(constants.NAVIGATION_ANIMATION_DURATION)
        opacity_animation.setEasingCurve(QEasingCurve.Type.InOutQuad)

        # Quick fade
        opacity_animation.setKeyValueAt(0.0, 1.0)
        opacity_animation.setKeyValueAt(0.5, 0.95)
        opacity_animation.setKeyValueAt(1.0, 1.0)

        opacity_animation.start()

    def _on_song_selected(self, song_name: str) -> None:
        """
        Handle song selection.

        Args:
            song_name: Name of the selected song
        """
        self._logger.info(f"Song selected: {song_name}")

        # TODO: Implement actual game launch
        # For now, show a confirmation
        reply = QMessageBox.question(
            self,
            "Start Game",
            f"Start playing '{song_name}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            self._logger.info(f"Launching game for: {song_name}")
            # Here you would launch the actual Just Dance game
            QMessageBox.information(
                self,
                "Launching",
                f"Starting game: {song_name}\n\n"
                f"Prediction frames: {self._settings_manager.get_prediction_frames()}"
            )

    def closeEvent(self, event) -> None:
        """Handle window close event."""
        self._logger.info("Application closing")

        # Stop any playing audio
        self._media_manager.stop_audio(fade_out=False)

        event.accept()


class SplashScreen(QSplashScreen):
    """
    Custom splash screen for application startup.
    """

    def __init__(self) -> None:
        """Initialize the splash screen."""
        pixmap = self._create_pixmap()
        super().__init__(pixmap)

    def _create_pixmap(self) -> QPixmap:
        """Create the splash screen pixmap."""
        pixmap = QPixmap(400, 300)
        pixmap.fill(QColor(constants.THEME_WHITE))

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Title
        title_font = QFont(constants.FONT_FAMILY, 36, QFont.Weight.Bold)
        painter.setFont(title_font)
        painter.setPen(QColor(constants.ACCENT_PRIMARY))

        title_rect = pixmap.rect()
        title_rect.setTop(80)
        painter.drawText(
            title_rect,
            Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop,
            constants.APP_NAME
        )

        # Subtitle
        subtitle_font = QFont(constants.FONT_FAMILY, 14)
        painter.setFont(subtitle_font)
        painter.setPen(QColor(constants.TEXT_SECONDARY))

        subtitle_rect = pixmap.rect()
        subtitle_rect.setTop(130)
        painter.drawText(
            subtitle_rect,
            Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop,
            "Loading..."
        )

        # Version
        version_font = QFont(constants.FONT_FAMILY, 10)
        painter.setFont(version_font)
        painter.setPen(QColor(constants.TEXT_TERTIARY))

        version_rect = pixmap.rect()
        version_rect.setBottom(30)
        painter.drawText(
            version_rect,
            Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignBottom,
            f"Version {constants.APP_VERSION}"
        )

        painter.end()

        return pixmap

    def show_message(self, message: str) -> None:
        self.showMessage(
            message,
            Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignHCenter,
            QColor(constants.TEXT_SECONDARY)
        )


def create_application() -> QApplication:
    """
    Create and configure the QApplication instance.

    Returns:
        Configured QApplication
    """
    # Enable high DPI scaling
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = QApplication(sys.argv)

    # Set application properties
    app.setApplicationName(constants.APP_NAME)
    app.setApplicationVersion(constants.APP_VERSION)
    app.setOrganizationName("JustDance")

    # Set font
    font = QFont(constants.FONT_FAMILY, constants.FONT_SIZE_BODY)
    app.setFont(font)

    return app


def show_splash_and_start(app: QApplication) -> None:
    splash = SplashScreen()
    splash.show()

    def start_main():
        main_window = MainWindow()
        main_window.show()
        splash.finish(main_window)

    QTimer.singleShot(100, start_main)

    app.exec()
