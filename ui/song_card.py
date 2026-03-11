"""
Song Card module for Just Dance UI.
Creates interactive card widgets for displaying songs with hover effects.
"""

import logging
from pathlib import Path
from typing import Optional, Callable

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QGraphicsDropShadowEffect, QSizePolicy)
from PyQt6.QtCore import (Qt, QSize, QTimer, pyqtSignal, QPropertyAnimation,
                          QEasingCurve, QRectF, QParallelAnimationGroup)
from PyQt6.QtGui import QFont, QPixmap, QColor, QPainter, QBrush, QPen

import constants
from media_manager import SongInfo, get_media_manager


class RoundedLabel(QLabel):
    """
    Custom label with rounded corners for displaying thumbnails.
    """

    def __init__(self, radius: float = constants.BORDER_RADIUS_LG,
                 parent: Optional[QWidget] = None) -> None:
        """
        Initialize the rounded label.

        Args:
            radius: Corner radius in pixels
            parent: Parent widget
        """
        super().__init__(parent)
        self._radius = radius
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setScaledContents(False)

    def set_radius(self, radius: float) -> None:
        """Set the corner radius."""
        self._radius = radius
        self.update()

    def paintEvent(self, event) -> None:
        """Override paint event to draw rounded rectangle."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Create rounded rectangle path
        rect = QRectF(self.rect())
        path = self._rounded_rect(rect, self._radius)

        # Clip to rounded rectangle
        painter.setClipPath(path)

        # Draw the pixmap
        if self.pixmap():
            scaled_pixmap = self.pixmap().scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            x_offset = (scaled_pixmap.width() - self.width()) // 2
            y_offset = (scaled_pixmap.height() - self.height()) // 2
            painter.drawPixmap(-x_offset, -y_offset, scaled_pixmap)

    def _rounded_rect(self, rect: QRectF, radius: float) -> 'QPainterPath':
        """Create a rounded rectangle path."""
        from PyQt6.QtGui import QPainterPath
        path = QPainterPath()
        path.addRoundedRect(rect, radius, radius)
        return path


class SongCard(QWidget):
    """
    Interactive card widget for displaying a song.
    Features thumbnail, title, and hover effects for audio preview.
    """

    # Signals
    clicked = pyqtSignal(str)  # Emits song name when clicked
    hovered = pyqtSignal(str)  # Emits song name when hovered
    unhovered = pyqtSignal()   # Emits when mouse leaves

    def __init__(self, song_info: SongInfo,
                 parent: Optional[QWidget] = None) -> None:
        """
        Initialize the song card.

        Args:
            song_info: SongInfo object containing song data
            parent: Parent widget
        """
        super().__init__(parent)
        self._logger = logging.getLogger(self.__class__.__name__)

        self._song_info = song_info
        self._media_manager = get_media_manager()
        self._is_hovered = False

        self._setup_ui()
        self._setup_animations()
        self._setup_shadow()

        # Enable mouse tracking
        self.setMouseTracking(True)

    def _setup_ui(self) -> None:
        """Set up the user interface components."""
        # Main container
        self.setFixedWidth(constants.CARD_MIN_WIDTH)
        height = int(constants.CARD_MIN_WIDTH / constants.CARD_ASPECT_RATIO) + 60
        self.setMinimumHeight(height)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

        # Layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(constants.SPACING_SM)

        # Thumbnail container with rounded corners
        self._thumbnail_container = QWidget()
        self._thumbnail_container.setFixedSize(
        constants.CARD_MIN_WIDTH,
        int(constants.CARD_MIN_WIDTH / constants.CARD_ASPECT_RATIO))
        self._thumbnail_container.setStyleSheet(f"""
            QWidget {{
                background-color: {constants.THEME_LIGHT_GRAY};
                border-radius: {constants.BORDER_RADIUS_LG}px;
            }}
        """)

        # Thumbnail label
        self._thumbnail_label = RoundedLabel(
            radius=constants.BORDER_RADIUS_LG,
            parent=self._thumbnail_container
        )
        self._thumbnail_label.setFixedSize(
            constants.CARD_MIN_WIDTH - 4,
            int(constants.CARD_MIN_WIDTH // constants.CARD_ASPECT_RATIO - 4)
        )
        self._thumbnail_label.move(2, 2)

        # Set thumbnail pixmap if available
        if self._song_info.thumbnail:
            self._thumbnail_label.setPixmap(self._song_info.thumbnail)
        else:
            # Show placeholder
            self._thumbnail_label.setStyleSheet(f"""
                RoundedLabel {{
                    background-color: {constants.THEME_LIGHT_GRAY};
                    border: 2px dashed {constants.THEME_MEDIUM_GRAY};
                    border-radius: {constants.BORDER_RADIUS_LG}px;
                }}
            """)
            placeholder_label = QLabel("No Preview", self._thumbnail_label)
            placeholder_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            placeholder_label.setStyleSheet(f"""
                color: {constants.TEXT_TERTIARY};
                font-size: {constants.FONT_SIZE_CAPTION}px;
            """)
            placeholder_label.setFixedSize(self._thumbnail_label.size())
            placeholder_label.move(2, 2)

        # Audio indicator
        if self._song_info.has_audio:
            self._audio_indicator = QLabel("🎵", self._thumbnail_container)
            self._audio_indicator.setFixedSize(24, 24)
            self._audio_indicator.move(
                self._thumbnail_container.width() - 30,
                8
            )
            self._audio_indicator.setStyleSheet("""
                background: rgba(0, 0, 0, 0.5);
                border-radius: 12px;
                padding: 2px;
            """)
            self._audio_indicator.setVisible(False)

        layout.addWidget(self._thumbnail_container)

        # Song title
        self._title_label = QLabel(self._song_info.name, self)
        self._title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._title_label.setStyleSheet(f"""
            color: {constants.TEXT_PRIMARY};
            font-family: {constants.FONT_FAMILY};
            font-size: {constants.FONT_SIZE_BODY}px;
            font-weight: {constants.FONT_WEIGHT_MEDIUM};
        """)
        self._title_label.setWordWrap(True)
        self._title_label.setTextInteractionFlags(Qt.TextInteractionFlag.NoTextInteraction)

        layout.addWidget(self._title_label)

    def _setup_animations(self) -> None:
        """Set up hover animations."""
        # Scale animation for card
        self._scale_animation = QPropertyAnimation(self, b"geometry")
        self._scale_animation.setDuration(constants.CARD_TRANSITION_DURATION)
        self._scale_animation.setEasingCurve(QEasingCurve.Type.InOutQuad)

        # Shadow animation
        shadow_effect = self.graphicsEffect()
        if shadow_effect:
            self._shadow_animation = QPropertyAnimation(shadow_effect, b"blurRadius")
            self._shadow_animation.setDuration(constants.CARD_TRANSITION_DURATION)

    def _setup_shadow(self) -> None:
        """Set up drop shadow effect."""
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(12)
        shadow.setXOffset(0)
        shadow.setYOffset(4)
        shadow.setColor(QColor(0, 0, 0, 40))
        self.setGraphicsEffect(shadow)

    def enterEvent(self, event) -> None:
        """Handle mouse enter event for hover effects."""
        super().enterEvent(event)

        if not self._is_hovered:
            self._is_hovered = True

            # Animate scale up
            current_geometry = self.geometry()
            new_geometry = current_geometry.adjusted(
                -int(constants.CARD_MIN_WIDTH * (constants.CARD_HOVER_SCALE - 1) / 2),
                -int((constants.CARD_MIN_WIDTH // constants.CARD_ASPECT_RATIO + 60) *
                     (constants.CARD_HOVER_SCALE - 1) / 2),
                int(constants.CARD_MIN_WIDTH * (constants.CARD_HOVER_SCALE - 1) / 2),
                int((constants.CARD_MIN_WIDTH // constants.CARD_ASPECT_RATIO + 60) *
                    (constants.CARD_HOVER_SCALE - 1) / 2)
            )

            self._scale_animation.setStartValue(current_geometry)
            self._scale_animation.setEndValue(new_geometry)
            self._scale_animation.start()

            # Animate shadow
            shadow = self.graphicsEffect()
            if shadow:
                shadow_anim = QPropertyAnimation(shadow, b"blurRadius")
                shadow_anim.setDuration(constants.CARD_TRANSITION_DURATION)
                shadow_anim.setStartValue(12)
                shadow_anim.setEndValue(20)
                shadow_anim.start()

            # Show audio indicator
            if hasattr(self, '_audio_indicator'):
                self._audio_indicator.setVisible(True)

            # Play audio preview
            if self._song_info.has_audio and self._song_info.audio_path:
                self._media_manager.play_audio(self._song_info.audio_path)

            # Emit hovered signal
            self.hovered.emit(self._song_info.name)

    def leaveEvent(self, event) -> None:
        """Handle mouse leave event."""
        super().leaveEvent(event)

        if self._is_hovered:
            self._is_hovered = False

            # Animate scale back
            current_geometry = self.geometry()
            new_geometry = current_geometry.adjusted(
                int(constants.CARD_MIN_WIDTH * (constants.CARD_HOVER_SCALE - 1) / 2),
                int((constants.CARD_MIN_WIDTH // constants.CARD_ASPECT_RATIO + 60) *
                    (constants.CARD_HOVER_SCALE - 1) / 2),
                -int(constants.CARD_MIN_WIDTH * (constants.CARD_HOVER_SCALE - 1) / 2),
                -int((constants.CARD_MIN_WIDTH // constants.CARD_ASPECT_RATIO + 60) *
                     (constants.CARD_HOVER_SCALE - 1) / 2)
            )

            self._scale_animation.setStartValue(current_geometry)
            self._scale_animation.setEndValue(new_geometry)
            self._scale_animation.start()

            # Animate shadow back
            shadow = self.graphicsEffect()
            if shadow:
                shadow_anim = QPropertyAnimation(shadow, b"blurRadius")
                shadow_anim.setDuration(constants.CARD_TRANSITION_DURATION)
                shadow_anim.setStartValue(20)
                shadow_anim.setEndValue(12)
                shadow_anim.start()

            # Hide audio indicator
            if hasattr(self, '_audio_indicator'):
                self._audio_indicator.setVisible(False)

            # Stop audio preview
            self._media_manager.stop_audio(fade_out=True)

            # Emit unhovered signal
            self.unhovered.emit()

    def mousePressEvent(self, event) -> None:
        """Handle mouse press event."""
        if event.button() == Qt.MouseButton.LeftButton:
            # Visual feedback
            self._animate_click()

            # Emit clicked signal
            self.clicked.emit(self._song_info.name)

        super().mousePressEvent(event)

    def _animate_click(self) -> None:
        """Animate click feedback."""
        # Brief scale down then restore
        animation = QPropertyAnimation(self, b"windowOpacity")
        animation.setDuration(100)
        animation.setStartValue(1.0)
        animation.setKeyValueAt(0.5, 0.9)
        animation.setEndValue(1.0)
        animation.start()

    def get_song_info(self) -> SongInfo:
        """Get the song info object."""
        return self._song_info


class AnimatedSongCard(SongCard):
    """
    Enhanced song card with additional animations and polish.
    """

    def __init__(self, song_info: SongInfo,
                 parent: Optional[QWidget] = None) -> None:
        """Initialize the animated song card."""
        super().__init__(song_info, parent)

        # Additional styling
        self.setStyleSheet(f"""
            SongCard {{
                background-color: {constants.THEME_WHITE};
                border-radius: {constants.BORDER_RADIUS_LG}px;
            }}
            SongCard:hover {{
                background-color: {constants.THEME_OFF_WHITE};
            }}
        """)

    def _setup_shadow(self) -> None:
        """Set up enhanced shadow effect."""
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(15)
        shadow.setXOffset(0)
        shadow.setYOffset(6)
        shadow.setColor(QColor(0, 0, 0, 50))
        self.setGraphicsEffect(shadow)
