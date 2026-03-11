"""
Settings Page module for Just Dance UI.
Page for configuring application settings, particularly movement prediction frames.
"""

import logging
from typing import Optional

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QPushButton, QSlider, QFrame,
                             QGraphicsDropShadowEffect)
from PyQt6.QtCore import (Qt, QSize, pyqtSignal, QPropertyAnimation,
                          QEasingCurve, QRect)
from PyQt6.QtGui import QFont, QColor, QPainter, QPainterPath, QPixmap

import constants
from settings_manager import get_settings_manager


class BackButton(QPushButton):
    """
    Custom back button with arrow icon.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        """Initialize the back button."""
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self) -> None:
        """Set up button UI."""
        self.setFixedSize(44, 44)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip("Go back")

        # Create arrow icon
        self._update_icon()

        self.setStyleSheet(f"""
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

    def _update_icon(self) -> None:
        """Update the button icon."""
        pixmap = QPixmap(24, 24)
        pixmap.fill(Qt.GlobalColor.transparent)

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        painter.setPen(QColor(constants.TEXT_PRIMARY))
        painter.setBrush(QColor(constants.TEXT_PRIMARY))
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)

        # Draw arrow (left arrow)
        from PyQt6.QtCore import QPointF
        path = QPainterPath()
        path.moveTo(14, 6)
        path.lineTo(8, 12)
        path.lineTo(14, 18)
        path.lineTo(13, 19)
        path.lineTo(5, 12)
        path.lineTo(13, 5)
        path.lineTo(14, 6)
        path.closeSubpath()

        painter.fillPath(path, QColor(constants.TEXT_PRIMARY))

        painter.end()

        from PyQt6.QtGui import QIcon
        self.setIcon(QIcon(pixmap))
        self.setIconSize(QSize(24, 24))


class LabelSlider(QWidget):
    """
    Custom slider widget with label and value display.
    """

    value_changed = pyqtSignal(int)

    def __init__(self, title: str, min_value: int, max_value: int,
                 default_value: int, parent: Optional[QWidget] = None) -> None:
        """
        Initialize the label slider.

        Args:
            title: Label title
            min_value: Minimum slider value
            max_value: Maximum slider value
            default_value: Default value
            parent: Parent widget
        """
        super().__init__(parent)
        self._title = title
        self._min_value = min_value
        self._max_value = max_value
        self._current_value = default_value

        self._setup_ui()
        self._update_value_display()

    def _setup_ui(self) -> None:
        """Set up the slider UI."""
        layout = QVBoxLayout(self)
        layout.setSpacing(constants.SPACING_SM)
        layout.setContentsMargins(0, 0, 0, 0)

        # Title row with value
        title_layout = QHBoxLayout()
        title_layout.setSpacing(constants.SPACING_MD)

        # Title label
        self._title_label = QLabel(self._title)
        title_font = QFont(constants.FONT_FAMILY, constants.FONT_SIZE_BODY,
                           constants.FONT_WEIGHT_MEDIUM)
        self._title_label.setFont(title_font)
        self._title_label.setStyleSheet(f"""
            color: {constants.TEXT_PRIMARY};
        """)

        # Value label
        self._value_label = QLabel()
        value_font = QFont(constants.FONT_FAMILY, constants.FONT_SIZE_BODY,
                          constants.FONT_WEIGHT_BOLD)
        self._value_label.setFont(value_font)
        self._value_label.setStyleSheet(f"""
            color: {constants.ACCENT_PRIMARY};
            min-width: 40px;
        """)
        self._value_label.setAlignment(Qt.AlignmentFlag.AlignRight)

        title_layout.addWidget(self._title_label)
        title_layout.addStretch()
        title_layout.addWidget(self._value_label)

        layout.addLayout(title_layout)

        # Description
        self._description_label = QLabel(self._get_description())
        desc_font = QFont(constants.FONT_FAMILY, constants.FONT_SIZE_CAPTION,
                         constants.FONT_WEIGHT_NORMAL)
        self._description_label.setFont(desc_font)
        self._description_label.setStyleSheet(f"""
            color: {constants.TEXT_SECONDARY};
        """)
        self._description_label.setWordWrap(True)

        layout.addWidget(self._description_label)

        # Slider
        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setMinimum(self._min_value)
        self._slider.setMaximum(self._max_value)
        self._slider.setValue(self._current_value)
        self._slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self._slider.setTickInterval(5)
        self._slider.setSingleStep(constants.PREDICTION_FRAMES_STEP)
        self._slider.setPageStep(5)

        self._slider.setStyleSheet(f"""
            QSlider::groove:horizontal {{
                height: {constants.SLIDER_HEIGHT}px;
                background: {constants.THEME_GRAY};
                border-radius: {constants.SLIDER_HEIGHT // 2}px;
                margin: {constants.SLIDER_RANGE_PADDING}px 0;
            }}
            QSlider::handle:horizontal {{
                width: {constants.SLIDER_HANDLE_SIZE}px;
                height: {constants.SLIDER_HANDLE_SIZE}px;
                margin: -{(constants.SLIDER_HANDLE_SIZE - constants.SLIDER_HEIGHT) // 2}px 0;
                background: {constants.ACCENT_PRIMARY};
                border-radius: {constants.SLIDER_HANDLE_SIZE // 2}px;
            }}
            QSlider::handle:horizontal:hover {{
                background: {constants.ACCENT_HOVER};
            }}
            QSlider::sub-page:horizontal {{
                background: {constants.ACCENT_PRIMARY};
                border-radius: {constants.SLIDER_HEIGHT // 2}px;
            }}
            QSlider::tick:horizontal {{
                height: 6px;
                width: 2px;
                background: {constants.THEME_MEDIUM_GRAY};
            }}
        """)

        self._slider.valueChanged.connect(self._on_value_changed)

        layout.addWidget(self._slider)

        # Min/Max labels
        range_layout = QHBoxLayout()
        range_layout.setSpacing(0)

        min_label = QLabel(str(self._min_value))
        min_label.setFont(desc_font)
        min_label.setStyleSheet(f"color: {constants.TEXT_TERTIARY};")

        max_label = QLabel(str(self._max_value))
        max_label.setFont(desc_font)
        max_label.setStyleSheet(f"color: {constants.TEXT_TERTIARY};")
        max_label.setAlignment(Qt.AlignmentFlag.AlignRight)

        range_layout.addWidget(min_label)
        range_layout.addStretch()
        range_layout.addWidget(max_label)

        layout.addLayout(range_layout)

    def _get_description(self) -> str:
        """Get the description text based on current value."""
        return "Adjust the number of frames used for movement prediction. Higher values provide smoother predictions but may add latency."

    def _on_value_changed(self, value: int) -> None:
        """Handle slider value change."""
        self._current_value = value
        self._update_value_display()
        self._update_description()
        self.value_changed.emit(value)

    def _update_value_display(self) -> None:
        """Update the value label."""
        self._value_label.setText(str(self._current_value))

    def _update_description(self) -> None:
        """Update the description based on current value."""
        self._description_label.setText(self._get_description())

    def setValue(self, value: int) -> None:
        """Set the slider value."""
        value = max(self._min_value, min(self._max_value, value))
        self._slider.setValue(value)
        self._current_value = value
        self._update_value_display()

    def value(self) -> int:
        """Get the current value."""
        return self._current_value


class SettingsPage(QWidget):
    """
    Settings page for configuring application settings.
    """

    # Signals
    back_requested = pyqtSignal()  # Emits when back button is clicked

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        """Initialize the settings page."""
        super().__init__(parent)
        self._logger = logging.getLogger(self.__class__.__name__)

        self._settings_manager = get_settings_manager()

        self._setup_ui()
        self._load_settings()

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
        self._header = QWidget()
        self._header.setFixedHeight(constants.HEADER_HEIGHT)
        self._header.setStyleSheet(f"""
            background-color: {constants.THEME_WHITE};
            border-bottom: 1px solid {constants.THEME_GRAY};
        """)

        header_layout = QHBoxLayout(self._header)
        header_layout.setContentsMargins(
            constants.GRID_PADDING,
            0,
            constants.GRID_PADDING,
            0
        )
        header_layout.setSpacing(constants.SPACING_MD)

        # Back button
        self._back_button = BackButton()
        self._back_button.clicked.connect(self.back_requested)

        # Title
        title_label = QLabel("Settings")
        title_font = QFont(constants.FONT_FAMILY, constants.FONT_SIZE_HEADING,
                          constants.FONT_WEIGHT_BOLD)
        title_label.setFont(title_font)
        title_label.setStyleSheet(f"""
            color: {constants.TEXT_PRIMARY};
        """)

        header_layout.addWidget(self._back_button)
        header_layout.addWidget(title_label)

        # Add shadow to header
        shadow = QGraphicsDropShadowEffect(self._header)
        shadow.setBlurRadius(8)
        shadow.setXOffset(0)
        shadow.setYOffset(2)
        shadow.setColor(QColor(0, 0, 0, 30))
        self._header.setGraphicsEffect(shadow)

        main_layout.addWidget(self._header)

        # Content area
        content_layout = QVBoxLayout()
        content_layout.setContentsMargins(
            constants.GRID_PADDING,
            constants.GRID_PADDING,
            constants.GRID_PADDING,
            constants.GRID_PADDING
        )
        content_layout.setSpacing(constants.SPACING_LG)
        content_layout.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter)

        # Settings card container
        settings_container = QWidget()
        settings_container.setMaximumWidth(600)
        settings_layout = QVBoxLayout(settings_container)
        settings_layout.setSpacing(constants.SPACING_LG)

        # Prediction frames setting
        prediction_card = self._create_settings_card()
        self._prediction_slider = LabelSlider(
            title="Movement Prediction Frames",
            min_value=constants.MIN_PREDICTION_FRAMES,
            max_value=constants.MAX_PREDICTION_FRAMES,
            default_value=self._settings_manager.get_prediction_frames()
        )
        self._prediction_slider.value_changed.connect(self._on_prediction_frames_changed)

        card_layout = prediction_card.findChild(QVBoxLayout)
        if card_layout:
            card_layout.addWidget(self._prediction_slider)

        settings_layout.addWidget(prediction_card)

        # About section
        about_card = self._create_about_card()
        settings_layout.addWidget(about_card)

        # Add stretch to push content up
        settings_layout.addStretch()

        # Scroll area wrapper
        from PyQt6.QtWidgets import QScrollArea
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll_area.setStyleSheet(f"""
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
        """)
        scroll_area.setWidget(settings_container)

        content_layout.addWidget(scroll_area)
        main_layout.addLayout(content_layout)

    def _create_settings_card(self) -> QFrame:
        """Create a settings card frame."""
        card = QFrame()
        card.setStyleSheet(f"""
            QFrame {{
                background-color: {constants.THEME_WHITE};
                border-radius: {constants.BORDER_RADIUS_LG}px;
            }}
        """)

        # Add shadow
        shadow = QGraphicsDropShadowEffect(card)
        shadow.setBlurRadius(12)
        shadow.setXOffset(0)
        shadow.setYOffset(4)
        shadow.setColor(QColor(0, 0, 0, 40))
        card.setGraphicsEffect(shadow)

        layout = QVBoxLayout(card)
        layout.setContentsMargins(
            constants.SPACING_LG,
            constants.SPACING_LG,
            constants.SPACING_LG,
            constants.SPACING_LG
        )
        layout.setSpacing(constants.SPACING_MD)

        # Card title
        title_label = QLabel("Movement Prediction")
        title_font = QFont(constants.FONT_FAMILY, constants.FONT_SIZE_SUBHEADING,
                          constants.FONT_WEIGHT_BOLD)
        title_label.setFont(title_font)
        title_label.setStyleSheet(f"""
            color: {constants.TEXT_PRIMARY};
            margin-bottom: {constants.SPACING_XS}px;
        """)

        layout.addWidget(title_label)

        return card

    def _create_about_card(self) -> QFrame:
        """Create an about card."""
        card = QFrame()
        card.setStyleSheet(f"""
            QFrame {{
                background-color: {constants.THEME_WHITE};
                border-radius: {constants.BORDER_RADIUS_LG}px;
            }}
        """)

        # Add shadow
        shadow = QGraphicsDropShadowEffect(card)
        shadow.setBlurRadius(12)
        shadow.setXOffset(0)
        shadow.setYOffset(4)
        shadow.setColor(QColor(0, 0, 0, 40))
        card.setGraphicsEffect(shadow)

        layout = QVBoxLayout(card)
        layout.setContentsMargins(
            constants.SPACING_LG,
            constants.SPACING_LG,
            constants.SPACING_LG,
            constants.SPACING_LG
        )
        layout.setSpacing(constants.SPACING_MD)

        # Card title
        title_label = QLabel("About")
        title_font = QFont(constants.FONT_FAMILY, constants.FONT_SIZE_SUBHEADING,
                          constants.FONT_WEIGHT_BOLD)
        title_label.setFont(title_font)
        title_label.setStyleSheet(f"""
            color: {constants.TEXT_PRIMARY};
        """)

        # Version info
        version_label = QLabel(f"Version {constants.APP_VERSION}")
        version_font = QFont(constants.FONT_FAMILY, constants.FONT_SIZE_BODY,
                            constants.FONT_WEIGHT_NORMAL)
        version_label.setFont(version_font)
        version_label.setStyleSheet(f"""
            color: {constants.TEXT_SECONDARY};
        """)

        # Description
        desc_label = QLabel(
            "Just Dance UI - A motion-based rhythm game interface. "
            "Select songs, customize settings, and get ready to dance!"
        )
        desc_font = QFont(constants.FONT_FAMILY, constants.FONT_SIZE_CAPTION,
                         constants.FONT_WEIGHT_NORMAL)
        desc_label.setFont(desc_font)
        desc_label.setStyleSheet(f"""
            color: {constants.TEXT_SECONDARY};
        """)
        desc_label.setWordWrap(True)

        layout.addWidget(title_label)
        layout.addWidget(version_label)
        layout.addWidget(desc_label)

        return card

    def _load_settings(self) -> None:
        """Load current settings."""
        prediction_frames = self._settings_manager.get_prediction_frames()
        self._prediction_slider.setValue(prediction_frames)

    def _on_prediction_frames_changed(self, value: int) -> None:
        """Handle prediction frames value change."""
        self._logger.info(f"Prediction frames changed to: {value}")
        self._settings_manager.set_prediction_frames(value)
