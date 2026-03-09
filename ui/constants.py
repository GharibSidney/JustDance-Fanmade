"""
Constants module for Just Dance UI application.
All configuration values and magic numbers are defined here.
"""

import os
from pathlib import Path

# ============================================================================
# APPLICATION CONFIGURATION
# ============================================================================

APP_NAME = "Just Dance UI"
APP_VERSION = "1.0.0"

# ============================================================================
# WINDOW SETTINGS
# ============================================================================

WINDOW_MIN_WIDTH = 1024
WINDOW_MIN_HEIGHT = 768
WINDOW_DEFAULT_WIDTH = 1280
WINDOW_DEFAULT_HEIGHT = 800

# ============================================================================
# PATH CONFIGURATION
# ============================================================================

# Base paths - relative to the application executable/script
BASE_DIR =  "../"#Path(__file__).parent.resolve()
MUSICS_DIR = BASE_DIR / "musics"
SETTINGS_FILE = BASE_DIR / "settings.json"
CACHE_DIR = BASE_DIR / "cache"

# Subdirectory names within each song folder
VIDEO_SUBDIR = "video"
AUDIO_SUBDIR = "audio"

# ============================================================================
# THEME COLORS - WHITE THEME
# ============================================================================

# Primary colors
THEME_WHITE = "#FFFFFF"
THEME_OFF_WHITE = "#F8F9FA"
THEME_LIGHT_GRAY = "#F1F3F5"
THEME_GRAY = "#E9ECEF"
THEME_DARK_GRAY = "#DEE2E6"
THEME_MEDIUM_GRAY = "#ADB5BD"

# Text colors
TEXT_PRIMARY = "#212529"
TEXT_SECONDARY = "#6C757D"
TEXT_TERTIARY = "#ADB5BD"
TEXT_INVERSE = "#FFFFFF"

# Accent colors - Just Dance style (energetic pink/red)
ACCENT_PRIMARY = "#FF2D55"
ACCENT_HOVER = "#E6002E"
ACCENT_LIGHT = "#FF6B8A"

# Status colors
SUCCESS_COLOR = "#28A745"
WARNING_COLOR = "#FFC107"
ERROR_COLOR = "#DC3545"

# ============================================================================
# TYPOGRAPHY
# ============================================================================

FONT_FAMILY = "Segoe UI"
FONT_FAMILY_ALT = "Roboto"

# Font sizes
FONT_SIZE_TITLE = 32
FONT_SIZE_HEADING = 24
FONT_SIZE_SUBHEADING = 18
FONT_SIZE_BODY = 14
FONT_SIZE_CAPTION = 12
FONT_SIZE_SMALL = 10

# Font weights
FONT_WEIGHT_BOLD = 600
FONT_WEIGHT_MEDIUM = 500
FONT_WEIGHT_NORMAL = 400

# ============================================================================
# LAYOUT & SPACING
# ============================================================================

# Spacing constants (8px grid system)
SPACING_XXS = 4
SPACING_XS = 8
SPACING_SM = 12
SPACING_MD = 16
SPACING_LG = 24
SPACING_XL = 32
SPACING_XXL = 48

# Border radius
BORDER_RADIUS_SM = 4
BORDER_RADIUS_MD = 8
BORDER_RADIUS_LG = 12
BORDER_RADIUS_XL = 16

# Card dimensions
CARD_MIN_WIDTH = 240
CARD_MAX_WIDTH = 320
CARD_ASPECT_RATIO = 16 / 9  # 16:9 video aspect ratio

# Grid settings
GRID_MIN_COLUMNS = 2
GRID_MAX_COLUMNS = 5
GRID_SPACING = 24
GRID_PADDING = 32

# ============================================================================
# SHADOW & EFFECTS
# ============================================================================

SHADOW_COLOR = "rgba(0, 0, 0, 0.1)"
SHADOW_HOVER_COLOR = "rgba(0, 0, 0, 0.15)"
SHADOW_CARD = f"0 4px 6px {SHADOW_COLOR}"
SHADOW_CARD_HOVER = f"0 8px 16px {SHADOW_HOVER_COLOR}"

# Animation durations (milliseconds)
ANIMATION_DURATION_FAST = 150
ANIMATION_DURATION_NORMAL = 250
ANIMATION_DURATION_SLOW = 400

# ============================================================================
# MEDIA PROCESSING
# ============================================================================

# Thumbnail settings
THUMBNAIL_WIDTH = 320
THUMBNAIL_HEIGHT = 180
THUMBNAIL_QUALITY = 85
THUMBNAIL_CACHE_TTL = 3600  # Cache time-to-live in seconds

# Video processing
VIDEO_FRAME_EXTRACT_POSITION = 0.5  # Extract frame at 50% of video (middle)
VIDEO_SUPPORTED_FORMATS = [".mp4", ".avi", ".mov", ".mkv", ".webm"]

# Audio processing
AUDIO_SUPPORTED_FORMATS = [".mp3", ".wav", ".ogg", ".flac", ".m4a"]
AUDIO_FADE_DURATION = 500  # milliseconds

# ============================================================================
# SETTINGS DEFAULT VALUES
# ============================================================================

# Movement prediction settings
DEFAULT_PREDICTION_FRAMES = 15
MIN_PREDICTION_FRAMES = 5
MAX_PREDICTION_FRAMES = 30
PREDICTION_FRAMES_STEP = 1

# Settings keys
SETTING_PREDICTION_FRAMES = "prediction_frames"

# ============================================================================
# UI COMPONENT CONFIGURATION
# ============================================================================

# Header
HEADER_HEIGHT = 72
HEADER_BACKGROUND = THEME_WHITE

# Navigation
NAVIGATION_ANIMATION_DURATION = 300

# Song card
CARD_HOVER_SCALE = 1.02
CARD_TRANSITION_DURATION = 200

# Scrollbar
SCROLLBAR_WIDTH = 8
SCROLLBAR_RADIUS = 4

# Button
BUTTON_HEIGHT = 44
BUTTON_PADDING_H = 24
BUTTON_PADDING_V = 12

# Slider
SLIDER_HEIGHT = 8
SLIDER_HANDLE_SIZE = 20
SLIDER_RANGE_PADDING = 20

# ============================================================================
# ERROR HANDLING
# ============================================================================

ERROR_LOG_FILE = BASE_DIR / "error.log"
MAX_LOG_FILE_SIZE = 5 * 1024 * 1024  # 5MB

# Error messages
ERROR_NO_MUSICS_FOLDER = "No 'musics' folder found. Create a 'musics' folder in the application directory."
ERROR_INVALID_SONG_STRUCTURE = "Invalid song folder structure. Each song needs 'Video' and 'Audio' subfolders."
ERROR_VIDEO_LOAD_FAILED = "Failed to load video file"
ERROR_AUDIO_LOAD_FAILED = "Failed to load audio file"
ERROR_SETTINGS_SAVE_FAILED = "Failed to save settings"

# ============================================================================
# CACHE CONFIGURATION
# ============================================================================

CACHE_THUMBNAILS_DIR = CACHE_DIR / "thumbnails"
CACHE_MAX_SIZE = 100 * 1024 * 1024  # 100MB

# ============================================================================
# DEBUG & DEVELOPMENT
# ============================================================================

DEBUG_MODE = False
LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR
