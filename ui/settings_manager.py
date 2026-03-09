"""
Settings Manager module for Just Dance UI.
Handles loading, saving, and managing application settings.
"""

import json
import os
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import constants


class SettingsManager:
    """
    Singleton class for managing application settings.
    Provides methods to load, save, and access settings with persistence.
    """

    _instance: Optional['SettingsManager'] = None
    _settings: Dict[str, Any] = {}
    _logger: logging.Logger = None

    def __new__(cls) -> 'SettingsManager':
        """Implement singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        """Initialize the settings manager."""
        if self._initialized:
            return

        self._setup_logging()
        self._load_settings()
        self._initialized = True

    def _setup_logging(self) -> None:
        """Configure logging for this module."""
        self._logger = logging.getLogger(self.__class__.__name__)
        self._logger.setLevel(getattr(logging, constants.LOG_LEVEL))

        # Create logs directory if it doesn't exist
        constants.ERROR_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

        # File handler
        file_handler = logging.FileHandler(constants.ERROR_LOG_FILE)
        file_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        self._logger.addHandler(file_handler)

    def _load_settings(self) -> None:
        """Load settings from the JSON file or create default settings."""
        try:
            if constants.SETTINGS_FILE.exists():
                with open(constants.SETTINGS_FILE, 'r', encoding='utf-8') as f:
                    self._settings = json.load(f)
                self._logger.info(f"Settings loaded from {constants.SETTINGS_FILE}")
            else:
                self._settings = self._get_default_settings()
                self._save_settings()
                self._logger.info("Default settings created")
        except json.JSONDecodeError as e:
            self._logger.error(f"Failed to parse settings file: {e}")
            self._settings = self._get_default_settings()
        except Exception as e:
            self._logger.error(f"Failed to load settings: {e}")
            self._settings = self._get_default_settings()

    def _get_default_settings(self) -> Dict[str, Any]:
        """Return default settings dictionary."""
        return {
            constants.SETTING_PREDICTION_FRAMES: constants.DEFAULT_PREDICTION_FRAMES
        }

    def _save_settings(self) -> bool:
        """Save current settings to the JSON file."""
        try:
            # Ensure parent directory exists
            constants.SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)

            with open(constants.SETTINGS_FILE, 'w', encoding='utf-8') as f:
                json.dump(self._settings, f, indent=4, ensure_ascii=False)

            self._logger.info("Settings saved successfully")
            return True
        except Exception as e:
            self._logger.error(f"Failed to save settings: {e}")
            return False

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a setting value by key.

        Args:
            key: The setting key to retrieve
            default: Default value if key doesn't exist

        Returns:
            The setting value or default
        """
        return self._settings.get(key, default)

    def set(self, key: str, value: Any, save: bool = True) -> bool:
        """
        Set a setting value and optionally save to file.

        Args:
            key: The setting key to set
            value: The value to set
            save: Whether to save to file immediately

        Returns:
            True if successful, False otherwise
        """
        old_value = self._settings.get(key)

        if old_value == value:
            return True

        self._settings[key] = value
        self._logger.debug(f"Setting '{key}' changed from {old_value} to {value}")

        if save:
            return self._save_settings()
        return True

    def get_prediction_frames(self) -> int:
        """Get the movement prediction frames setting."""
        return self.get(
            constants.SETTING_PREDICTION_FRAMES,
            constants.DEFAULT_PREDICTION_FRAMES
        )

    def set_prediction_frames(self, frames: int) -> bool:
        """
        Set the movement prediction frames setting.

        Args:
            frames: Number of frames for prediction (clamped to valid range)

        Returns:
            True if successful
        """
        # Clamp to valid range
        frames = max(constants.MIN_PREDICTION_FRAMES,
                     min(constants.MAX_PREDICTION_FRAMES, frames))

        return self.set(constants.SETTING_PREDICTION_FRAMES, frames)

    def reset_to_defaults(self) -> bool:
        """Reset all settings to default values."""
        self._settings = self._get_default_settings()
        return self._save_settings()

    @property
    def all_settings(self) -> Dict[str, Any]:
        """Get all current settings as a dictionary."""
        return self._settings.copy()

    def reload(self) -> None:
        """Reload settings from file."""
        self._load_settings()


# Module-level convenience functions
_settings_manager: Optional[SettingsManager] = None


def get_settings_manager() -> SettingsManager:
    """Get the singleton settings manager instance."""
    global _settings_manager
    if _settings_manager is None:
        _settings_manager = SettingsManager()
    return _settings_manager
