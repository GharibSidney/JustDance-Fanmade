import os
# Add torch DLL directory explicitly
import site
for sp in site.getsitepackages():
    dll_path = os.path.join(sp, "torch", "lib")
    if os.path.exists(dll_path):
        os.add_dll_directory(dll_path)
        break

import platform
if platform.system() == "Windows":
    import ctypes
    from importlib.util import find_spec
    try:
        if (spec := find_spec("torch")) and spec.origin and os.path.exists(
            dll_path := os.path.join(os.path.dirname(spec.origin), "lib", "c10.dll")
        ):
            ctypes.CDLL(os.path.normpath(dll_path))
    except Exception:
        pass
import torch
"""
Just Dance UI - Main Entry Point

A Python-based user interface for the Just Dance game.
Provides song selection, audio preview on hover, and settings management.
"""

import sys
import logging
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

import constants
from main_window import create_application, show_splash_and_start


def setup_logging() -> None:
    """Configure application-wide logging."""
    # Create logs directory if it doesn't exist
    constants.ERROR_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, constants.LOG_LEVEL),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(constants.ERROR_LOG_FILE),
            logging.StreamHandler(sys.stdout)
        ]
    )


def main() -> int:
    """
    Main entry point for the application.

    Returns:
        Exit code (0 for success)
    """
    # Setup logging
    setup_logging()

    logger = logging.getLogger(__name__)
    logger.info(f"Starting {constants.APP_NAME} v{constants.APP_VERSION}")

    try:
        # Create application
        app = create_application()

        # Ensure required directories exist
        _ensure_directories()

        # Show splash and start main window
        show_splash_and_start(app)

        # Run event loop
        return app.exec()

    except Exception as e:
        logger.critical(f"Application failed to start: {e}", exc_info=True)
        return 1


def _ensure_directories() -> None:
    """Ensure required directories exist."""
    # Create musics directory
    constants.MUSICS_DIR.mkdir(parents=True, exist_ok=True)

    # Create cache directory
    constants.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    constants.CACHE_THUMBNAILS_DIR.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    sys.exit(main())
