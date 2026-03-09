"""
Media Manager module for Just Dance UI.
Handles video frame extraction, audio playback, and media caching.
"""

import os
import logging
import hashlib
from pathlib import Path
from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass

import cv2
import numpy as np
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import QObject, pyqtSignal, QThread

import constants


@dataclass
class SongInfo:
    """Data class representing a song with its metadata."""
    name: str
    folder_path: Path
    video_path: Optional[Path] = None
    audio_path: Optional[Path] = None
    thumbnail: Optional[QPixmap] = None
    has_video: bool = False
    has_audio: bool = False


class AudioPlayer(QObject):
    """
    Audio player class for handling audio playback.
    Uses pygame mixer for low-latency audio playback.
    """
    import pygame

    # Signals
    playback_started = pyqtSignal()
    playback_stopped = pyqtSignal()
    playback_error = pyqtSignal(str)

    def __init__(self) -> None:
        """Initialize the audio player."""
        super().__init__()
        self._logger = logging.getLogger(self.__class__.__name__)
        self._current_sound = None
        self._current_channel = None
        self._is_playing = False

        # Initialize pygame mixer
        try:
            self.pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
            self._logger.info("Audio player initialized successfully")
        except Exception as e:
            self._logger.error(f"Failed to initialize audio mixer: {e}")
            self.playback_error.emit(str(e))

    def play(self, audio_path: Path, fade_ms: int = 0) -> bool:
        """
        Play an audio file.

        Args:
            audio_path: Path to the audio file
            fade_ms: Fade in duration in milliseconds

        Returns:
            True if successful, False otherwise
        """
        if not audio_path or not audio_path.exists():
            self._logger.warning(f"Audio file not found: {audio_path}")
            return False

        try:
            # Stop current playback
            self.stop()

            # Load and play new audio
            self._current_sound = self.pygame.mixer.Sound(str(audio_path))
            self._current_channel = self._current_sound.play(fade_ms=fade_ms)
            self._is_playing = True

            self.playback_started.emit()
            self._logger.debug(f"Playing audio: {audio_path.name}")
            return True

        except Exception as e:
            self._logger.error(f"Failed to play audio {audio_path}: {e}")
            self.playback_error.emit(str(e))
            return False

    def stop(self, fade_ms: int = 0) -> None:
        """
        Stop current audio playback.

        Args:
            fade_ms: Fade out duration in milliseconds
        """
        if self._current_sound and self._current_channel:
            try:
                if fade_ms > 0:
                    # Fade out and stop
                    self._current_channel.fadeout(fade_ms)
                else:
                    self._current_channel.stop()

                self._is_playing = False
                self.playback_stopped.emit()
            except Exception as e:
                self._logger.error(f"Error stopping audio: {e}")

    def is_playing(self) -> bool:
        """Check if audio is currently playing."""
        return self._is_playing and self._current_channel is not None and self._current_channel.get_busy()

    def set_volume(self, volume: float) -> None:
        """
        Set the playback volume.

        Args:
            volume: Volume level (0.0 to 1.0)
        """
        volume = max(0.0, min(1.0, volume))
        if self._current_sound:
            self._current_sound.set_volume(volume)


class MediaManager(QObject):
    """
    Singleton class for managing media files (videos and audio).
    Handles scanning, thumbnail generation, and caching.
    """

    _instance: Optional['MediaManager'] = None

    # Signals
    scan_started = pyqtSignal()
    scan_completed = pyqtSignal(list)
    scan_error = pyqtSignal(str)
    thumbnail_ready = pyqtSignal(str, QPixmap)

    def __new__(cls) -> 'MediaManager':
        """Implement singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        """Initialize the media manager."""
        if self._initialized:
            return

        super().__init__()
        self._logger = logging.getLogger(self.__class__.__name__)
        self._logger.setLevel(getattr(logging, constants.LOG_LEVEL))

        # File handlers
        self._audio_player: Optional[AudioPlayer] = None

        # Cache
        self._thumbnail_cache: Dict[str, QPixmap] = {}
        self._song_list: List[SongInfo] = []

        # Ensure cache directory exists
        constants.CACHE_THUMBNAILS_DIR.mkdir(parents=True, exist_ok=True)

        self._initialized = True
        self._logger.info("Media manager initialized")

    @property
    def audio_player(self) -> AudioPlayer:
        """Get the audio player instance."""
        if self._audio_player is None:
            self._audio_player = AudioPlayer()
        return self._audio_player

    def scan_music_folder(self) -> List[SongInfo]:
        """
        Scan the musics folder and extract song information.

        Returns:
            List of SongInfo objects
        """
        self.scan_started.emit()
        self._song_list = []

        musics_path = constants.MUSICS_DIR

        if not musics_path.exists():
            self._logger.warning(f"Musics folder not found: {musics_path}")
            self.scan_error.emit(constants.ERROR_NO_MUSICS_FOLDER)
            self.scan_completed.emit([])
            return []

        try:
            # Iterate through subdirectories
            for item in sorted(musics_path.iterdir()):
                if not item.is_dir():
                    continue

                song_info = self._process_song_folder(item)
                if song_info and (song_info.has_video or song_info.has_audio):
                    self._song_list.append(song_info)

            self._logger.info(f"Scan completed. Found {len(self._song_list)} songs")
            self.scan_completed.emit(self._song_list)
            return self._song_list

        except Exception as e:
            self._logger.error(f"Error scanning music folder: {e}")
            self.scan_error.emit(str(e))
            self.scan_completed.emit([])
            return []

    def _process_song_folder(self, folder_path: Path) -> Optional[SongInfo]:
        """
        Process a single song folder and extract metadata.

        Args:
            folder_path: Path to the song folder

        Returns:
            SongInfo object or None if invalid
        """
        song_name = folder_path.name

        # Look for video file
        video_path = self._find_media_file(
            folder_path / constants.VIDEO_SUBDIR,
            constants.VIDEO_SUPPORTED_FORMATS
        )

        # Look for audio file
        audio_path = self._find_media_file(
            folder_path / constants.AUDIO_SUBDIR,
            constants.AUDIO_SUPPORTED_FORMATS
        )

        if not video_path and not audio_path:
            self._logger.debug(f"Skipping {song_name}: no video or audio found")
            return None

        song_info = SongInfo(
            name=song_name,
            folder_path=folder_path,
            video_path=video_path,
            audio_path=audio_path,
            has_video=video_path is not None,
            has_audio=audio_path is not None
        )

        # Generate thumbnail if video exists
        if video_path:
            thumbnail = self._extract_video_thumbnail(video_path)
            if thumbnail:
                song_info.thumbnail = thumbnail

        return song_info

    def _find_media_file(self, folder: Path, extensions: List[str]) -> Optional[Path]:
        """
        Find a media file in the specified folder with given extensions.

        Args:
            folder: Folder to search
            extensions: List of file extensions to look for

        Returns:
            Path to the media file or None
        """
        if not folder.exists():
            return None

        for ext in extensions:
            files = list(folder.glob(f"*{ext}"))
            if files:
                return files[0]  # Return first matching file

        return None

    def _extract_video_thumbnail(self, video_path: Path) -> Optional[QPixmap]:
        """
        Extract a thumbnail from the middle frame of a video.

        Args:
            video_path: Path to the video file

        Returns:
            QPixmap thumbnail or None on failure
        """
        # Check cache first
        cache_key = self._get_cache_key(video_path)
        if cache_key in self._thumbnail_cache:
            return self._thumbnail_cache[cache_key]

        try:
            # Open video file
            cap = cv2.VideoCapture(str(video_path))

            if not cap.isOpened():
                self._logger.error(f"Failed to open video: {video_path}")
                return None

            # Get total frames
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if total_frames <= 0:
                self._logger.error(f"Invalid frame count for video: {video_path}")
                cap.release()
                return None

            # Calculate middle frame position
            middle_frame = int(total_frames * constants.VIDEO_FRAME_EXTRACT_POSITION)

            # Set position and read frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame)
            ret, frame = cap.read()

            if not ret or frame is None:
                self._logger.error(f"Failed to read frame from video: {video_path}")
                cap.release()
                return None

            cap.release()

            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Resize to thumbnail dimensions
            height, width, _ = frame_rgb.shape
            aspect_ratio = width / height

            if aspect_ratio > constants.CARD_ASPECT_RATIO:
                new_width = int(constants.THUMBNAIL_HEIGHT * aspect_ratio)
                new_height = constants.THUMBNAIL_HEIGHT
            else:
                new_width = constants.THUMBNAIL_WIDTH
                new_height = int(constants.THUMBNAIL_WIDTH / aspect_ratio)

            frame_resized = cv2.resize(frame_rgb, (new_width, new_height),
                                        interpolation=cv2.INTER_LANCZOS4)

            # Create QImage and QPixmap
            h, w, ch = frame_resized.shape
            bytes_per_line = ch * w
            qt_image = QImage(frame_resized.data, w, h, bytes_per_line,
                             QImage.Format.Format_RGB888)

            pixmap = QPixmap.fromImage(qt_image)

            # Cache the result
            self._thumbnail_cache[cache_key] = pixmap

            self._logger.debug(f"Generated thumbnail for: {video_path.name}")
            return pixmap

        except Exception as e:
            self._logger.error(f"Error extracting thumbnail from {video_path}: {e}")
            return None

    def _get_cache_key(self, file_path: Path) -> str:
        """
        Generate a cache key for a file based on its path and modification time.

        Args:
            file_path: Path to the file

        Returns:
            Cache key string
        """
        mtime = file_path.stat().st_mtime if file_path.exists() else 0
        key_string = f"{file_path}:{mtime}"
        return hashlib.md5(key_string.encode()).hexdigest()

    def get_song_list(self) -> List[SongInfo]:
        """Get the list of scanned songs."""
        return self._song_list

    def get_song_by_name(self, name: str) -> Optional[SongInfo]:
        """
        Get a song by its name.

        Args:
            name: Song name to search for

        Returns:
            SongInfo object or None
        """
        for song in self._song_list:
            if song.name == name:
                return song
        return None

    def play_audio(self, audio_path: Path) -> bool:
        """
        Play audio for a song.

        Args:
            audio_path: Path to the audio file

        Returns:
            True if successful
        """
        return self.audio_player.play(audio_path, fade_ms=constants.AUDIO_FADE_DURATION)

    def stop_audio(self, fade_out: bool = True) -> None:
        """
        Stop current audio playback.

        Args:
            fade_out: Whether to fade out the audio
        """
        fade_ms = constants.AUDIO_FADE_DURATION if fade_out else 0
        self.audio_player.stop(fade_ms=fade_ms)

    def clear_cache(self) -> None:
        """Clear the thumbnail cache."""
        self._thumbnail_cache.clear()
        self._logger.info("Thumbnail cache cleared")


# Module-level convenience functions
def get_media_manager() -> MediaManager:
    """Get the singleton media manager instance."""
    return MediaManager()
