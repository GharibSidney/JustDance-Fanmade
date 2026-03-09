# Just Dance UI

A Python-based user interface for the Just Dance game. Provides an interactive song selection interface with video thumbnails and audio preview on hover.

## Features

- **Song Selection**: Browse and select songs from your music library
- **Video Thumbnails**: Automatic extraction of middle frames from videos as song thumbnails
- **Audio Preview**: Hover over any song card to preview a short audio clip
- **Settings**: Configure movement prediction frame count
- **White Theme**: Clean, modern white interface design
- **Responsive Design**: Adapts to different screen sizes

## Requirements

- Python 3.10+
- PyQt6
- opencv-python
- pygame
- numpy

## Installation

1. Install dependencies:
```bash
pip install PyQt6 opencv-python pygame numpy
```

2. Run the application:
```bash
python main.py
```

## Music Folder Structure

Place your music files in the `musics` folder with the following structure:

```
musics/
├── SongName1/
│   ├── Video/
│   │   └── song_video.mp4
│   └── Audio/
│       └── song_audio.mp3
├── SongName2/
│   ├── Video/
│   │   └── dance_video.avi
│   └── Audio/
│       └── dance_audio.wav
└── ...
```

### Supported Formats

- **Video**: MP4, AVI, MOV, MKV, WEBM
- **Audio**: MP3, WAV, OGG, FLAC, M4A

## Usage

1. **Browse Songs**: The main page displays all available songs as cards with video thumbnails
2. **Preview Audio**: Hover over any song card to hear a preview
3. **Select a Song**: Click on a song card to start playing
4. **Settings**: Click the gear icon to access settings

### Settings

- **Movement Prediction Frames**: Adjust the number of frames used for movement prediction (5-30 frames)
  - Higher values provide smoother predictions but may add latency
  - Lower values respond faster but may be less accurate

## Configuration

Settings are stored in `settings.json` in the application directory.

## Project Structure

```
.
├── constants.py          # Application constants and configuration
├── main.py              # Application entry point
├── main_window.py       # Main window and navigation
├── song_page.py         # Song selection page
├── song_card.py         # Song card widget
├── settings_page.py     # Settings page
├── settings_manager.py  # Settings persistence
├── media_manager.py     # Video/audio handling
└── musics/              # Music files directory
    └── (song folders)
```