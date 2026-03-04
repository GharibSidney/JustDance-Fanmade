import yt_dlp

url = "https://www.youtube.com/watch?v=NJh5idlanrc"

ydl_opts = {
    'format': 'bestvideo[height<=360]+bestaudio/best[height<=360]',
    'outtmpl': 'downloads/%(title)s.%(ext)s',
}

with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    ydl.download([url])