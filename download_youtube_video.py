# import yt_dlp

# url = "https://www.youtube.com/watch?v=NJh5idlanrc"

# ydl_opts = {
#     'format': 'bestvideo[height<=720]+bestaudio/best[height<=720]',
#     'outtmpl': 'downloads/%(title)s.%(ext)s',
# }

# with yt_dlp.YoutubeDL(ydl_opts) as ydl:
#     ydl.download([url])


# Normal format
import yt_dlp

url = "https://youtu.be/TyjsSXEn4mE?si=MhWm7Uu7Htac3arG"

ydl_opts = {
    'outtmpl': 'downloads/%(title)s.%(ext)s',
    # 'merge_output_format': 'mp4',
}

with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    ydl.download([url])