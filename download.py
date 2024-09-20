import yt_dlp

def download_audio_ytdlp(youtube_url, download_path='.'):
    # Options to download only the audio
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',  # Choose 'mp3', 'm4a', 'wav', etc.
            'preferredquality': '192',  # Audio quality (bitrate)
        }],
        'outtmpl': f'{download_path}/audio',  # Save with title as file name
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([youtube_url])