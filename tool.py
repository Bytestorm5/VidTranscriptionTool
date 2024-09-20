from transcribe import transcribe
from download import download_audio_ytdlp

link = "https://www.youtube.com/watch?v=5RpiOeyYkMo"

download_audio_ytdlp(link)
transcribe()