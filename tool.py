from transcribe import transcribe
from download import download_audio_ytdlp

link = input("Paste your Youtube Video Link here:")

download_audio_ytdlp(link)
transcribe()