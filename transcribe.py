import whisper
from whisper.decoding import DecodingOptions
from whisper.tokenizer import LANGUAGES
from pydub import AudioSegment
import math
import os
import shutil
from tqdm import tqdm
import torch
import sys
import argparse
from urllib.request import urlopen
import json

def transcribe(languages: list[str]):
    # Check if CUDA is available and set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Define the directory for chunks
    chunks_dir = 'chunks'

    # Clear the chunks directory if it exists, then recreate it
    if os.path.exists(chunks_dir):
        shutil.rmtree(chunks_dir)
    os.makedirs(chunks_dir)

    # Load the audio from the video file
    audio = AudioSegment.from_file('video.mp4')
    audio_length = len(audio)  # Length in milliseconds

    # Define chunk length (10 minutes in milliseconds)
    chunk_length = 10 * 60 * 1000  # 600,000 milliseconds

    # Calculate the number of chunks
    number_of_chunks = math.ceil(audio_length / chunk_length)

    # Load the Whisper model onto the device
    model = whisper.load_model("large", device=device)

    # Initialize a variable to hold the transcribed text
    full_text = ""

    # Process each chunk with a progress bar
    for i in tqdm(range(number_of_chunks), desc="Processing Chunks"):
        start = i * chunk_length
        end = min((i + 1) * chunk_length, audio_length)
        chunk = audio[start:end]
        
        # Export the chunk to the chunks directory
        chunk_filename = os.path.join(chunks_dir, f"chunk_{i}.mp3")
        chunk.export(chunk_filename, format="mp3")
        
        # Transcribe the audio chunk
        result = model.transcribe(chunk_filename, language=languages[0])  # Use the first language
        full_text += result["text"] + " "
        
        # Optionally, remove the chunk file after processing
        # os.remove(chunk_filename)

    # Print the combined transcribed text
    print(full_text)
    with open("OUTPUT.txt", 'w+', encoding='utf-8') as writer:
        writer.write(full_text)

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Transcribe audio with Whisper.")
    parser.add_argument(
        "-l", "--languages", nargs="+", required=True,
        help="List of languages to transcribe audio into."
    )
    args = parser.parse_args()

    # Fetch valid Whisper languages
    valid_languages = LANGUAGES

    # Validate provided languages
    for lang in args.languages:
        if lang not in valid_languages:
            print(f"Error: '{lang}' is not a valid language. Valid options are:")
            print(", ".join(valid_languages.keys()))
            sys.exit(1)

    # Run transcription
    transcribe(args.languages)
