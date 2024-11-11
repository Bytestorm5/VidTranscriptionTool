import torch.amp
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

    # First Loop: Split the audio into chunks
    print("Splitting audio into chunks...")
    for i in tqdm(range(number_of_chunks), desc="Splitting Chunks"):
        start = i * chunk_length
        end = min((i + 1) * chunk_length, audio_length)
        chunk = audio[start:end]
        
        # Export the chunk to the chunks directory
        chunk_filename = os.path.join(chunks_dir, f"chunk_{i}.mp3")
        chunk.export(chunk_filename, format="mp3")

    # Load the Whisper model onto the device
    model = whisper.load_model("large", device=device)

    # Initialize a variable to hold the transcribed text
    full_text = ""

    # Prepare the prompt based on languages
    languages_formatted = [LANGUAGES[l] if l in LANGUAGES else l for l in languages]
    prompt = f"This transcript is primarily written in " + " & ".join(languages_formatted) + "."

    # Second Loop: Transcribe each chunk
    print("Transcribing chunks...")
    chunk_files = sorted(os.listdir(chunks_dir))  # Ensure consistent order
    for chunk_file in tqdm(chunk_files, desc="Transcribing Chunks"):
        chunk_path = os.path.join(chunks_dir, chunk_file)
        
        # Transcribe the audio chunk
        with torch.amp.autocast_mode.autocast(device):
            result = model.transcribe(chunk_path, initial_prompt=prompt)
        full_text += result["text"] + " "
        
        # Optionally, remove the chunk file after processing
        # os.remove(chunk_path)

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
