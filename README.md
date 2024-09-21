# Video Transcription Tool

Quick thing I threw together because someone I know was about to spend far too much money ($5) on an online transcription service.

Runs completely locally, and completely for free (as long as you don't count electricity.. or the price of your hardware.. etc.)

## Setup

### PyTorch
Install PyTorch from here as per your system specs: https://pytorch.org/#stable

PyTorch is not included in requirements.txt as its specifics are different per system.

Typically you should be able to find your CUDA version by running `nvcc --version`. If the command isn't recognized, you need to install Nvidia Cuda Toolkit.

### ffmpeg
ffmpeg is needed for the `whisper` library. Details can be found in the whisper repo: https://github.com/openai/whisper

## Usage
Run `tool.py`, and paste the link to the video you want to transcribe.
