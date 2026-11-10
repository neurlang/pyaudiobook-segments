# pyaudiobook-segments
Python Audio Book Segments Cutting for ML training

## Initial state:

**Problem**: You have a directory with `.mp3` files and corresponding `.mp3.txt` files

## Final state:

**Solution**: You need to create a hugging face dataset for ASR training.
You run `./run.sh` and a two parquet directories will be generated with
segmented audio and texts: `train/` and `dev/` shard ready for hugging face.
Each audio will have 16Khz and be mono FLAC, less than 30s, with corresponding
text and duration in `.parquet`.

## Before you `run.sh`

0. Install `ffmpeg`, `uv`
1. Customize `LANGUAGE` in `jsons.py` (based on your audiobook language)
2. Customize `ALLOWED` character set in `pack.py`

## Credits

- `ffmpeg`
- `stable_whisper`
- `soundfile`
