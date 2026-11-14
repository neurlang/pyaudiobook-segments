import glob
import os

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

MODEL_ID = "kinit/whisper-large-v3-sk"
AUDIO_DIR = "."  # folder containing .flac files
OUTPUT_FILE = "flacs.tsv"
BATCH_SIZE = 16  # 30-s chunks per forward pass (lower it if you run out of GPU memory)
NUM_WORKERS = 4  # background workers for loading / resampling / splitting
TARGET_SR = 16_000
MAX_CHUNK_S = 30.0  # Whisper's window
SEARCH_FROM_S = 20.0  # cut at the quietest spot between 20 s and 30 s of each chunk
MIN_TAIL_S = 0.3  # drop a leftover tail shorter than this


def split_audio(audio):
    """Split into <= 30 s pieces, cutting at the quietest 100 ms frame so words aren't sliced."""
    max_len = int(MAX_CHUNK_S * TARGET_SR)
    if len(audio) == 0:
        return [np.zeros(TARGET_SR // 10, dtype=np.float32)]
    frame = int(0.1 * TARGET_SR)
    chunks, start = [], 0
    while len(audio) - start > max_len:
        lo, hi = start + int(SEARCH_FROM_S * TARGET_SR), start + max_len
        seg = audio[lo:hi]
        n = len(seg) // frame
        rms = np.sqrt((seg[: n * frame].reshape(n, frame) ** 2).mean(axis=1))
        cut = lo + int(rms.argmin()) * frame + frame // 2
        chunks.append(audio[start:cut])
        start = cut
    tail = audio[start:]
    if len(tail) >= MIN_TAIL_S * TARGET_SR or not chunks:
        chunks.append(tail)
    return chunks


class FlacDataset(Dataset):
    """One item per file: (name, [chunks of mono 16 kHz float32 audio, each <= 30 s])."""

    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        audio, sr = sf.read(path, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != TARGET_SR:
            audio = AF.resample(torch.from_numpy(audio), sr, TARGET_SR).numpy()
        return os.path.basename(path), split_audio(audio)


def identity(x):  # top-level so it also works with spawn-based multiprocessing
    return x


def clean(text):
    return text.strip().replace("\t", " ").replace("\n", " ")


def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        MODEL_ID,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    )
    model.to(device).eval()
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    paths = sorted(glob.glob(os.path.join(AUDIO_DIR, "*.flac")))
    if not paths:
        print(f"No .flac files found in {os.path.abspath(AUDIO_DIR)}")
        return

    loader = DataLoader(
        FlacDataset(paths),
        batch_size=None,  # each item is already one file; we batch chunks ourselves
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=identity,
    )

    pending = []  # (name, chunk_idx, audio) waiting for a batch
    parts = {}  # name -> list of chunk transcripts
    remaining = {}  # name -> chunks still to decode
    results = {}

    def transcribe(batch):
        feats = processor.feature_extractor(
            [audio for _, _, audio in batch],
            sampling_rate=TARGET_SR,
            return_tensors="pt",
        ).input_features.to(device, torch_dtype)
        with torch.inference_mode():
            # Plain short-form decoding (no timestamps), same mode as the original script
            ids = model.generate(feats, language="sk", task="transcribe")
        return processor.batch_decode(ids, skip_special_tokens=True)

    def drain(f, final=False):
        while len(pending) >= BATCH_SIZE or (final and pending):
            batch = pending[:BATCH_SIZE]
            del pending[:BATCH_SIZE]
            for (name, idx, _), text in zip(batch, transcribe(batch)):
                parts[name][idx] = text.strip()
                remaining[name] -= 1
                if remaining[name] == 0:  # all chunks of this file are done
                    full = clean(" ".join(t for t in parts.pop(name) if t))
                    results[name] = full
                    f.write(f"{name}\t{full}\n")
                    f.flush()
                    tqdm.write(f"{name}: {full}")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for name, chunks in tqdm(loader, total=len(paths), desc="Loading files", unit="file"):
            parts[name] = [None] * len(chunks)
            remaining[name] = len(chunks)
            pending.extend((name, i, c) for i, c in enumerate(chunks))
            drain(f)
        drain(f, final=True)

    # Rewrite the TSV in filename order
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for name in sorted(results):
            f.write(f"{name}\t{results[name]}\n")


if __name__ == "__main__":
    main()
