import json, subprocess, os, glob, re, sys, traceback
import soundfile as sf

MAX_DUR = 30.0

def clean(t: str) -> str:
    t = t.replace('\t', ' ').replace('\n', ' ').replace('\r', ' ')
    t = re.sub(r'\s+', ' ', t)
    return t.strip()

def process(json_path: str) -> int:
    audio = json_path[:-5]
    if not os.path.isfile(audio):
        raise FileNotFoundError(f"missing audio for {json_path}: {audio}")
    stem = os.path.splitext(audio)[0].replace(' ', '_')
    data = json.load(open(json_path, encoding='utf-8'))

    file_dur = float(sf.info(audio).duration)        # authoritative file length

    # ---- 1. kept segments ----
    kept = []
    for seg in data.get('segments', []):
        text = clean(seg.get('text', ''))
        if not text:
            continue
        start, end = seg['start'], seg['end']
        if end - start < 0.05:
            continue
        kept.append({'start': start, 'end': end, 'text': text})

    # ---- 2. padded cut points ----
    rows, i, skipped_long = [], 0, 0
    n = len(kept)
    for idx, seg in enumerate(kept):
        prev_seg = kept[idx - 1] if idx > 0     else None
        next_seg = kept[idx + 1] if idx < n - 1 else None

        if prev_seg:
            cut_start = (prev_seg['end'] + seg['start']) / 2
        else:
            cut_start = 0.0                          # first: from start of file

        if next_seg:
            cut_end = (seg['end'] + next_seg['start']) / 2
        else:
            cut_end = file_dur                       # last: to end of file

        # clamp to file bounds just in case timestamps overshoot
        cut_start = max(0.0, min(cut_start, file_dur))
        cut_end   = max(0.0, min(cut_end,   file_dur))

        if cut_end - cut_start > MAX_DUR:
            skipped_long += 1
            continue
        if cut_end - cut_start < 0.05:
            continue

        i += 1
        out = f"{stem}_{i:03d}.flac"
        subprocess.run([
            'ffmpeg', '-y', '-loglevel', 'error',
            '-i', audio,
            '-ss', f'{cut_start:.3f}', '-to', f'{cut_end:.3f}',
            '-ar', '16000', '-ac', '1',
            '-c:a', 'flac', out
        ], check=True)
        rows.append(f"{out}\t{seg['text']}")

    tsv = f"{stem}.tsv"
    with open(tsv, 'w', encoding='utf-8') as f:
        f.write('\n'.join(rows) + '\n')
    if skipped_long:
        print(f"     ({skipped_long} padded slices > {MAX_DUR}s skipped)")
    return len(rows)

total_ok = total_fail = 0
for jf in sorted(glob.glob('*.json')):
    try:
        n = process(jf)
        total_ok += 1
        print(f"OK   {jf}: {n} segments")
    except Exception as e:
        total_fail += 1
        print(f"FAIL {jf}: {e}")
        traceback.print_exc()

print(f"\nDone: {total_ok} ok, {total_fail} failed")
