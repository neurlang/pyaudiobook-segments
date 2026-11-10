import glob, os, re, json, random, hashlib, sys
import soundfile as sf
import pyarrow as pa
import pyarrow.parquet as pq

# ---- your allowed character set (digits included via 012...54...9678) ----
ALLOWED = set(
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789"
    " ,áíž.ýčúéťšľôďňóĎä?Č-Ú:–%„“ĺŠ§/Ž!ÁĽ);ŕ´Íöř’&ŤÉő\"ěüÓ+ů‒'Ňçà”ćʺÔÝ=»°«ë•"
)
MAX_DUR = 30.0
DEV_FRAC = 0.10          # ~10% of source files to dev
SEED = 0

def text_ok(t: str) -> bool:
    t = t.strip()
    if not t:
        return False
    return all(ch in ALLOWED for ch in t)

def is_dev(stem: str) -> bool:
    h = int(hashlib.md5(stem.encode('utf-8')).hexdigest(), 16)
    return (h % 1000) < int(DEV_FRAC * 1000)

def load_tsv(path):
    for line in open(path, encoding='utf-8'):
        line = line.rstrip('\n')
        if not line:
            continue
        fname, _, text = line.partition('\t')
        yield fname, text

def collect():
    train_rows, dev_rows = [], []
    stats = {"total": 0, "bad_text": 0, "too_long": 0, "missing": 0, "kept": 0}
    for tsv in sorted(glob.glob('*.tsv')):
        src_stem = os.path.splitext(tsv)[0]          # "01 Uvod"
        bucket = dev_rows if is_dev(src_stem) else train_rows
        for fname, text in load_tsv(tsv):
            stats["total"] += 1
            if not os.path.isfile(fname):
                stats["missing"] += 1
                continue
            if not text_ok(text):
                stats["bad_text"] += 1
                continue
            dur = float(sf.info(fname).duration)
            if dur > MAX_DUR:
                stats["too_long"] += 1
                continue
            with open(fname, 'rb') as f:
                audio_bytes = f.read()
            bucket.append({
                "audio": {"bytes": audio_bytes, "path": fname},
                "text": text.strip(),
                "duration": dur,
            })
            stats["kept"] += 1
    return train_rows, dev_rows, stats

SCHEMA = pa.schema([
    ("audio", pa.struct([("bytes", pa.binary()), ("path", pa.string())])),
    ("text", pa.string()),
    ("duration", pa.float32()),
])

def write_parquet(rows, outdir, shard_size=1000):
    os.makedirs(outdir, exist_ok=True)
    if not rows:
        print(f"  (no rows for {outdir})")
        return 0
    shard = 0
    for i in range(0, len(rows), shard_size):
        table = pa.Table.from_pylist(rows[i:i+shard_size], schema=SCHEMA)
        pq.write_table(table, os.path.join(outdir, f"part-{shard:05d}.parquet"))
        shard += 1
    return len(rows)

def main():
    random.seed(SEED)
    train_rows, dev_rows, stats = collect()
    print("stats:", json.dumps(stats, ensure_ascii=False))
    n_tr = write_parquet(train_rows, "train")
    n_dv = write_parquet(dev_rows, "dev")
    print(f"train: {n_tr} rows, dev: {n_dv} rows")

if __name__ == "__main__":
    main()
