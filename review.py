#!/usr/bin/env python3
import glob, csv, sys, os

# --- Levenshtein (fast if a lib is installed, else pure-Python DP) ---
try:
    from Levenshtein import distance as lev
except ImportError:
    try:
        from rapidfuzz.distance import Levenshtein as _rf
        def lev(a, b): return _rf.distance(a, b)
    except ImportError:
        def lev(a, b):
            if a == b: return 0
            la, lb = len(a), len(b)
            if la == 0: return lb
            if lb == 0: return la
            if la < lb:
                a, b = b, a
                la, lb = lb, la
            prev = list(range(lb + 1))
            for i, ca in enumerate(a, 1):
                cur = [i]
                for j, cb in enumerate(b, 1):
                    cost = 0 if ca == cb else 1
                    cur.append(min(prev[j] + 1, cur[j-1] + 1, prev[j-1] + cost))
                prev = cur
            return prev[lb]

def read_tsv(path):
    rows = []
    with open(path, newline='', encoding='utf-8', errors='replace') as f:
        for r in csv.reader(f, delimiter='\t'):
            if len(r) >= 2:
                rows.append((r[0], r[1]))
    return rows

def main():
    fixed = []
    for p in sorted(glob.glob('*.fixed.tsv')):
        fixed.extend(read_tsv(p))
    fixed.sort(key=lambda x: x[0])

    flacs = read_tsv('flacs.tsv')
    flacs.sort(key=lambda x: x[0])

    if len(fixed) != len(flacs):
        print(f"warning: length mismatch fixed={len(fixed)} flacs={len(flacs)}",
              file=sys.stderr)

    results = []
    for i, ((kf, tf), (kx, tx)) in enumerate(zip(fixed, flacs)):
        if kf != kx:
            print(f"warning: key mismatch at row {i}: {kf!r} vs {kx!r}",
                  file=sys.stderr)
            continue
        d = lev(tf, tx)
        # (distance, key, flacs_text, fixed_text)  -- note output column order
        results.append((d, kf, tx, tf))

    results.sort(key=lambda r: r[0], reverse=True)

    for d, k, tx, tf in results[:100]:
        if not os.path.isfile(k):
            continue
        print(f"{d}\t{k}\t{tx}\t{tf}")

if __name__ == '__main__':
    main()
