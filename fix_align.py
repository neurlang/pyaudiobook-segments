#!/usr/bin/env python3
"""Re-align segment boundaries in REF to match HYP using Levenshtein.

Usage:
    python fix_align.py REF.tsv HYP.tsv SOL.tsv

Dependencies: python-Levenshtein  (pip install python-Levenshtein)
              or rapidfuzz        (pip install rapidfuzz)
"""

import csv
import sys


# --------------------------------------------------------------------------- #
# Levenshtein opcodes
# --------------------------------------------------------------------------- #
def get_opcodes(a: str, b: str):
    try:
        import Levenshtein
        return [tuple(op) for op in Levenshtein.opcodes(a, b)]
    except ImportError:
        pass
    try:
        from rapidfuzz.distance import Levenshtein as RF
        return [
            (op.tag.name.lower(), op.src_start, op.src_end,
             op.dest_start, op.dest_end)
            for op in RF.opcodes(a, b)
        ]
    except ImportError:
        sys.exit("Install python-Levenshtein or rapidfuzz")


# --------------------------------------------------------------------------- #
# TSV I/O
# --------------------------------------------------------------------------- #
def read_tsv(path):
    rows = []
    with open(path, encoding="utf-8", newline="") as fh:
        for row in csv.reader(fh, delimiter="\t"):
            if row:
                rows.append(row)
    return rows


def ensure_sorted(rows, name):
    keys = [r[0] for r in rows]
    if keys != sorted(keys):
        print(f"{name}: not sorted – resorting by column 0", file=sys.stderr)
        rows = sorted(rows, key=lambda r: r[0])
    return rows


# --------------------------------------------------------------------------- #
# REF-newline repair
# --------------------------------------------------------------------------- #
def adjust_ref_newlines(ref_full: str, hyp_full: str, opcodes) -> str:
    """Return a copy of ref_full in which every '\n' is replaced by ' ' unless
    the aligned HYP character is also '\n'.

    Why: REF encodes word boundaries as newlines; HYP may encode the same
    boundary as a space (or nothing). If we leave those newlines in when we
    split, we get accidental word merges like 'sním'.
    """
    n = len(ref_full)
    hyp_at_ref = [None] * n
    for tag, i1, i2, j1, j2 in opcodes:
        if tag == "equal":
            for k in range(i2 - i1):
                hyp_at_ref[i1 + k] = hyp_full[j1 + k]
        elif tag == "replace":
            m = min(i2 - i1, j2 - j1)
            for k in range(m):
                hyp_at_ref[i1 + k] = hyp_full[j1 + k]
            if j2 > j1:                       # extra REF chars – best effort
                for k in range(m, i2 - i1):
                    hyp_at_ref[i1 + k] = hyp_full[j2 - 1]

    out = list(ref_full)
    for i, c in enumerate(ref_full):
        if c == "\n" and hyp_at_ref[i] != "\n":
            out[i] = " "
    return "".join(out)


# --------------------------------------------------------------------------- #
# Boundary mapping
# --------------------------------------------------------------------------- #
def map_boundary(b: int, opcodes, ref_len: int) -> int:
    """Map HYP position b (a segment end / '\n' index) to a REF position."""
    for tag, i1, i2, j1, j2 in opcodes:
        if j2 <= b:
            continue
        if j1 >= b:
            return i1
        if tag == "equal":
            return i1 + (b - j1)
        if tag == "replace":
            return min(i1 + (b - j1), i2)
        return i1
    return ref_len


# --------------------------------------------------------------------------- #
# Leading-punctuation cleanup
# --------------------------------------------------------------------------- #
PUNCT_START = set(".,;:!?…“")


def fix_leading_punct(segments):
    for i in range(1, len(segments)):
        seg = segments[i]
        j = 0
        while j < len(seg) and seg[j] in PUNCT_START:
            j += 1
        if j == 0 or j >= len(seg):
            continue                         # nothing to move, or would empty
        if not segments[i - 1].strip():
            continue                         # nowhere to move
        moved = seg[:j]
        rest = seg[j:].lstrip()
        segments[i - 1] = segments[i - 1].rstrip() + moved
        segments[i] = rest
    return segments


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(ref_path, hyp_path, out_path):
    ref_rows = ensure_sorted(read_tsv(ref_path), ref_path)
    hyp_rows = ensure_sorted(read_tsv(hyp_path), hyp_path)

    # Tolerate extra keys in HYP that aren't relevant to this REF file.
    # Keep HYP rows only for keys present in REF, and re-index by REF order.
    hyp_by_key = {h[0]: h for h in hyp_rows}
    missing = [r[0] for r in ref_rows if r[0] not in hyp_by_key]
    if missing:
        sys.exit(f"Keys in REF missing from HYP: {missing[:5]}"
                 f"{'...' if len(missing) > 5 else ''}")
    hyp_rows = [hyp_by_key[r[0]] for r in ref_rows]

    if not ref_rows:
        open(out_path, "w").close()
        return

    ref_segs = [r[1] if len(r) > 1 else "" for r in ref_rows]
    hyp_segs = [h[1] if len(h) > 1 else "" for h in hyp_rows]

    # 1. Concatenate with '\n' (newlines preserved for alignment!)
    ref_full = "\n".join(ref_segs)
    hyp_full = "\n".join(hyp_segs)

    # 2. Align on the *full* strings
    opcodes = get_opcodes(ref_full, hyp_full)

    # 3. Repair REF newlines that HYP treats as spaces / deletions
    ref_adj = adjust_ref_newlines(ref_full, hyp_full, opcodes)

    # 4. HYP boundaries = positions of '\n' inside hyp_full
    hyp_bounds, pos = [], 0
    for seg in hyp_segs[:-1]:
        pos += len(seg)
        hyp_bounds.append(pos)               # index of the '\n'
        pos += 1

    # 5. Map to REF and enforce monotonicity
    ref_bounds, prev = [], 0
    for b in hyp_bounds:
        rb = map_boundary(b, opcodes, len(ref_adj))
        rb = max(prev, min(rb, len(ref_adj)))
        ref_bounds.append(rb)
        prev = rb

    # 6. Split REF_adj
    edges = [0] + ref_bounds + [len(ref_adj)]
    sol_segs = [ref_adj[edges[i]:edges[i + 1]] for i in range(len(edges) - 1)]

    # 7. Clean: any stray '\n' inside a segment becomes space; trim edges
    sol_segs = [s.replace("\n", " ").strip() for s in sol_segs]

    # 8. No segment may start with . , … ; : ! ?
    sol_segs = fix_leading_punct(sol_segs)

    # 9. Write
    with open(out_path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        for row, seg in zip(ref_rows, sol_segs):
            w.writerow([row[0], seg])


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit("Usage: fix_align.py REF.tsv HYP.tsv SOL.tsv")
    main(*sys.argv[1:])
