#!/bin/bash
#
# before run, check language in jsons.py and allowed chars in pack.py also lang-specific model in recognize.py

wd="$(dirname "$0")"

uv run --with stable-ts python3 "$wd/jsons.py"
uv run --with soundfile python3 "$wd/cutit.py"
uv run --with transformers --with soundfile --with torch --with tqdm python3 "$wd/recognize.py"
rm *.fixed.tsv
for file in *.tsv; do     [[ "$file" == "flacs.tsv" ]] && continue;
 uv run --with python-Levenshtein python3 "$wd/fix_align.py" "$file" "flacs.tsv" "$file.fixed.tsv";
done
uv run --with soundfile --with pyarrow python3 "$wd/pack.py"
