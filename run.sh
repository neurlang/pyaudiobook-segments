#!/bin/bash
#
# before run, check language in jsons.py and allowed chars in pack.py

uv run --with stable-ts python3 jsons.py
uv run --with soundfile python3 cutit.py
uv run --with soundfile --with pyarrow python3 pack.py
