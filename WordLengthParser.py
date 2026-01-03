#!/usr/bin/env python3
"""
Split a word list into per-length files.

- Reads an input Words.txt (one word per line)
- Normalizes to uppercase (crossword-friendly)
- Keeps only A–Z words by default (configurable)
- Writes output files like: words-2.txt, words-3.txt, ...
- Also writes a summary file: lengths.json (counts per length)

Usage:
  python split_words_by_length.py --input Data/Words.txt --outdir Data/words_by_length
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from typing import Dict, List


ALPHA_RE = re.compile(r"^[A-Za-z]+$")


def iter_words(path: str, *, uppercase: bool, alpha_only: bool, min_len: int, max_len: int | None):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            w = line.strip()
            if not w:
                continue

            # Optional: strip surrounding quotes or weird whitespace
            # (you can expand this if your list includes more formats)
            w = w.strip()

            if uppercase:
                w = w.upper()

            if alpha_only and not ALPHA_RE.match(w):
                continue

            L = len(w)
            if L < min_len:
                continue
            if max_len is not None and L > max_len:
                continue

            yield w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to Words.txt")
    ap.add_argument("--outdir", required=True, help="Output directory for words-N.txt files")
    ap.add_argument("--min-len", type=int, default=2, help="Minimum word length to keep (default: 2)")
    ap.add_argument("--max-len", type=int, default=None, help="Maximum word length to keep (default: no limit)")
    ap.add_argument("--keep-non-alpha", action="store_true",
                    help="Keep words with non A–Z characters (default: filter them out)")
    ap.add_argument("--no-uppercase", action="store_true",
                    help="Do not uppercase words (default: uppercase them)")
    ap.add_argument("--dedupe", action="store_true",
                    help="Remove duplicates within each length (costs memory)")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    uppercase = not args.no_uppercase
    alpha_only = not args.keep_non_alpha

    # Collect words by length.
    # For huge lists, this is usually still fine (strings dominate memory),
    # but if you want streaming writes instead, say so and I’ll provide that version.
    by_len: Dict[int, List[str]] = defaultdict(list)
    seen = defaultdict(set) if args.dedupe else None

    total = 0
    for w in iter_words(
        args.input,
        uppercase=uppercase,
        alpha_only=alpha_only,
        min_len=args.min_len,
        max_len=args.max_len,
    ):
        L = len(w)
        if seen is not None:
            if w in seen[L]:
                continue
            seen[L].add(w)
        by_len[L].append(w)
        total += 1

    # Write files
    counts = {}
    for L in sorted(by_len.keys()):
        out_path = os.path.join(args.outdir, f"words-{L}.txt")
        # Optional: sort for determinism (helpful for debugging / reproducible builds)
        by_len[L].sort()
        with open(out_path, "w", encoding="utf-8") as out:
            out.write("\n".join(by_len[L]))
            out.write("\n")
        counts[L] = len(by_len[L])

    # Write a small manifest
    manifest_path = os.path.join(args.outdir, "lengths.json")
    with open(manifest_path, "w", encoding="utf-8") as mf:
        json.dump(
            {
                "input": os.path.abspath(args.input),
                "total_words_written": total,
                "min_len": args.min_len,
                "max_len": args.max_len,
                "alpha_only": alpha_only,
                "uppercased": uppercase,
                "counts_by_length": counts,
            },
            mf,
            indent=2,
            sort_keys=True,
        )

    print(f"Done. Wrote {len(counts)} files to {args.outdir}")
    print(f"Total words written: {total}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
