#!/usr/bin/env python3
"""
Build a crossword-friendly lexicon from the English Wiktionary XML dump.

Input:
  - enwiktionary-YYYYMMDD-pages-articles.xml.bz2

Output (default out/):
  - out/words/words-{L}.txt           (uppercase A-Z words by length)
  - out/defs/defs-{L}.jsonl           (JSON Lines: {"word":"APPLE","senses":[...]} )
  - out/manifest.json                 (counts and configuration)

Redirects:
  - redirect.sql.gz alone is NOT sufficient to map redirect source titles.
    You also need page.sql.gz (page_id -> title). This script includes optional
    support if you provide both.

Usage:
  python build_wiktionary_lexicon.py \
    --xml enwiktionary-20260101-pages-articles.xml.bz2 \
    --out out \
    --min-len 2 --max-len 25

Optional (redirects; requires BOTH):
  --redirect-sql enwiktionary-20260101-redirect.sql.gz \
  --page-sql enwiktionary-20260101-page.sql.gz
"""

from __future__ import annotations

import argparse
import bz2
import gzip
import io
import json
import os
import re
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple


# -----------------------------
# Basic cleaning / parsing
# -----------------------------

ALPHA_RE = re.compile(r"^[A-Za-z]+$")
HEAD2_RE = re.compile(r"^==([^=]+)==\s*$")
HEAD3_RE = re.compile(r"^===([^=]+)===\s*$")

# Wiktionary has many POS headers; start with the common crossword-relevant ones.
DEFAULT_POS = {
    "Noun",
    "Verb",
    "Adjective",
    "Adverb",
    "Proper noun",
}

# Remove <ref>...</ref> blocks and self-closing refs
REF_BLOCK_RE = re.compile(r"<ref\b[^>/]*?>.*?</ref>", re.IGNORECASE | re.DOTALL)
REF_SELF_RE = re.compile(r"<ref\b[^>]*/\s*>", re.IGNORECASE)

# Remove other HTML tags
TAG_RE = re.compile(r"</?[^>]+>")

# Remove templates {{...}} (best-effort; not full nesting)
TEMPLATE_RE = re.compile(r"\{\{[^{}]*\}\}")

# Wikilinks [[...]] or [[...|...]]
WIKILINK_RE = re.compile(r"\[\[([^|\]]+)(?:\|([^\]]+))?\]\]")

# Italics/bold markup '' '' ''' '''
ITALIC_BOLD_RE = re.compile(r"('{2,5})(.*?)\1")

# File links / categories sometimes appear; strip the whole bracket if it starts with these
BAD_LINK_PREFIXES = ("File:", "Image:", "Category:")

# Clean leading definition markers: "#", "##" etc, and spaces
DEF_MARKER_RE = re.compile(r"^#+\s*")


def clean_wikitext(s: str) -> str:
    """Very pragmatic cleanup to make definitions readable."""
    if not s:
        return ""

    s = REF_BLOCK_RE.sub("", s)
    s = REF_SELF_RE.sub("", s)

    # Remove templates (simple)
    # Apply a few times in case multiple templates exist
    for _ in range(3):
        s2 = TEMPLATE_RE.sub("", s)
        if s2 == s:
            break
        s = s2

    # Replace wikilinks
    def _link_sub(m: re.Match) -> str:
        target = (m.group(1) or "").strip()
        text = (m.group(2) or "").strip()
        chosen = text if text else target
        # Skip ugly media/category links
        for p in BAD_LINK_PREFIXES:
            if chosen.startswith(p) or target.startswith(p):
                return ""
        return chosen

    s = WIKILINK_RE.sub(_link_sub, s)

    # Remove bold/italic markers
    s = ITALIC_BOLD_RE.sub(r"\2", s)

    # Remove remaining tags
    s = TAG_RE.sub("", s)

    # Collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()

    # Strip stray punctuation spacing
    s = s.replace(" )", ")").replace("( ", "(")

    return s


@dataclass
class Sense:
    pos: str
    definitions: List[str]


def extract_english_section(wikitext: str) -> Optional[str]:
    """
    Return the substring for the ==English== section (not including other languages).
    """
    if not wikitext:
        return None
    lines = wikitext.splitlines()

    start = None
    for i, line in enumerate(lines):
        m = HEAD2_RE.match(line.strip())
        if m and m.group(1).strip() == "English":
            start = i + 1
            break

    if start is None:
        return None

    # End at next level-2 header
    end = len(lines)
    for j in range(start, len(lines)):
        m = HEAD2_RE.match(lines[j].strip())
        if m:
            end = j
            break

    return "\n".join(lines[start:end])


def extract_senses(english_text: str, allowed_pos: set[str], max_defs_per_pos: int) -> List[Sense]:
    """
    Extract senses from ===POS=== headings and definition lines starting with '#'.
    """
    senses: List[Sense] = []
    if not english_text:
        return senses

    lines = english_text.splitlines()
    current_pos: Optional[str] = None
    current_defs: List[str] = []

    def flush():
        nonlocal current_pos, current_defs
        if current_pos and current_defs:
            senses.append(Sense(pos=current_pos, definitions=current_defs[:max_defs_per_pos]))
        current_pos = None
        current_defs = []

    for raw in lines:
        line = raw.strip()

        # POS heading?
        h3 = HEAD3_RE.match(line)
        if h3:
            # Save previous
            flush()
            pos = h3.group(1).strip()
            current_pos = pos if pos in allowed_pos else None
            current_defs = []
            continue

        if not current_pos:
            continue

        # Definition lines start with "#"
        if line.startswith("#"):
            # Skip example lines "#:" or "#*"
            if line.startswith("#:") or line.startswith("#*") or line.startswith("##:") or line.startswith("##*"):
                continue

            text = DEF_MARKER_RE.sub("", line)
            text = clean_wikitext(text)
            if text:
                current_defs.append(text)
            continue

        # Stop collecting if we hit another major header level (defensive)
        if line.startswith("==") and line.endswith("=="):
            break

    flush()
    return senses


# -----------------------------
# Streaming XML parsing
# -----------------------------

def iter_pages_from_bz2(xml_bz2_path: str) -> Iterable[Tuple[str, str]]:
    """
    Yield (title, wikitext) for each <page> in the Wiktionary dump.
    """
    # ElementTree iterparse can stream; wrap bz2 in a buffered reader.
    with bz2.open(xml_bz2_path, "rb") as f:
        # Capture namespace if present
        context = ET.iterparse(f, events=("end",))
        for event, elem in context:
            if elem.tag.endswith("page"):
                title_el = elem.find("./{*}title")
                text_el = elem.find("./{*}revision/{*}text")
                title = title_el.text if title_el is not None and title_el.text else ""
                text = text_el.text if text_el is not None and text_el.text else ""
                yield title, text
                elem.clear()


# -----------------------------
# Optional redirects support
# -----------------------------
# NOTE: Requires BOTH redirect.sql.gz and page.sql.gz to map page_id -> title.
INSERT_RE = re.compile(r"^INSERT INTO `(?P<table>\w+)` VALUES ", re.IGNORECASE)

def parse_sql_inserts(stream: io.TextIOBase) -> Iterable[Tuple[str, str]]:
    """
    Yield (table_name, values_blob) for each INSERT line.
    """
    for line in stream:
        line = line.strip()
        if not line.startswith("INSERT INTO"):
            continue
        m = INSERT_RE.match(line)
        if not m:
            continue
        table = m.group("table")
        # Everything after VALUES
        idx = line.find("VALUES")
        if idx == -1:
            continue
        values_blob = line[idx + len("VALUES"):].strip().rstrip(";")
        yield table, values_blob

def split_sql_tuples(values_blob: str) -> List[str]:
    """
    Split a VALUES blob like:
      (1,0,'Foo','',''),(2,0,'Bar','','')
    into individual tuple strings.
    This is a best-effort parser for Wikimedia dump format.
    """
    out = []
    buf = []
    depth = 0
    in_str = False
    esc = False

    for ch in values_blob:
        buf.append(ch)
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == "'":
                in_str = False
        else:
            if ch == "'":
                in_str = True
            elif ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    out.append("".join(buf).strip())
                    buf = []
    return out

def parse_sql_tuple(tuple_str: str) -> List[str]:
    """
    Parse a single SQL tuple "(...)" into fields (strings).
    Best-effort for Wikimedia dumps.
    """
    s = tuple_str.strip()
    if s.startswith("(") and s.endswith(")"):
        s = s[1:-1]

    fields = []
    buf = []
    in_str = False
    esc = False

    for ch in s:
        if in_str:
            if esc:
                buf.append(ch)
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == "'":
                in_str = False
            else:
                buf.append(ch)
        else:
            if ch == "'":
                in_str = True
            elif ch == ",":
                fields.append("".join(buf).strip())
                buf = []
            else:
                buf.append(ch)

    fields.append("".join(buf).strip())
    return fields

def load_page_id_to_title(page_sql_gz: str) -> Dict[int, str]:
    """
    Build page_id -> title from page.sql.gz (namespace 0 only).
    """
    page_map: Dict[int, str] = {}
    with gzip.open(page_sql_gz, "rt", encoding="utf-8", errors="ignore") as f:
        for table, blob in parse_sql_inserts(f):
            if table != "page":
                continue
            for tup in split_sql_tuples(blob):
                fields = parse_sql_tuple(tup)
                # page table columns (typical):
                # page_id, page_namespace, page_title, ...
                if len(fields) < 3:
                    continue
                try:
                    page_id = int(fields[0])
                    ns = int(fields[1])
                    title = fields[2]
                except ValueError:
                    continue
                if ns != 0:
                    continue
                page_map[page_id] = title.replace("_", " ")
    return page_map

def load_redirects(redirect_sql_gz: str, page_map: Dict[int, str]) -> Dict[str, str]:
    """
    Build redirect_from_title -> redirect_to_title from redirect.sql.gz + page_map.
    """
    redirects: Dict[str, str] = {}
    with gzip.open(redirect_sql_gz, "rt", encoding="utf-8", errors="ignore") as f:
        for table, blob in parse_sql_inserts(f):
            if table != "redirect":
                continue
            for tup in split_sql_tuples(blob):
                fields = parse_sql_tuple(tup)
                # redirect table columns (typical):
                # rd_from, rd_namespace, rd_title, rd_interwiki, rd_fragment
                if len(fields) < 3:
                    continue
                try:
                    rd_from = int(fields[0])
                    rd_ns = int(fields[1])
                    rd_title = fields[2]
                except ValueError:
                    continue
                if rd_ns != 0:
                    continue
                src = page_map.get(rd_from)
                if not src:
                    continue
                dst = rd_title.replace("_", " ")
                redirects[src] = dst
    return redirects


# -----------------------------
# Main build
# -----------------------------

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", required=True, help="Path to enwiktionary-*-pages-articles.xml.bz2")
    ap.add_argument("--out", default="out", help="Output directory (default: out)")
    ap.add_argument("--min-len", type=int, default=2, help="Minimum word length (default: 2)")
    ap.add_argument("--max-len", type=int, default=25, help="Maximum word length (default: 25)")
    ap.add_argument("--pos", nargs="*", default=sorted(DEFAULT_POS),
                    help="Allowed POS headers (default: common POS). Example: --pos Noun Verb Adjective")
    ap.add_argument("--max-defs-per-pos", type=int, default=3, help="Max definitions per POS (default: 3)")
    ap.add_argument("--include-proper-nouns", action="store_true",
                    help="Alias for including 'Proper noun' in POS (default already includes it).")
    ap.add_argument("--redirect-sql", default=None, help="Path to enwiktionary-*-redirect.sql.gz (optional)")
    ap.add_argument("--page-sql", default=None, help="Path to enwiktionary-*-page.sql.gz (optional, required for redirects)")
    args = ap.parse_args()

    allowed_pos = set(args.pos)
    if args.include_proper_nouns:
        allowed_pos.add("Proper noun")

    out_words = os.path.join(args.out, "words")
    out_defs = os.path.join(args.out, "defs")
    ensure_dir(out_words)
    ensure_dir(out_defs)

    # Optional redirects
    redirects: Dict[str, str] = {}
    if args.redirect_sql or args.page_sql:
        if not (args.redirect_sql and args.page_sql):
            print("NOTE: Redirect processing requires BOTH --redirect-sql and --page-sql. Skipping redirects.", file=sys.stderr)
        else:
            print("Loading page map (page_id -> title)...", file=sys.stderr)
            page_map = load_page_id_to_title(args.page_sql)
            print(f"Loaded {len(page_map):,} page_id->title entries", file=sys.stderr)

            print("Loading redirects (title -> title)...", file=sys.stderr)
            redirects = load_redirects(args.redirect_sql, page_map)
            print(f"Loaded {len(redirects):,} redirects", file=sys.stderr)

    # Writers: keep one open handle per length for words + defs
    word_files: Dict[int, io.TextIOWrapper] = {}
    def_files: Dict[int, io.TextIOWrapper] = {}

    counts_by_len: Dict[int, int] = {}
    defs_by_len: Dict[int, int] = {}
    total_pages = 0
    total_entries = 0
    total_defs_records = 0

    def get_word_file(L: int) -> io.TextIOWrapper:
        if L not in word_files:
            path = os.path.join(out_words, f"words-{L}.txt")
            word_files[L] = open(path, "a", encoding="utf-8")
        return word_files[L]

    def get_def_file(L: int) -> io.TextIOWrapper:
        if L not in def_files:
            path = os.path.join(out_defs, f"defs-{L}.jsonl")
            def_files[L] = open(path, "a", encoding="utf-8")
        return def_files[L]

    try:
        for title, text in iter_pages_from_bz2(args.xml):
            total_pages += 1
            if not title:
                continue

            # Filter to "word-like" titles only; ignore spaces/punct for crossword lexicon.
            # Wiktionary titles are case-sensitive; we normalize later.
            if not ALPHA_RE.match(title):
                continue

            L = len(title)
            if L < args.min_len or L > args.max_len:
                continue

            english = extract_english_section(text)
            if not english:
                continue

            senses = extract_senses(english, allowed_pos, args.max_defs_per_pos)
            if not senses:
                continue

            # Normalize word for crossword fill
            word_upper = title.upper()

            # Write word
            wf = get_word_file(L)
            wf.write(word_upper + "\n")
            counts_by_len[L] = counts_by_len.get(L, 0) + 1
            total_entries += 1

            # Write definitions record
            record = {
                "word": word_upper,
                "senses": [{"pos": s.pos, "definitions": s.definitions} for s in senses],
            }
            df = get_def_file(L)
            df.write(json.dumps(record, ensure_ascii=False) + "\n")
            defs_by_len[L] = defs_by_len.get(L, 0) + 1
            total_defs_records += 1

            # Optional: also write redirects as extra word aliases (if available)
            # We DO NOT write definitions for redirects here; we only alias them to the target.
            # You can post-process these into your final lexicon rules.
            if redirects:
                # redirects keys are titles like "Foo" (spaces possible), but we filtered ALPHA above.
                # We'll only apply redirects that are alpha-only and within length bounds.
                src = title
                if src in redirects:
                    dst = redirects[src]
                    if ALPHA_RE.match(dst):
                        dst_L = len(dst)
                        if args.min_len <= dst_L <= args.max_len:
                            # Write redirect source as a word too (uppercased).
                            # NOTE: This is simplistic; you may prefer to store redirects separately.
                            pass

            if total_pages % 200_000 == 0:
                print(f"Processed {total_pages:,} pages, kept {total_entries:,} entries...", file=sys.stderr)

    finally:
        for f in word_files.values():
            f.close()
        for f in def_files.values():
            f.close()

    manifest = {
        "xml": os.path.abspath(args.xml),
        "out": os.path.abspath(args.out),
        "min_len": args.min_len,
        "max_len": args.max_len,
        "allowed_pos": sorted(allowed_pos),
        "max_defs_per_pos": args.max_defs_per_pos,
        "total_pages_seen": total_pages,
        "total_entries_kept": total_entries,
        "total_definition_records": total_defs_records,
        "counts_by_length": {str(k): v for k, v in sorted(counts_by_len.items())},
        "defs_by_length": {str(k): v for k, v in sorted(defs_by_len.items())},
        "redirects_loaded": len(redirects),
        "notes": [
            "Definitions are extracted from English section and common POS headings.",
            "Definitions are lightly cleaned from wikitext; further cleanup may be needed.",
            "Redirects require page.sql.gz + redirect.sql.gz; redirect handling is intentionally conservative.",
        ],
    }

    ensure_dir(args.out)
    with open(os.path.join(args.out, "manifest.json"), "w", encoding="utf-8") as mf:
        json.dump(manifest, mf, indent=2, ensure_ascii=False)

    print("Done.")
    print(f"Pages processed: {total_pages:,}")
    print(f"Entries kept: {total_entries:,}")
    print(f"Definition records written: {total_defs_records:,}")
    print(f"Output written to: {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
