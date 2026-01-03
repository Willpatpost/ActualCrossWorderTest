#!/usr/bin/env python3
"""
Build a crossword-friendly lexicon from the English Wiktionary XML dump.

Input:
  - enwiktionary-YYYYMMDD-pages-articles.xml.bz2

Output (default out/):
  - out/words/words-{L}.txt           (uppercase A-Z words by length)
  - out/defs/defs-{L}.jsonl           (JSON Lines: {"word":"APPLE","senses":[...]} )
  - out/manifest.json                 (counts and configuration)

Notes:
  - Streaming parse; does NOT load the whole dump into memory.
  - Improves on naive template stripping by rendering common templates.
  - Filters out junk definitions like "." that arise from template-only senses.

Usage:
  python build_wiktionary_lexicon.py --xml enwiktionary-20260101-pages-articles.xml.bz2 --out out

Optional redirects support (requires BOTH):
  --redirect-sql enwiktionary-20260101-redirect.sql.gz
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
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple


# -----------------------------
# Regex & constants
# -----------------------------

ALPHA_RE = re.compile(r"^[A-Za-z]+$")

# Level 2 headers: ==English==
HEAD2_RE = re.compile(r"^==\s*([^=]+?)\s*==\s*$")

# POS headings appear commonly as level 3, sometimes level 4 under Etymology sections
HEAD3_RE = re.compile(r"^===\s*([^=]+?)\s*===\s*$")
HEAD4_RE = re.compile(r"^====\s*([^=]+?)\s*====\s*$")

# Definition markers: "#", "##", etc.
DEF_MARKER_RE = re.compile(r"^#+\s*")

# Skip example/usage lines
EXAMPLE_PREFIXES = ("#:", "#*", "##:", "##*")

DEFAULT_POS = {
    "Noun",
    "Verb",
    "Adjective",
    "Adverb",
    "Proper noun",
}

BAD_LINK_PREFIXES = ("File:", "Image:", "Category:")

# Remove <ref>...</ref> and self-closing <ref .../>
REF_BLOCK_RE = re.compile(r"<ref\b[^>/]*?>.*?</ref>", re.IGNORECASE | re.DOTALL)
REF_SELF_RE = re.compile(r"<ref\b[^>]*/\s*>", re.IGNORECASE)

# Remove HTML comments
HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)

# Remove math/nowiki blocks entirely
NOWIKI_RE = re.compile(r"<nowiki\b[^>]*>.*?</nowiki>", re.IGNORECASE | re.DOTALL)
MATH_RE = re.compile(r"<math\b[^>]*>.*?</math>", re.IGNORECASE | re.DOTALL)

# Remove any remaining tags
TAG_RE = re.compile(r"</?[^>]+>")

# Wikilinks [[...]] or [[...|...]]
WIKILINK_RE = re.compile(r"\[\[([^|\]]+)(?:\|([^\]]+))?\]\]")

# Bold/italic markup '' '' ''' '''
ITALIC_BOLD_RE = re.compile(r"('{2,5})(.*?)\1")

# Basic table removal (wikitext tables can be multiline)
WIKITABLE_RE = re.compile(r"\{\|.*?\|\}", re.DOTALL)

# Cleanup whitespace
WS_RE = re.compile(r"\s+")

# “Meaningful text” check: if removing non-alphanumerics leaves nothing, it’s junk.
MEANINGFUL_RE = re.compile(r"[A-Za-z0-9]")


# -----------------------------
# Template parsing (nested-safe)
# -----------------------------

def split_template_parts(s: str) -> List[str]:
    """
    Split template body into parts separated by |, ignoring nested {{ }} and [[ ]].
    Example: "abbreviation of|en|International Business Machines" -> ["abbreviation of","en","International Business Machines"]
    """
    parts: List[str] = []
    buf: List[str] = []
    i = 0
    depth_tpl = 0
    depth_link = 0
    while i < len(s):
        ch = s[i]
        # Track template nesting
        if s.startswith("{{", i):
            depth_tpl += 1
            buf.append("{{")
            i += 2
            continue
        if s.startswith("}}", i) and depth_tpl > 0:
            depth_tpl -= 1
            buf.append("}}")
            i += 2
            continue
        # Track wikilink nesting
        if s.startswith("[[", i):
            depth_link += 1
            buf.append("[[")
            i += 2
            continue
        if s.startswith("]]", i) and depth_link > 0:
            depth_link -= 1
            buf.append("]]")
            i += 2
            continue

        if ch == "|" and depth_tpl == 0 and depth_link == 0:
            parts.append("".join(buf).strip())
            buf = []
            i += 1
            continue

        buf.append(ch)
        i += 1

    parts.append("".join(buf).strip())
    return parts


def render_template(name: str, args: List[str]) -> str:
    """
    Render common Wiktionary templates to human-readable plain text.
    For unknown templates, keep the most useful argument (often arg0/arg1), or return "".
    """
    n = name.strip().lower()

    # Helper: get first non-empty arg
    def first_arg(*idxs: int) -> str:
        for ix in idxs:
            if 0 <= ix < len(args):
                v = args[ix].strip()
                if v:
                    return v
        return ""

    # Some templates have "en" as first arg
    def drop_lang_prefix(a: List[str]) -> List[str]:
        if a and a[0].strip().lower() in ("en", "eng", "english"):
            return a[1:]
        return a

    args2 = drop_lang_prefix(args)

    # Label/qualifier templates: we can keep them as parentheses
    if n in ("lb", "lbl", "label", "labels"):
        # {{lb|en|obsolete|slang}} -> "(obsolete, slang)"
        labs = [x.strip() for x in args2 if x.strip()]
        if labs:
            return "(" + ", ".join(labs[:6]) + ")"
        return ""

    # Common “relation” templates
    if n in ("abbreviation of", "abbr of"):
        tgt = first_arg(0) or first_arg(1)
        return f"Abbreviation of {tgt}".strip()

    if n in ("initialism of", "init of"):
        tgt = first_arg(0) or first_arg(1)
        return f"Initialism of {tgt}".strip()

    if n in ("acronym of",):
        tgt = first_arg(0) or first_arg(1)
        return f"Acronym of {tgt}".strip()

    if n in ("alternative form of", "alternative spelling of", "alt form", "alt spelling"):
        tgt = first_arg(0) or first_arg(1)
        return f"Alternative form of {tgt}".strip()

    if n in ("misspelling of",):
        tgt = first_arg(0) or first_arg(1)
        return f"Misspelling of {tgt}".strip()

    # Inflection templates
    if n in ("plural of", "pl of"):
        tgt = first_arg(0) or first_arg(1)
        return f"Plural of {tgt}".strip()

    if n in ("past of", "past tense of"):
        tgt = first_arg(0) or first_arg(1)
        return f"Past tense of {tgt}".strip()

    if n in ("present participle of", "pres part of"):
        tgt = first_arg(0) or first_arg(1)
        return f"Present participle of {tgt}".strip()

    if n in ("comparative of",):
        tgt = first_arg(0) or first_arg(1)
        return f"Comparative of {tgt}".strip()

    if n in ("superlative of",):
        tgt = first_arg(0) or first_arg(1)
        return f"Superlative of {tgt}".strip()

    # Proper noun-ish templates
    if n in ("surname", "given name"):
        return name.capitalize()

    # Taxonomy-ish templates: keep the displayed name (often first arg)
    if n in ("taxlink", "taxfmt", "taxon", "taxlite"):
        return first_arg(0)

    # Link-ish templates:
    # {{l|en|word}} or {{link|word}} -> "word"
    if n in ("l", "link", "m", "mention"):
        # l|en|WORD typically: args2[0] is the word
        return first_arg(0) if args2 else first_arg(0)

    # Gloss templates: {{gloss|...}} -> "(...)" or just the gloss
    if n in ("gloss",):
        g = first_arg(0)
        return f"({g})" if g else ""

    # Unknown template fallback:
    # Often first useful payload is first non-empty arg (post language).
    # This is better than deleting everything (which creates "." garbage).
    payload = ""
    if args2:
        payload = first_arg(0)
    else:
        payload = first_arg(0)
    return payload


def replace_templates_nested(s: str, max_passes: int = 4) -> str:
    """
    Replace templates {{...}} with rendered text, handling nesting safely.
    Multiple passes allow inner templates to be resolved first.
    """
    if "{{" not in s:
        return s

    for _ in range(max_passes):
        out: List[str] = []
        i = 0
        changed = False

        while i < len(s):
            if s.startswith("{{", i):
                # find matching }}
                depth = 0
                j = i
                while j < len(s):
                    if s.startswith("{{", j):
                        depth += 1
                        j += 2
                        continue
                    if s.startswith("}}", j):
                        depth -= 1
                        j += 2
                        if depth == 0:
                            break
                        continue
                    j += 1

                if depth == 0:
                    body = s[i + 2 : j - 2].strip()
                    parts = split_template_parts(body)
                    name = parts[0] if parts else ""
                    args = parts[1:] if len(parts) > 1 else []
                    rendered = render_template(name, args)
                    out.append(rendered)
                    i = j
                    changed = True
                    continue
                # If we didn't find a close, treat as literal
            out.append(s[i])
            i += 1

        s2 = "".join(out)
        s2 = WS_RE.sub(" ", s2).strip()
        s = s2
        if not changed:
            break

    return s


# -----------------------------
# Wikitext cleaning
# -----------------------------

def clean_wikitext(s: str) -> str:
    """Pragmatic cleanup to make definitions readable, without deleting meaning."""
    if not s:
        return ""

    s = HTML_COMMENT_RE.sub("", s)
    s = REF_BLOCK_RE.sub("", s)
    s = REF_SELF_RE.sub("", s)
    s = NOWIKI_RE.sub("", s)
    s = MATH_RE.sub("", s)

    # Remove tables (often noisy)
    s = WIKITABLE_RE.sub("", s)

    # Render templates (nested)
    s = replace_templates_nested(s)

    # Replace wikilinks [[...]] / [[...|...]]
    def _link_sub(m: re.Match) -> str:
        target = (m.group(1) or "").strip()
        text = (m.group(2) or "").strip()
        chosen = text if text else target
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
    s = WS_RE.sub(" ", s).strip()

    # Normalize stray punctuation spacing
    s = s.replace(" )", ")").replace("( ", "(")
    s = re.sub(r"\s+([,;:.!?])", r"\1", s)

    # If it's just punctuation after cleanup, treat as empty
    if not MEANINGFUL_RE.search(s):
        return ""

    return s


# -----------------------------
# Section / senses extraction
# -----------------------------

@dataclass
class Sense:
    pos: str
    definitions: List[str]


def extract_english_section(wikitext: str) -> Optional[str]:
    """Return the substring for the ==English== section only."""
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

    end = len(lines)
    for j in range(start, len(lines)):
        m = HEAD2_RE.match(lines[j].strip())
        if m:
            end = j
            break

    return "\n".join(lines[start:end])


def extract_senses(english_text: str, allowed_pos: set[str], max_defs_per_pos: int) -> List[Sense]:
    """
    Extract senses from POS headings (===POS=== or ====POS====) and definition lines starting with '#'.
    Works across multiple Etymology subsections.
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
        if not line:
            continue

        # POS heading?
        h3 = HEAD3_RE.match(line)
        h4 = HEAD4_RE.match(line)
        if h3 or h4:
            flush()
            pos = (h3.group(1) if h3 else h4.group(1)).strip()
            current_pos = pos if pos in allowed_pos else None
            current_defs = []
            continue

        if not current_pos:
            continue

        # Definition lines
        if line.startswith("#"):
            if any(line.startswith(p) for p in EXAMPLE_PREFIXES):
                continue

            text = DEF_MARKER_RE.sub("", line)
            text = clean_wikitext(text)

            # After full cleanup, drop non-meaningful entries (prevents "." cases)
            if text and MEANINGFUL_RE.search(text):
                current_defs.append(text)
            continue

        # Defensive: stop collecting if we hit a higher-level header
        # (e.g., ==Something==)
        if line.startswith("==") and line.endswith("=="):
            break

    flush()
    return senses


# -----------------------------
# Streaming XML parsing
# -----------------------------

def iter_pages_from_bz2(xml_bz2_path: str) -> Iterable[Tuple[str, str]]:
    """Yield (title, wikitext) for each <page> in the Wiktionary dump."""
    with bz2.open(xml_bz2_path, "rb") as f:
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
# Optional redirects support (unchanged; conservative)
# -----------------------------

INSERT_RE = re.compile(r"^INSERT INTO `(?P<table>\w+)` VALUES ", re.IGNORECASE)

def parse_sql_inserts(stream: io.TextIOBase) -> Iterable[Tuple[str, str]]:
    for line in stream:
        line = line.strip()
        if not line.startswith("INSERT INTO"):
            continue
        m = INSERT_RE.match(line)
        if not m:
            continue
        table = m.group("table")
        idx = line.find("VALUES")
        if idx == -1:
            continue
        values_blob = line[idx + len("VALUES"):].strip().rstrip(";")
        yield table, values_blob

def split_sql_tuples(values_blob: str) -> List[str]:
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
    page_map: Dict[int, str] = {}
    with gzip.open(page_sql_gz, "rt", encoding="utf-8", errors="ignore") as f:
        for table, blob in parse_sql_inserts(f):
            if table != "page":
                continue
            for tup in split_sql_tuples(blob):
                fields = parse_sql_tuple(tup)
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
    redirects: Dict[str, str] = {}
    with gzip.open(redirect_sql_gz, "rt", encoding="utf-8", errors="ignore") as f:
        for table, blob in parse_sql_inserts(f):
            if table != "redirect":
                continue
            for tup in split_sql_tuples(blob):
                fields = parse_sql_tuple(tup)
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
# Progress reporting
# -----------------------------

def format_seconds(sec: float) -> str:
    sec = max(0.0, sec)
    m, s = divmod(int(sec), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"

def progress_bar(frac: float, width: int = 28) -> str:
    frac = 0.0 if frac < 0 else 1.0 if frac > 1 else frac
    filled = int(frac * width)
    return "[" + "#" * filled + "-" * (width - filled) + "]"

def best_effort_compressed_pos(bz2_file_obj) -> Optional[int]:
    """
    Try to get the underlying compressed file position to compute % done.
    This relies on implementation details; if it fails, we return None.
    """
    try:
        # bz2.BZ2File typically has ._fp (the underlying file object)
        fp = getattr(bz2_file_obj, "_fp", None)
        if fp and hasattr(fp, "tell"):
            return fp.tell()
    except Exception:
        return None
    return None


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
                    help="Allowed POS headers (default: common POS).")
    ap.add_argument("--max-defs-per-pos", type=int, default=3, help="Max definitions per POS (default: 3)")
    ap.add_argument("--redirect-sql", default=None, help="Path to enwiktionary-*-redirect.sql.gz (optional)")
    ap.add_argument("--page-sql", default=None, help="Path to enwiktionary-*-page.sql.gz (optional, required for redirects)")
    ap.add_argument("--overwrite", action="store_true",
                    help="Overwrite existing output files instead of appending.")
    ap.add_argument("--dedupe", action="store_true",
                    help="Dedupe words per length (small memory overhead; helpful if you rerun without --overwrite).")
    ap.add_argument("--progress-every", type=int, default=200_000,
                    help="Print progress every N pages processed (default: 200000)")
    args = ap.parse_args()

    allowed_pos = set(args.pos)

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

    # Optional in-memory dedupe per length
    seen_words_by_len: Dict[int, set[str]] = {} if args.dedupe else {}

    counts_by_len: Dict[int, int] = {}
    defs_by_len: Dict[int, int] = {}
    total_pages = 0
    total_entries = 0
    total_defs_records = 0

    mode = "w" if args.overwrite else "a"

    def get_word_file(L: int) -> io.TextIOWrapper:
        if L not in word_files:
            path = os.path.join(out_words, f"words-{L}.txt")
            word_files[L] = open(path, mode, encoding="utf-8", buffering=1024 * 1024)
        return word_files[L]

    def get_def_file(L: int) -> io.TextIOWrapper:
        if L not in def_files:
            path = os.path.join(out_defs, f"defs-{L}.jsonl")
            def_files[L] = open(path, mode, encoding="utf-8", buffering=1024 * 1024)
        return def_files[L]

    xml_abs = os.path.abspath(args.xml)
    compressed_size = None
    try:
        compressed_size = os.path.getsize(xml_abs)
    except OSError:
        compressed_size = None

    start_time = time.time()
    last_report = start_time

    try:
        # We want compressed byte progress, so we open bz2 ourselves here for best-effort tell()
        with bz2.open(args.xml, "rb") as f:
            context = ET.iterparse(f, events=("end",))
            for event, elem in context:
                if not elem.tag.endswith("page"):
                    continue

                total_pages += 1

                title_el = elem.find("./{*}title")
                text_el = elem.find("./{*}revision/{*}text")
                title = title_el.text if title_el is not None and title_el.text else ""
                text = text_el.text if text_el is not None and text_el.text else ""
                elem.clear()

                if not title:
                    continue

                # Only alpha-only titles (crossword-friendly)
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

                word_upper = title.upper()

                if args.dedupe:
                    sset = seen_words_by_len.setdefault(L, set())
                    if word_upper in sset:
                        continue
                    sset.add(word_upper)

                # Write word
                wf = get_word_file(L)
                wf.write(word_upper + "\n")
                counts_by_len[L] = counts_by_len.get(L, 0) + 1
                total_entries += 1

                # Write definitions record (compact JSON)
                record = {
                    "word": word_upper,
                    "senses": [{"pos": s.pos, "definitions": s.definitions} for s in senses],
                }
                df = get_def_file(L)
                df.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
                defs_by_len[L] = defs_by_len.get(L, 0) + 1
                total_defs_records += 1

                # Progress
                if total_pages % args.progress_every == 0:
                    now = time.time()
                    elapsed = now - start_time
                    rate = total_pages / elapsed if elapsed > 0 else 0.0

                    # Best-effort % based on compressed bytes consumed
                    pos = best_effort_compressed_pos(f)
                    frac = None
                    if compressed_size and pos is not None and compressed_size > 0:
                        frac = min(1.0, max(0.0, pos / compressed_size))

                    # ETA from compressed bytes (best-effort) else unknown
                    eta = ""
                    if frac is not None and frac > 0 and frac < 1:
                        eta_sec = elapsed * (1 - frac) / frac
                        eta = f" ETA {format_seconds(eta_sec)}"

                    bar = progress_bar(frac) + f" {frac*100:5.1f}%" if frac is not None else "[working…]"
                    print(
                        f"{bar}  pages {total_pages:,}  kept {total_entries:,}  "
                        f"{rate:,.0f} pages/s  elapsed {format_seconds(elapsed)}{eta}",
                        file=sys.stderr,
                    )

    finally:
        for f in word_files.values():
            try: f.close()
            except Exception: pass
        for f in def_files.values():
            try: f.close()
            except Exception: pass

    manifest = {
        "xml": xml_abs,
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
            "Definitions extracted from ==English== section under POS headings (===POS=== or ====POS====).",
            "Templates are rendered for common patterns rather than deleted; nested templates handled.",
            "Junk definitions (e.g. '.') are filtered out after cleanup.",
            "Progress percent/ETA are best-effort based on compressed bytes consumed.",
        ],
    }

    ensure_dir(args.out)
    with open(os.path.join(args.out, "manifest.json"), "w", encoding="utf-8") as mf:
        json.dump(manifest, mf, indent=2, ensure_ascii=False)

    print("Done.", file=sys.stderr)
    print(f"Pages processed: {total_pages:,}", file=sys.stderr)
    print(f"Entries kept: {total_entries:,}", file=sys.stderr)
    print(f"Definition records written: {total_defs_records:,}", file=sys.stderr)
    print(f"Output written to: {os.path.abspath(args.out)}", file=sys.stderr)


if __name__ == "__main__":
    main()
