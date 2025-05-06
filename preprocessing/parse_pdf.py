#!/usr/bin/env python3
"""
parse_pdf.py – Extract the 61 English poems from Ibn ʿArabi’s bilingual
*Translator of Desires* PDF.
(Everything is unchanged except locate_titles – see that section.)
"""

from __future__ import annotations
import argparse, json, re, unicodedata
from pathlib import Path
import pdfplumber

# ─────────────────────  canonical titles (book order)  ──────────────
ALL_TITLES = [
    "Bewildered", "Release", "The Offering",
    "As Night Let Down Its Curtain", "Harmony Gone",
    "Artemisia and Moringa", "Gowns of Dark", "Who Forever",
    "Soft- Eyed Graces", "Mirror", "Gentle Now, Doves", "Sunblaze",
    "Grief Between", "Hadith of Love", "Just a Flash", "Star Shepherd",
    "God Curse My Love", "As Cool as Life",
    "The Tombs of Those Who Loved Them", "In a Bad Way",
    "Your Wish", "Blacksilver", "Blaze", "Stay Now", "Vanished",
    "Vintage of Adam", "Old Shrine", "No New Moon Risen", "Circling",
    "Like Sába’s Lost Tribes", "Áda Trail", "Fifty Years",
    "Drowning Eyes", "Sweet the Gaze", "You Now",
    "Lightning over Gháda", "Come Down to the Waters", "A Persian Girl",
    "Day Falls Night", "Odd to Even", "My Only Guide", "Sign Masters",
    "Parting’s Hour", "Chimera", "Where Gone", "Cool Lightning",
    "The Turning", "Brave", "In the Ruins of My Body", "Done",
    "Nightwalker", "Rídwa", "Like a Doubled Letter", "Dár al- Fálak",
    "No Cure", "Baghdad Song", "Red Rise", "Is There a Way",
    "I Came to Know", "Artemisia and Arâr", "Tigris Song",
]

# ─────────────────────  normalisation helpers  ──────────────────────
DASHES = r'[‑‒–—−]'
QUOTES = {"’": "'", "‘": "'", "ʼ": "'"}

def strip_diacritics(s: str) -> str:
    return ''.join(ch for ch in unicodedata.normalize('NFKD', s)
                   if not unicodedata.combining(ch))

def normalize(s: str) -> str:
    s = strip_diacritics(s)
    s = re.sub(DASHES, '-', s)
    s = re.sub(r'\s*-\s*', '-', s)        # remove spaces around dashes
    for bad, good in QUOTES.items():
        s = s.replace(bad, good)
    s = re.sub(r'\s+', ' ', s)
    return s.lower().strip()

# Map normalised form → canonical title
NORM2CANON = {normalize(t): t for t in ALL_TITLES}

# ─────────────────────  cleaning regexes  ───────────────────────────
ARABIC_RE   = re.compile(r'[\u0600-\u06FF]')
DIGITS_RE   = re.compile(r'^\s*\d+\s*$')
TITLE_DIGIT = re.compile(r'\s+\d+\s*$')
REF_RE      = re.compile(r'\s*\[\s*\d+\s*\]\s*')
SOFT_HYPH   = re.compile(r'-\n(\w)')

# ─────────────────────  1 ▸ extract & clean  ────────────────────────
def clean_lines(raw: str) -> list[str]:
    out = []
    for ln in raw.splitlines():
        if ARABIC_RE.search(ln) or DIGITS_RE.fullmatch(ln):
            continue
        ln = TITLE_DIGIT.sub('', ln)
        ln = REF_RE.sub(' ', ln).rstrip()
        if ln:
            out.append(strip_diacritics(ln))
    return SOFT_HYPH.sub(r'\1', '\n'.join(out)).splitlines()

# ─────────────────────  2 ▸ locate headers  (only change)  ──────────
def locate_titles(lines: list[str]) -> list[tuple[str, int]]:
    """
    Scan once through the text.  A line is accepted as a header **only**
    if it matches the *next* expected title in book order *and* is no
    longer than the title plus three chars (to allow a page‑number
    suffix).  This skips any title phrase that appears out of order
    inside the Translator’s Introduction.
    """
    positions     : list[tuple[str, int]] = []
    expect_index  = 0                                   # next poem to find
    char_pos      = 0

    for ln in lines:
        if expect_index >= len(ALL_TITLES):
            break
        candidate = ALL_TITLES[expect_index]
        if normalize(ln) == normalize(candidate):
            raw = ln.strip()
            if len(raw) <= len(candidate) + 3:
                positions.append((candidate, char_pos))
                expect_index += 1                      # move to next poem
        char_pos += len(ln) + 1                        # +1 for '\n'

    return positions

# ─────────────────────  3 ▸ slice & stanza‑ise  ─────────────────────
def slice_poems(text: str, pos: list[tuple[str,int]]) -> dict[str,str]:
    poems = {}
    for i, (title, start) in enumerate(pos):
        body_start = text.find('\n', start) + 1
        body_end   = pos[i+1][1] if i+1 < len(pos) else len(text)
        poems[title] = text[body_start:body_end].strip()
    return poems

def to_stanzas(poem: str, title: str, n: int = 4) -> list[str]:
    t_norm = normalize(title)
    lines  = [l for l in poem.splitlines()
              if l.strip() and normalize(l) != t_norm]   # drop stray header
    return ['\n'.join(lines[i:i+n]) for i in range(0, len(lines), n)]

# ─────────────────────  main driver  ────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('pdf', type=Path)
    ap.add_argument('-o', '--out', type=Path,
                    help='output JSON (default: <PDF>_poems.json)')
    ap.add_argument('-n', '--lines', type=int, default=4,
                    help='lines per stanza (default 4)')
    args = ap.parse_args()

    if not args.pdf.exists():
        ap.error(f"{args.pdf} not found")

    with pdfplumber.open(args.pdf) as pdf:
        raw = '\n'.join(p.extract_text() or '' for p in pdf.pages)

    lines      = clean_lines(raw)
    full_text  = '\n'.join(lines)
    headers    = locate_titles(lines)
    poems_raw  = slice_poems(full_text, headers)
    poems      = {t: to_stanzas(txt, t, args.lines) for t, txt in poems_raw.items()}

    out_path = args.out or args.pdf.with_name(args.pdf.stem + '_poems.json')
    out_path.write_text(json.dumps(poems, ensure_ascii=False, indent=2), 'utf-8')
    print(f"✔ {len(poems)} poems → {out_path}")

if __name__ == '__main__':
    main()
