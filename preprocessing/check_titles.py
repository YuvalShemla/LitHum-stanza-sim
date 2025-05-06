#!/usr/bin/env python3
"""
check_titles.py – Verify that the poem‑dict JSON contains every expected title
                  from Ibn ʿArabi’s *Translator of Desires*.

usage
-----
    python check_titles.py Ibn_Arabi_poems.json
"""

from __future__ import annotations
import argparse, json, sys
from pathlib import Path

ALL_TITLES: list[str] = [
    "Bewildered", "Release", "The Offering",
    "As Night Let Down Its Curtain", "Harmony Gone",
    "Artemisia and Moringa", "Gowns of Dark", "Who Forever",
    "Soft‑Eyed Graces", "Mirror", "Gentle Now, Doves", "Sunblaze",
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
    "Nightwalker", "Rídwa", "Like a Doubled Letter", "Dár al‑Fálak",
    "No Cure", "Baghdad Song", "Red Rise", "Is There a Way",
    "I Came to Know", "Artemisia and Arâr", "Tigris Song",
]

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('json_path', type=Path, help='poems JSON from parse_pdf.py')
    args = ap.parse_args()

    if not args.json_path.exists():
        ap.error(f'{args.json_path} not found')

    data = json.loads(args.json_path.read_text('utf-8'))

    expected = set(ALL_TITLES)
    found    = set(data.keys())

    missing = sorted(expected - found)
    extra   = sorted(found - expected)

    if not missing and not extra:
        print("✅ All 61 titles present – nothing missing, nothing extra.")
        sys.exit(0)

    if missing:
        print("⚠️  Missing titles:")
        for t in missing:
            print(f"   • {t}")
    if extra:
        print("\n⚠️  Unexpected titles in JSON:")
        for t in extra:
            print(f"   • {t}")

if __name__ == '__main__':
    main()
