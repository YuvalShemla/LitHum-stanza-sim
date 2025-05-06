#!/usr/bin/env python3
"""
check_npz.py – sanity‑check poems_embeddings.npz

usage
-----
    python check_npz.py poems_embeddings.npz
"""

from __future__ import annotations
import argparse, sys
import numpy as np
from sentence_transformers import SentenceTransformer

# ---------- config --------------------------------------------------
TEST_LINE   = "In a rush of saffron and musk\nbeauty falls bewildered"
EXPECT_TITLE = "Bewildered"
EXPECT_STANZA = "beauty falls bewildered"  # short substring to check
MODEL_NAME  = "all-MiniLM-L6-v2"           # keep in sync with make_embeddings
TOP_K       = 3                            # show top‑k results
THRESH      = 0.50                         # similarity threshold for pass
# --------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", help=".npz produced by make_embeddings.py")
    args = ap.parse_args()

    data  = np.load(args.npz, allow_pickle=True)
    titles     = data["titles"]
    stanza_idx = data["stanza_idx"]
    embeds     = data["embeddings"]

    model = SentenceTransformer(MODEL_NAME)
    q_vec = model.encode(TEST_LINE, normalize_embeddings=True)

    sims  = embeds @ q_vec                # cosine similarity
    best  = sims.argsort()[::-1][:TOP_K]   # top‑k indices

    print(f"Query line:\n  {TEST_LINE}\n")
    print("Top matches:")
    ok = False
    for rank, i in enumerate(best, 1):
        score = float(sims[i])
        title = titles[i]
        idx   = int(stanza_idx[i])
        print(f"{rank:>2}.  {score: .3f}  |  {title}  (stanza {idx})")
        if rank == 1 and score >= THRESH and title == EXPECT_TITLE:
            ok = True

    if ok:
        print("\n✅ PASS – Embeddings look good.")
        sys.exit(0)
    else:
        print("\n❌ FAIL – Expected top hit to be "
              f"'{EXPECT_TITLE}' with cosine ≥ {THRESH}.")
        sys.exit(1)

if __name__ == "__main__":
    main()
