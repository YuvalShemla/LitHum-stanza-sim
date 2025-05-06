#!/usr/bin/env python3
"""
make_embeddings.py  –  Add sentence‑transformer embeddings to each stanza
                       in poems.json.

output   poems_embeddings.npz  with three arrays:
    titles        shape (N,)
    stanza_idx    shape (N,)
    embeddings    shape (N, dim)
"""

import argparse, json, numpy as np
from pathlib import Path
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# ── CLI ─────────────────────────────────────────────────────────────
ap = argparse.ArgumentParser()
ap.add_argument("json_file", type=Path,
                help="poems.json produced by parse_pdf.py")
ap.add_argument("-m", "--model", default="all-MiniLM-L6-v2",
                help="sentence‑transformers model name")
ap.add_argument("-o", "--out",   type=Path,
                help="output .npz (default <JSON>_embeddings.npz)")
args = ap.parse_args()

# ── load poems ------------------------------------------------------
poems = json.loads(args.json_file.read_text("utf-8"))

rows = []                           # (title, stanza_idx, text)
for title, stanzas in poems.items():
    for i, stanza in enumerate(stanzas):
        rows.append((title, i, stanza))

titles, idxs, texts = zip(*rows)    # unzip
print(f"→ {len(texts)} stanzas to embed")

# ── encode ----------------------------------------------------------
model = SentenceTransformer(args.model)
emb   = model.encode(
    list(texts),
    batch_size=64,
    show_progress_bar=True,
    normalize_embeddings=True       # cosine similarity = dot product
).astype(np.float32)                # reduce storage

# ── save ------------------------------------------------------------
out_path = (
    args.out or args.json_file.with_name(args.json_file.stem + "_embeddings.npz")
)
np.savez_compressed(
    out_path,
    titles=np.array(titles),
    stanza_idx=np.array(idxs, dtype=np.int32),
    embeddings=emb,
)
print(f"✔ saved embeddings → {out_path}  (shape {emb.shape})")
