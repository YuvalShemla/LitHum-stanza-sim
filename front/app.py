import json
import numpy as np
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer
import time
from pathlib import Path
import pickle


BASE_DIR = Path(__file__).parent           # == front/

# ---------- helpers ------------------------------------------------
@st.cache_resource(show_spinner="Loading embedding model…")
def load_model():
    model_path = BASE_DIR / "models" / "all-mpnet-base-v2"
    return SentenceTransformer(str(model_path))

@st.cache_resource(show_spinner="Loading data…")
def load_data():
    poems_path      = BASE_DIR / "Ibn_Arabi_poems.json"
    embeddings_path = BASE_DIR / "Ibn_Arabi_poems_embeddings.npz"

    # ---- poems dict ------------------------------------------------
    with poems_path.open(encoding="utf-8") as f:
        poems = json.load(f)   # {title: [stanza1, stanza2, …]}

    # ---- stanza lookup table --------------------------------------
    stanza_lookup = []         # [(title, stanza_idx, stanza_txt), …]
    for title, stanzas in poems.items():
        stanza_lookup.extend(
            (title, i, s) for i, s in enumerate(stanzas)
        )

    # ---- embeddings ----------------------------------------------
    # npz assumes the array is stored under arr_0
    embeddings = np.load(embeddings_path, allow_pickle=True)["embeddings"]
    # If your .npz has a named array, use np.load(...)[<key>]

    return poems, stanza_lookup, embeddings

model = load_model()
poems, stanza_lookup, embeddings = load_data()

# ---- Similarity Search ----
def top_k_stanzas(query: str, k: int = 5, show_poem: bool = False):
    """
    Return the top‑k most similar stanzas (cosine‑sim) to `query`.

    Parameters
    ----------
    query : str
        Free‑text query.
    k : int, default 5
        Number of matches to return.
    show_poem : bool, default False
        If True, include the full poem text in the results.

    Returns
    -------
    pandas.DataFrame
        Columns: rank · similarity · title · stanza_id · stanza_text
        (+ full_poem when `show_poem` is True)
    """
    # -- 1. embed the query --------------------------------------------------
    q_vec = (
        model.encode(query,
                     convert_to_tensor=True,     # keep as torch.Tensor
                     normalize_embeddings=True)  # unit‑length
             .cpu()                            # ensure on CPU
             .numpy()                          # → ndarray (1, 768)
             .squeeze()                        # → shape (768,)
    )

    # dtype match (embeddings might be float64)
    if q_vec.dtype != embeddings.dtype:
        q_vec = q_vec.astype(embeddings.dtype, copy=False)

    # -- 2. cosine similarities ---------------------------------------------
    sims  = embeddings @ q_vec                # (N,768) · (768,) → (N,)
    best  = sims.argsort()[-k:][::-1]         # top‑k indices

    # -- 3. build results ----------------------------------------------------
    rows = []
    for rank, idx in enumerate(best, 1):
        score = float(sims[idx])
        title, stanza_idx, stanza_txt = stanza_lookup[idx]

        row = {
            "rank":        rank,
            "similarity":  f"{score:.3f}",
            "title":       title,
            "stanza_id":   stanza_idx,
            "stanza_text": stanza_txt,
        }
        if show_poem:
            row["full_poem"] = "\n".join(poems[title])

        rows.append(row)

    cols = ["rank", "similarity", "title", "stanza_id", "stanza_text"]
    if show_poem:
        cols.append("full_poem")

    return pd.DataFrame(rows)[cols]


# ---- Streamlit UI ----
st.title("Ibn Arabi Stanza Similarity Search")
st.markdown(
    """
    _This app finds the most contextually similar stanzas to your query, 
    using transformer sentence embeddings.
    Feel free to play with the settings (to the left) and try out different queries.
    Currently, the search is limited to "Translator of Desires", which contains 512 stanzas in 61 poems._
    """
)
with st.sidebar:
    st.markdown("### Settings")
    k         = st.slider("How many results?", 1, 20, 5)
    show_poem = st.checkbox("Include full poem", False)

query = st.text_input("Enter your query", placeholder="e.g. hopelessness")
if st.button("Search") and query.strip():
    t0 = time.time()
    df = top_k_stanzas(query.strip(), k, show_poem)
    st.success(f"Found in {time.time()-t0:.2f} s")
    if not df.empty:
        st.write(f"Top {k} matches:")
        st.dataframe(df, use_container_width=True)
    else:
        st.warning("No matches found.")
else:
    st.info("Type a query and press **Search**.")