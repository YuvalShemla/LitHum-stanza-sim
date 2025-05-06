import json
import numpy as np
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer
import time

# ---- Paths ----
NPZ_PATH   = "Ibn_Arabi_poems_embeddings.npz"
POEMS_JSON = "Ibn_Arabi_poems.json"

# ---- Load Data ----
@st.cache_resource(show_spinner="Loading transformer model…")
def load_model():
    return SentenceTransformer("models/all-mpnet-base-v2")

@st.cache_data(show_spinner="Loading embeddings…")
def load_data():
    poems = json.loads(open(POEMS_JSON, encoding="utf-8").read())
    stanza_lookup = []
    for title, stanzas in poems.items():
        for i, s in enumerate(stanzas):
            stanza_lookup.append((title, i, s))
    data = np.load(NPZ_PATH, allow_pickle=True)
    embeddings = data["embeddings"]
    return poems, stanza_lookup, embeddings

model = load_model()
poems, stanza_lookup, embeddings = load_data()

# ---- Similarity Search ----
def top_k_stanzas(query: str, k: int = 5, show_poem: bool = False):
    q_vec = model.encode(query, normalize_embeddings=True)
    sims  = embeddings @ q_vec
    best  = sims.argsort()[-k:][::-1]

    rows = []
    for rank, idx in enumerate(best, 1):
        score = float(sims[idx])
        title, stanza_idx, stanza_txt = stanza_lookup[idx]
        row = dict(
            rank        = rank,
            similarity  = f"{score:.3f}",
            title       = title,
            stanza_id  = stanza_idx,
            stanza_text = stanza_txt,
        )
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