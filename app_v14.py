# app_v14.py
import os
import time
import streamlit as st
from dotenv import load_dotenv

from chroma_utils import (
    get_persistent_client,
    get_or_create_collection,
    chroma_stats,
    clear_collection,
    reset_chroma,
    DEFAULT_PATH,
    DEFAULT_COLLECTION,
)

from llama_utils import build_index_from_folder, query_index

# ============ Boot ============
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="RAG Assistant — v14", layout="wide")
st.title("🧠 RAG Assistant — v14 (Modular)")

# Sidebar Header
with st.sidebar:
    st.header("🔌 Connections")
    if OPENAI_API_KEY:
        st.success("OpenAI: connected")
    else:
        st.warning("OpenAI key missing (.env)")

    st.header("💾 Vector DB")
    client = get_persistent_client(DEFAULT_PATH)
    st.info("Using local Chroma (persistent)")

    # --- Storage Inspector ---
    st.subheader("🧪 Storage Inspector")
    stats = chroma_stats(DEFAULT_PATH, client)
    c1, c2 = st.columns(2)
    c1.metric("DB Size (MB)", stats["db_size_mb"])
    c2.metric("# Collections", stats["collections"])
    c1.metric("# Embeddings", stats["embeddings"])
    c2.write(f"Path: `{stats['db_path']}`")

    colA, colB = st.columns(2)
    if colA.button("🔄 Refresh Stats"):
        st.rerun()
    if colB.button("♻️ Reset DB (delete folder)", type="secondary"):
        reset_chroma(DEFAULT_PATH)
        st.toast("Chroma folder deleted. Restarting client...")
        time.sleep(0.8)
        st.rerun()

# ============ Main Tabs ============
tab_chat, tab_metrics, tab_settings = st.tabs(["💬 Chat", "📊 Metrics", "⚙️ Settings"])

# ----------------- Chat Tab -----------------
with tab_chat:
    st.subheader("Chat with your Documents")
    docs_col, actions_col = st.columns([3, 2])

    with docs_col:
        uploaded = st.file_uploader(
            "Upload documents (.txt, .pdf, .docx)", type=["txt", "pdf", "docx"], accept_multiple_files=True
        )
        docs_folder = "docs"
        os.makedirs(docs_folder, exist_ok=True)

        if uploaded:
            for f in uploaded:
                with open(os.path.join(docs_folder, f.name), "wb") as out:
                    out.write(f.read())
            st.success(f"Saved {len(uploaded)} file(s) to /{docs_folder}")

    with actions_col:
        st.markdown("#### Index Actions")
        if st.button("🧠 Rebuild Index"):
            try:
                _, _ = build_index_from_folder(
                    docs_folder, chroma_client=client, collection_name=DEFAULT_COLLECTION
                )
                st.success("Rebuilt index from /docs into Chroma.")
            except Exception as e:
                st.error(f"Index rebuild failed: {e}")

        if st.button("🗑️ Clear 'rag' Collection"):
            clear_collection(client, DEFAULT_COLLECTION)
            st.info("Cleared 'rag' collection. You can rebuild to repopulate.")

    st.divider()
    st.markdown("#### Ask a question")
    user_q = st.text_input("Your question", placeholder="e.g., Summarize the contents of my resume.")
    if st.button("Ask") and user_q:
        try:
            ans = query_index(user_q, chroma_client=client, collection_name=DEFAULT_COLLECTION)
            st.write(ans.response if hasattr(ans, "response") else str(ans))
        except Exception as e:
            st.error(f"Query failed: {e}")
            st.info("Tip: if you cleared the collection, click 'Rebuild Index' first.")

# ----------------- Metrics Tab -----------------
with tab_metrics:
    st.subheader("Run-time Metrics")
    s = chroma_stats(DEFAULT_PATH, client)
    st.json(s)
    st.caption("These metrics update after rebuild/reset. Use 'Refresh Stats' in the sidebar.")

# ----------------- Settings Tab -----------------
with tab_settings:
    st.subheader("Configuration")
    st.write(f"**Chroma Path:** `{DEFAULT_PATH}`")
    st.write(f"**Collection:** `{DEFAULT_COLLECTION}`")
    st.write(f"**OPENAI_API_KEY set:** {bool(OPENAI_API_KEY)}")
    st.caption("To change models/paths, edit llama_utils.py and chroma_utils.py.")
