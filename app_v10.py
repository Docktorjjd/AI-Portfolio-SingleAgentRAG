# ============================================================
# 🤖 RAG Assistant (v10): Auto-Rebuild + Dashboard Edition
# Streamlit + LlamaIndex + Chroma + OpenAI
# ============================================================

import os
import shutil
import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# --- Optional: Visualization library ---
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_OK = True
except Exception:
    MATPLOTLIB_OK = False

# ---------- Load Environment ----------
load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_KEY:
    st.success("✅ Environment variable loaded correctly.")
else:
    st.warning("⚠️ No OPENAI_API_KEY found in .env file")

# ---------- Version Info ----------
try:
    import importlib.metadata as importlib_metadata
    LI_VER = importlib_metadata.version("llama-index-core")
except Exception:
    LI_VER = "unknown"
st.caption(f"Using LlamaIndex version: {LI_VER}")

# ---------- Core Imports ----------
try:
    from openai import OpenAI
    from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
    from llama_index.embeddings.openai import OpenAIEmbedding
    from llama_index.vector_stores.chroma import ChromaVectorStore
    from llama_index.core.node_parser import SimpleNodeParser
    import chromadb
except ImportError as e:
    st.error(f"❌ Missing dependency: {e}")
    st.stop()

# ---------- App Title ----------
st.title("🧠 RAG Assistant — Dashboard Edition (v10)")

# ---------- Memory Mode ----------
memory_mode = st.radio(
    "🧩 Memory Mode",
    ["Short-Term (session only)", "Persistent (saves to disk)"],
    horizontal=True,
)

# ---------- Paths ----------
DOCS_DIR = "docs"
os.makedirs(DOCS_DIR, exist_ok=True)
persist_dir = "chroma_db" if memory_mode == "Persistent (saves to disk)" else None

# ---------- Upload Documents ----------
st.subheader("📤 Upload Documents")
uploaded_files = st.file_uploader(
    "Drop PDF, TXT, or DOCX files here",
    type=["pdf", "txt", "docx"],
    accept_multiple_files=True
)

# ---------- Local Chroma Client ----------
try:
    if persist_dir:
        chroma_client = chromadb.PersistentClient(path=persist_dir)
        mode = "persistent"
    else:
        chroma_client = chromadb.EphemeralClient()
        mode = "in-memory"
    st.success(f"🍀 Using local Chroma client ({mode})")

    # ✅ Explicitly create collection (avoid NoneType port bug)
    vector_store = ChromaVectorStore(
        chroma_collection=chroma_client.get_or_create_collection("rag_docs")
    )

except Exception as e:
    st.error(f"❌ Chroma initialization failed: {e}")
    st.stop()

# ---------- Reset & Rebuild Controls ----------
col1, col2 = st.columns(2)
with col1:
    if st.button("♻️ Reset Vector Index"):
        try:
            if persist_dir and os.path.exists(persist_dir):
                shutil.rmtree(persist_dir)
                os.makedirs(persist_dir, exist_ok=True)
            chroma_client = (
                chromadb.PersistentClient(path=persist_dir)
                if persist_dir
                else chromadb.EphemeralClient()
            )
            vector_store = ChromaVectorStore(
                chroma_collection=chroma_client.get_or_create_collection("rag_docs")
            )
            st.session_state.analytics = {"query_count": 0, "doc_hits": {}, "similarities": []}
            st.session_state.chat_history = []
            st.success("✅ Vector index reset successfully.")
        except Exception as e:
            st.error(f"⚠️ Failed to reset index: {e}")

with col2:
    if st.button("🧠 Rebuild Index"):
        try:
            with st.spinner("Rebuilding embeddings and index..."):
                documents = SimpleDirectoryReader(DOCS_DIR).load_data()
                embed_model = OpenAIEmbedding(model="text-embedding-3-small")
                storage_ctx = StorageContext.from_defaults(vector_store=vector_store)
                index = VectorStoreIndex.from_documents(
                    documents, embedding=embed_model, storage_context=storage_ctx
                )
                retriever = index.as_retriever(similarity_top_k=4)
                query_engine = index.as_query_engine(similarity_top_k=4, response_mode="compact")
            st.success("✅ Index rebuilt successfully!")
        except Exception as e:
            st.error(f"⚠️ Error rebuilding index: {e}")

# ---------- Auto-Rebuild on Upload (with progress + toast) ----------
if uploaded_files:
    with st.spinner("📚 Processing new uploads..."):
        try:
            total = len(uploaded_files)
            progress = st.progress(0)
            for i, f in enumerate(uploaded_files, 1):
                file_path = os.path.join(DOCS_DIR, f.name)
                with open(file_path, "wb") as out:
                    out.write(f.getbuffer())
                progress.progress(i / total)
            st.toast(f"✅ {total} file(s) saved to docs/", icon="📂")

            # ---- Rebuild embeddings automatically ----
            st.info("🧠 Rebuilding vector index with new documents...")
            documents = SimpleDirectoryReader(DOCS_DIR).load_data()
            embed_model = OpenAIEmbedding(model="text-embedding-3-small")
            storage_ctx = StorageContext.from_defaults(vector_store=vector_store)

            # Progress feedback for embeddings
            n_docs = len(documents)
            progress2 = st.progress(0)
            for i, doc in enumerate(documents):
                _ = embed_model.get_text_embedding(doc.text[:2000])  # simulate embedding progress
                progress2.progress((i + 1) / n_docs)

            index = VectorStoreIndex.from_documents(
                documents, embedding=embed_model, storage_context=storage_ctx
            )
            retriever = index.as_retriever(similarity_top_k=4)
            query_engine = index.as_query_engine(similarity_top_k=4, response_mode="compact")

            st.success(f"🎉 Auto-rebuild complete — indexed {n_docs} document(s)!")
            st.toast("🎉 Vector index rebuilt successfully!", icon="🚀")

        except Exception as e:
            st.error(f"⚠️ Auto-rebuild failed: {e}")

# ---------- Sidebar Analytics ----------
if "analytics" not in st.session_state:
    st.session_state.analytics = {"query_count": 0, "doc_hits": {}, "similarities": []}

with st.sidebar:
    st.header("📊 Analytics")
    st.metric("Total Queries", st.session_state.analytics["query_count"])
    if st.session_state.analytics["similarities"]:
        avg_sim = sum(st.session_state.analytics["similarities"]) / len(st.session_state.analytics["similarities"])
        st.metric("Avg. Similarity", f"{avg_sim:.3f}")
    else:
        st.metric("Avg. Similarity", "—")

# ---------- Chat ----------
st.subheader("💬 Chat with your Documents")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for m in st.session_state.chat_history:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if user_q := st.chat_input("Ask a question..."):
    with st.chat_message("user"):
        st.markdown(user_q)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                answer = query_engine.query(user_q).response
            except Exception as e:
                answer = f"⚠️ Error: {e}"
            st.markdown(answer)
    st.session_state.chat_history.append({"role": "user", "content": user_q})
    st.session_state.chat_history.append({"role": "assistant", "content": answer})
