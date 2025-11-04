# === Imports ===
import os
import streamlit as st
from dotenv import load_dotenv

# --- Core LlamaIndex (v0.12.x stable API) ---
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, ServiceContext

# --- LLM + Embeddings (robust fallback) ---
try:
    from llama_index.llms.openai import OpenAI
    from llama_index.embeddings.openai import OpenAIEmbedding
except ImportError:
    # Fallback for builds where OpenAI moved under core.*
    from llama_index.core.llms.openai import OpenAI
    from llama_index.core.embeddings.openai import OpenAIEmbedding

# --- Vector Store + Chroma ---
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

# --- Optional Visualization ---
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_OK = True
except Exception:
    MATPLOTLIB_OK = False

# === Environment ===
load_dotenv()

# === Chroma Persistence (Force Local Mode) ===
persist_dir = "chroma_db"
os.makedirs(persist_dir, exist_ok=True)

try:
    # 🚫 Force Local Only – no host/port logic, no int() conversion
    chroma_client = chromadb.PersistentClient(path=persist_dir)
    vector_store = ChromaVectorStore(chroma_client=chroma_client)
    st.sidebar.success("💾 Local Chroma initialized successfully (persistent mode)")
except Exception as e:
    st.sidebar.error(f"❌ Local Chroma initialization failed: {e}")
    vector_store = None


# === ✅ Verify LLM + RAG Query Pipeline ===

if vector_store:
    try:
        st.sidebar.success("✅ Vector store initialized successfully")

        # --- Build index & service context ---
        embed_model = OpenAIEmbedding(model="text-embedding-3-small")
        llm = OpenAI(model="gpt-4o-mini", temperature=0.3)
        service_context = ServiceContext.from_defaults(llm=llm, embed_model=embed_model)
        index = VectorStoreIndex.from_vector_store(vector_store, service_context=service_context)

        # --- Create query engine ---
        query_engine = index.as_query_engine()

        # --- UI: Chat Section ---
        st.header("💬 Chat with your Documents")
        user_query = st.text_input("Ask your assistant...")

        if user_query:
            with st.spinner("🤖 Thinking..."):
                response = query_engine.query(user_query)
                st.markdown("### 🧠 Response")
                st.write(response.response)
                st.caption("Source nodes:")
                for node in response.source_nodes:
                    st.write("-", node.node_id)

    except Exception as e:
        st.error(f"⚠️ RAG initialization failed: {e}")

else:
    st.sidebar.warning("⚠️ Vector store not initialized; skipping query setup.")

# --- Lazy-load model logic ---
def get_service_context():
    """Dynamically choose backend (OpenAI / Hugging Face)."""
    if MODE == "openai":
        from llama_index.llms import OpenAI
        from llama_index.embeddings import OpenAIEmbedding
        llm = OpenAI(model="gpt-4-turbo")
        embed_model = OpenAIEmbedding(model="text-embedding-3-large")
    elif MODE == "huggingface":
        from llama_index.llms.huggingface import HuggingFaceLLM
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        llm = HuggingFaceLLM(model="mistralai/Mistral-7B-Instruct-v0.2", tokenizer="mistralai/Mistral-7B-Instruct-v0.2")
        embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    else:
        st.warning("Offline mode: limited functionality.")
        llm, embed_model = None, None
    return ServiceContext.from_defaults(llm=llm, embed_model=embed_model)

# --- Helper functions ---
def rebuild_index():
    docs_path = "docs"
    if not os.path.exists(docs_path) or not os.listdir(docs_path):
        st.error("⚠️ No documents found in docs/. Upload first.")
        return
    try:
        with st.spinner("🔨 Rebuilding persistent vector index..."):
            documents = SimpleDirectoryReader(docs_path).load_data()
            service_context = get_service_context()
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            index = VectorStoreIndex.from_documents(
                documents,
                service_context=service_context,
                storage_context=storage_context,
            )
            st.session_state.index = index
            st.session_state.query_engine = index.as_query_engine()
            st.success("✅ Index rebuilt successfully and ready for querying!")
    except Exception as e:
        st.error(f"❌ Index rebuild failed: {e}")

def clear_cache():
    import shutil
    if os.path.exists(persist_dir):
        shutil.rmtree(persist_dir)
        os.makedirs(persist_dir, exist_ok=True)
        st.warning("🧹 Cleared all persistent vector data from disk.")

# --- Initialize Session State ---
if "index" not in st.session_state:
    st.session_state.index = None
if "query_engine" not in st.session_state:
    st.session_state.query_engine = None

# --- Sidebar Navigation ---
page = st.sidebar.radio("📂 Navigate to:", ["💬 Chat", "📊 Metrics", "⚙️ Settings"])

# ==============================
# 💬 CHAT PAGE
# ==============================
if page == "💬 Chat":
    st.subheader("Chat with your Documents")
    uploaded_files = st.file_uploader(
        "Upload documents (.txt, .pdf, .docx)",
        type=["txt", "pdf", "docx"],
        accept_multiple_files=True,
    )
    if uploaded_files:
        docs_path = "docs"
        os.makedirs(docs_path, exist_ok=True)
        for file in uploaded_files:
            with open(os.path.join(docs_path, file.name), "wb") as f:
                f.write(file.read())
        st.info(f"📄 {len(uploaded_files)} document(s) saved to /docs")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("♻️ Reset Index"):
            st.session_state.index = None
            st.session_state.query_engine = None
            st.warning("🧹 Index cleared.")
    with col2:
        if st.button("🧠 Rebuild Index"):
            rebuild_index()

    user_q = st.chat_input("Ask your assistant...")
    if user_q:
        if not st.session_state.query_engine:
            st.warning("⚠️ Build or reload your index first.")
        else:
            with st.spinner("🤔 Thinking..."):
                try:
                    ans = st.session_state.query_engine.query(user_q)
                    st.write(ans.response)
                except Exception as e:
                    st.error(f"❌ Query failed: {e}")

# ==============================
# 📊 METRICS PAGE
# ==============================
elif page == "📊 Metrics":
    st.subheader("RAG System Metrics")
    num_docs = len(os.listdir("docs")) if os.path.exists("docs") else 0
    st.metric("📚 Documents Loaded", num_docs)
    st.metric("💾 Persistence Active", "Yes ✅" if os.path.exists(persist_dir) else "No ❌")
    st.metric("🌐 Mode", ENV_LABEL)

    if MATPLOTLIB_OK:
        import numpy as np
        data = np.random.randint(50, 250, 10)
        plt.figure(figsize=(5, 3))
        plt.plot(data, marker="o", color="blue")
        plt.title("Simulated Token Usage per Query")
        plt.xlabel("Query #")
        plt.ylabel("Token Count")
        st.pyplot(plt)
    else:
        st.info("📉 Matplotlib not installed — charts unavailable.")

# ==============================
# ⚙️ SETTINGS PAGE
# ==============================
elif page == "⚙️ Settings":
    st.subheader("Settings & Cache Control")
    if st.button("🧹 Clear Persistent Cache"):
        clear_cache()
    st.write(f"**Active Environment:** {ENV_LABEL}")
    st.info("Restart app after clearing cache for clean state.")
