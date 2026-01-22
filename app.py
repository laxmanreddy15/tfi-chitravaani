import streamlit as st
import os

# Must be before model loads
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from rag.rag_pipeline import qa_chain

# --------------------------------------------------
# Page config (FIRST Streamlit command)
# --------------------------------------------------
st.set_page_config(page_title="TFI ChitraVaani", page_icon="🎬", layout="centered")

# --------------------------------------------------
# UI Header
# --------------------------------------------------
st.markdown(
    """
    <h1 style="text-align:center;">🎬 TFI ChitraVaani</h1>
    <p style="text-align:center; color:gray;">
        Ask questions strictly from the Tollywood movie dataset.<br>
        ❌ No external knowledge • ✅ No hallucinations
    </p>
    """,
    unsafe_allow_html=True,
)

st.divider()

# --------------------------------------------------
# Example Questions
# --------------------------------------------------
st.markdown("### 💡 Example Questions")

examples = [
    "Who directed Baahubali: The Beginning?",
    "List songs from Baahubali: The Beginning",
    "Who composed the music for Baahubali: The Beginning?",
    "What awards did Baahubali: The Beginning win?",
]

for q in examples:
    if st.button(q):
        st.session_state["query"] = q

# --------------------------------------------------
# Input
# --------------------------------------------------
query = st.text_input(
    "🔍 Ask a question about Tollywood movies:",
    value=st.session_state.get("query", ""),
    placeholder="e.g., Who directed Baahubali?",
)

# --------------------------------------------------
# Run RAG
# --------------------------------------------------
if query:
    with st.spinner("🔎 Searching movie knowledge..."):
        result = qa_chain(query)

    st.subheader("📌 Answer")
    st.success(result["result"])

    if result["source_documents"]:
        with st.expander("📂 Source Movies Used"):
            for doc in result["source_documents"]:
                st.write("🎞️", doc["movie_name"])

st.divider()

st.markdown(
    """
    <div style="text-align:center; color:gray; font-size:14px;">
        📚 Powered by curated Tollywood movie dataset<br>
        🛡️ Zero-hallucination RAG system
    </div>
    """,
    unsafe_allow_html=True,
)
