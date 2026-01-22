# app.py
import os
import streamlit as st

# ---------------------------------
# MUST BE FIRST STREAMLIT COMMAND
# ---------------------------------
st.set_page_config(
    page_title="TFI ChitraVaani",
    page_icon="🎬",
    layout="centered",
)

# ---------------------------------
# Environment safety
# ---------------------------------
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ---------------------------------
# Lazy-load RAG pipeline (IMPORTANT)
# ---------------------------------
@st.cache_resource(show_spinner=False)
def load_qa_chain():
    from rag.rag_pipeline import qa_chain

    return qa_chain


qa_chain = load_qa_chain()

# ---------------------------------
# UI Header
# ---------------------------------
st.markdown(
    """
    <h1 style="text-align:center;">🎬 TFI ChitraVaani</h1>
    <p style="text-align:center; color:gray; font-size:16px;">
        Ask questions strictly based on a curated Tollywood movie dataset.<br>
        ❌ No external knowledge &nbsp;•&nbsp; ✅ No hallucinations
    </p>
    """,
    unsafe_allow_html=True,
)

st.markdown("---")

# ---------------------------------
# Example questions
# ---------------------------------
st.markdown("### 💡 Try example questions")

example_questions = [
    "Who directed Baahubali: The Beginning?",
    "List songs from Baahubali: The Beginning",
    "Who composed the music for Baahubali: The Beginning?",
    "What awards did Baahubali: The Beginning win?",
]

cols = st.columns(2)
for i, q in enumerate(example_questions):
    if cols[i % 2].button(q):
        st.session_state["query"] = q

# ---------------------------------
# User Input
# ---------------------------------
query = st.text_input(
    "🔍 Ask a question about Tollywood movies:",
    value=st.session_state.get("query", ""),
    placeholder="e.g., Who directed Baahubali: The Beginning?",
)

# ---------------------------------
# Run RAG
# ---------------------------------
if query:
    with st.spinner("🔎 Searching movie knowledge..."):
        result = qa_chain(query)

    st.markdown("## 📌 Answer")
    st.success(result["result"])

    if result.get("source_documents"):
        with st.expander("📂 Source Movies Used"):
            for doc in result["source_documents"]:
                st.write("🎞️", doc.metadata.get("movie_name", "Unknown"))

# ---------------------------------
# Footer
# ---------------------------------
st.markdown("---")
st.markdown(
    """
    <div style="text-align:center; color:gray; font-size:14px;">
        📚 Powered by a curated Tollywood movie dataset<br>
        🛡️ Answers are generated only from available data
    </div>
    """,
    unsafe_allow_html=True,
)
