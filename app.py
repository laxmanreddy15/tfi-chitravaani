import streamlit as st
from rag.rag_pipeline import qa_chain

# ---------------------------------
# Page configuration
# ---------------------------------
st.set_page_config(
    page_title="TFI ChitraVaani",
    page_icon="🎬",
    layout="centered"
)

# ---------------------------------
# Title & description
# ---------------------------------
st.markdown(
    """
    <h1 style="text-align:center;">🎬 tfi-chitravaaani</h1>
    <p style="text-align:center; color:gray; font-size:16px;">
        Ask questions strictly based on a curated Tollywood movie dataset.<br>
        ❌ No external knowledge &nbsp;•&nbsp; ✅ No hallucination
    </p>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

# ---------------------------------
# Example questions (interactive)
# ---------------------------------
st.markdown("### 💡 Try these example questions")

example_questions = [
    "Who directed Baahubali: The Beginning?",
    "List songs from Baahubali: The Beginning",
    "Who composed the music for Baahubali: The Beginning?",
    "What awards did Baahubali: The Beginning win?"
]

cols = st.columns(2)
for i, q in enumerate(example_questions):
    if cols[i % 2].button(q):
        st.session_state["query"] = q

# ---------------------------------
# Question input
# ---------------------------------
query = st.text_input(
    "🔍 Ask a question about Tollywood movies:",
    value=st.session_state.get("query", ""),
    placeholder="e.g., Who directed Baahubali: The Beginning?"
)

# ---------------------------------
# Run RAG pipeline
# ---------------------------------
if query:
    with st.spinner("🔎 Searching movie knowledge..."):
        result = qa_chain(query)

    # -----------------------------
    # Answer section
    # -----------------------------
    st.markdown("## 📌 Answer")
    st.success(result["result"])

    # -----------------------------
    # Source transparency
    # -----------------------------
    if result["source_documents"]:
        with st.expander("📂 Source Movies Used"):
            for doc in result["source_documents"]:
                st.write("🎞️", doc.metadata.get("movie_name", "Unknown"))

# ---------------------------------
# Footer / Trust badge
# ---------------------------------
st.markdown("---")
st.markdown(
    """
    <div style="text-align:center; color:gray; font-size:14px;">
        📚 Powered by a curated Tollywood movie dataset<br>
        🛡️ Answers are generated only from available data
    </div>
    """,
    unsafe_allow_html=True
)
