import streamlit as st
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

st.set_page_config(page_title="TFI ChitraVaani", page_icon="🎬", layout="centered")

st.title("🎬 TFI ChitraVaani")
st.caption(
    "Ask questions strictly based on a curated Tollywood movie dataset.\n"
    "❌ No external knowledge • ✅ No hallucination"
)


@st.cache_resource
def load_qa_chain():
    from rag.rag_pipeline import qa_chain

    return qa_chain


qa_chain = load_qa_chain()

st.markdown("### 💡 Example Questions")

examples = [
    "Who directed Baahubali: The Beginning?",
    "List songs from Baahubali: The Beginning",
    "Who composed music for Baahubali: The Beginning?",
    "What awards did Baahubali: The Beginning win?",
]

cols = st.columns(2)
for i, q in enumerate(examples):
    if cols[i % 2].button(q):
        st.session_state["query"] = q

query = st.text_input(
    "🔍 Ask a question about Tollywood movies",
    value=st.session_state.get("query", ""),
    placeholder="e.g. Who directed Baahubali?",
)

if query:
    with st.spinner("🔎 Searching movie knowledge..."):
        result = qa_chain(query)

    st.markdown("## 📌 Answer")
    st.success(result["result"])

    if result["source_documents"]:
        with st.expander("📂 Source Movies Used"):
            for doc in result["source_documents"]:
                st.write("🎞️", doc.metadata.get("movie_name", "Unknown"))

st.markdown("---")
st.caption("📚 Powered by curated Tollywood dataset • 🛡️ No hallucinations")
