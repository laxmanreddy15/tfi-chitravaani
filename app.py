import os
import streamlit as st

# MUST be before any Streamlit UI calls
st.set_page_config(page_title="TFI ChitraVaani", page_icon="🎬", layout="centered")

# Environment fix (important for HuggingFace)
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# Lazy-load RAG pipeline (prevents cold-start crashes)
@st.cache_resource
def load_qa_chain():
    from rag.rag_pipeline import qa_chain

    return qa_chain


qa_chain = load_qa_chain()

# ---------------------------------
# UI
# ---------------------------------
st.markdown(
    """
    <h1 style="text-align:center;">🎬 TFI ChitraVaani</h1>
    <p style="text-align:center; color:gray;">
        Ask questions strictly based on Tollywood movie data
    </p>
    """,
    unsafe_allow_html=True,
)

st.markdown("### 💡 Example Questions")

examples = [
    "Who directed Baahubali: The Beginning?",
    "List songs from Baahubali: The Beginning",
    "Who composed the music for Baahubali?",
    "What awards did Baahubali win?",
]

for q in examples:
    if st.button(q):
        st.session_state["query"] = q

query = st.text_input("Ask a question:", value=st.session_state.get("query", ""))

if query:
    with st.spinner("Searching movie knowledge..."):
        result = qa_chain(query)

    st.success(result["result"])

    if result.get("source_documents"):
        with st.expander("Sources"):
            for doc in result["source_documents"]:
                st.write(doc.metadata.get("movie_name", "Unknown"))
