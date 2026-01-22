import streamlit as st
import os

st.set_page_config(page_title="TFI ChitraVaani", page_icon="🎬", layout="centered")

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from rag.rag_pipeline import qa_chain

st.title("🎬 TFI ChitraVaani")
st.caption("Tollywood QA • Dataset-only • No hallucinations")

query = st.text_input("Ask a Tollywood movie question:")

if query:
    with st.spinner("Searching..."):
        result = qa_chain(query)

    st.success(result["result"])

    if result["source_documents"]:
        with st.expander("Sources"):
            for doc in result["source_documents"]:
                st.write("🎞️", doc.metadata.get("movie_name"))
