# rag/rag_pipeline.py

import json
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline
import numpy as np


# -----------------------------
# 1️⃣ Load dataset
# -----------------------------
with open("data/movies.json", "r", encoding="utf-8") as f:
    movies = json.load(f)

documents = []
for movie in movies:
    content = "\n".join([f"{k}: {v}" for k, v in movie.items()])
    documents.append(
        Document(
            page_content=content,
            metadata={"movie_name": movie.get("movie_name", "Unknown")},
        )
    )


# -----------------------------
# 2️⃣ Embeddings (CPU-safe)
# -----------------------------
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

doc_embeddings = embeddings.embed_documents([doc.page_content for doc in documents])
doc_embeddings = np.array(doc_embeddings)


# -----------------------------
# 3️⃣ Simple Retriever (NO FAISS)
# -----------------------------
def retrieve_docs(query, k=3):
    query_embedding = embeddings.embed_query(query)
    scores = np.dot(doc_embeddings, query_embedding)
    top_indices = scores.argsort()[-k:][::-1]
    return [documents[i] for i in top_indices]


# -----------------------------
# 4️⃣ Anti-hallucination Prompt
# -----------------------------
PROMPT = PromptTemplate.from_template(
    """
You are a Tollywood Movie Knowledge Assistant.

Answer ONLY using the context below.
Do NOT use external knowledge.
Do NOT guess.

If the answer is not present, reply exactly:
"The requested information is not available in the provided dataset."

Context:
{context}

Question:
{question}

Answer:
"""
)


# -----------------------------
# 5️⃣ Local LLM (Streamlit-safe)
# -----------------------------
generator = pipeline(
    "text2text-generation", model="google/flan-t5-small", max_new_tokens=256
)

llm = HuggingFacePipeline(pipeline=generator)


# -----------------------------
# 6️⃣ QA Chain
# -----------------------------
def qa_chain(question: str):
    docs = retrieve_docs(question)

    if not docs:
        return {
            "result": "The requested information is not available in the provided dataset.",
            "source_documents": [],
        }

    context = "\n\n".join(doc.page_content for doc in docs)

    prompt = PROMPT.format(context=context, question=question)

    answer = llm.invoke(prompt)

    return {"result": answer, "source_documents": docs}
