import json
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# -----------------------------
# Load dataset
# -----------------------------
with open("data/movies.json", "r", encoding="utf-8") as f:
    movies = json.load(f)

documents = []
for movie in movies:
    content = "\n".join(f"{k}: {v}" for k, v in movie.items())
    documents.append(
        Document(
            page_content=content,
            metadata={"movie_name": movie.get("movie_name", "Unknown")},
        )
    )

# -----------------------------
# Embeddings
# -----------------------------
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

doc_texts = [doc.page_content for doc in documents]
doc_embeddings = embeddings.embed_documents(doc_texts)

# -----------------------------
# Prompt
# -----------------------------
PROMPT = PromptTemplate.from_template(
    """
You are a Tollywood Movie Knowledge Assistant.

Answer ONLY using the context below.
If the answer is not present, say:
"The requested information is not available in the provided dataset."

Context:
{context}

Question:
{question}

Answer:
"""
)

# -----------------------------
# Local LLM
# -----------------------------
generator = pipeline(
    "text2text-generation", model="google/flan-t5-small", max_new_tokens=256
)

llm = HuggingFacePipeline(pipeline=generator)


# -----------------------------
# QA function
# -----------------------------
def qa_chain(question: str):
    q_embedding = embeddings.embed_query(question)

    scores = cosine_similarity([q_embedding], doc_embeddings)[0]

    top_indices = np.argsort(scores)[-3:][::-1]
    top_docs = [documents[i] for i in top_indices]

    context = "\n\n".join(doc.page_content for doc in top_docs)

    prompt = PROMPT.format(context=context, question=question)

    answer = llm.invoke(prompt)

    return {"result": answer, "source_documents": top_docs}
