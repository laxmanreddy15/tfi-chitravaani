# rag/rag_pipeline.py

import json
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline

from transformers import pipeline


# -----------------------------
# 1️⃣ Load dataset (NO jq, NO loaders)
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
# 2️⃣ Embeddings (offline)
# -----------------------------
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


# -----------------------------
# 3️⃣ FAISS Vector Store
# -----------------------------
vectorstore = FAISS.from_documents(documents, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})


# -----------------------------
# 4️⃣ Strict anti-hallucination prompt
# -----------------------------
PROMPT = PromptTemplate.from_template(
    """
You are a Tollywood Movie Knowledge Assistant.

Answer the question ONLY using the information provided in the context below.
Do NOT use external knowledge.
Do NOT guess.

If the answer is not present in the context, reply exactly:
"The requested information is not available in the provided dataset."

Context:
{context}

Question:
{question}

Answer:
"""
)


# -----------------------------
# 5️⃣ Local LLM (offline, hackathon-safe)
# -----------------------------
generator = pipeline(
    "text2text-generation", model="google/flan-t5-small", max_new_tokens=256
)

llm = HuggingFacePipeline(pipeline=generator)


# -----------------------------
# 6️⃣ Custom QA function (simple & reliable)
# -----------------------------
def qa_chain(question: str):
    question_lower = question.lower()

    # All movie names in the dataset
    dataset_movie_names = {
        doc.metadata.get("movie_name", "").lower()
        for doc in retriever.vectorstore.docstore._dict.values()
        if doc.metadata.get("movie_name")
    }

    # Try to detect if question mentions ANY movie-like word
    mentioned_movies = [
        movie for movie in dataset_movie_names if movie in question_lower
    ]

    # 🚫 If question contains a movie name NOT in dataset → deny
    words = question_lower.split()
    potential_movie_words = [w for w in words if len(w) > 3]

    if not mentioned_movies:
        # If question looks like it's asking about a movie but none match dataset
        if any(
            word in question_lower
            for word in ["budget", "director", "songs", "cast", "imdb"]
        ):
            return {
                "result": "The requested information is not available in the provided dataset.",
                "source_documents": [],
            }

    # ✅ Retrieve documents
    docs = retriever.invoke(question)

    if not docs:
        return {
            "result": "The requested information is not available in the provided dataset.",
            "source_documents": [],
        }

    # Build context
    context = "\n\n".join(doc.page_content for doc in docs)

    prompt = PROMPT.format(context=context, question=question)

    answer = llm.invoke(prompt)

    return {"result": answer, "source_documents": docs}
