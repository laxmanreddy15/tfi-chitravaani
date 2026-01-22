# rag/rag_pipeline.py

import json
import numpy as np
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline


# --------------------------------------------------
# 1️⃣ Load movie dataset
# --------------------------------------------------
with open("data/movies.json", "r", encoding="utf-8") as f:
    movies = json.load(f)

texts = []
metadata = []

for movie in movies:
    content = "\n".join([f"{k}: {v}" for k, v in movie.items()])
    texts.append(content)
    metadata.append(movie.get("movie_name", "Unknown"))


# --------------------------------------------------
# 2️⃣ Load embedding model (safe for Streamlit)
# --------------------------------------------------
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
embeddings = embedding_model.encode(texts, convert_to_tensor=True)


# --------------------------------------------------
# 3️⃣ Load lightweight LLM (offline-friendly)
# --------------------------------------------------
generator = pipeline(
    "text2text-generation", model="google/flan-t5-small", max_new_tokens=256
)


# --------------------------------------------------
# 4️⃣ QA Function (NO hallucination)
# --------------------------------------------------
def qa_chain(question: str):
    question_embedding = embedding_model.encode(question, convert_to_tensor=True)

    # Compute cosine similarity
    scores = util.cos_sim(question_embedding, embeddings)[0]
    top_k = torch_topk(scores, k=3)

    if not top_k:
        return {
            "result": "The requested information is not available in the provided dataset.",
            "source_documents": [],
        }

    context = "\n\n".join(texts[i] for i in top_k)

    prompt = f"""
You are a Tollywood Movie Knowledge Assistant.

Answer ONLY using the context below.
Do NOT guess.
Do NOT use external knowledge.

If the answer is not present, reply exactly:
"The requested information is not available in the provided dataset."

Context:
{context}

Question:
{question}

Answer:
"""

    response = generator(prompt)[0]["generated_text"]

    return {
        "result": response,
        "source_documents": [{"movie_name": metadata[i]} for i in top_k],
    }


# --------------------------------------------------
# Helper: torch-free top-k
# --------------------------------------------------
def torch_topk(tensor, k=3):
    scores = tensor.cpu().numpy()
    indices = np.argsort(scores)[-k:][::-1]
    return indices.tolist()
