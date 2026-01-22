🎬 TFI ChitraVaani
Tollywood Movie Knowledge Assistant (RAG-based)
🏆 Hackathon-2 Project
📌 Problem Statement

Most AI chatbots answer questions using general internet knowledge and often hallucinate incorrect facts, especially when asked about specific domains like regional cinema.

For hackathons and real-world applications, this creates:

❌ Misinformation

❌ Lack of trust

❌ No clear data source

💡 Solution Overview

TFI ChitraVaani is a Retrieval-Augmented Generation (RAG) based web application that answers questions strictly from a curated Tollywood movie dataset.

The system:

Retrieves relevant movie information from a local dataset

Generates answers only from retrieved content

Explicitly refuses to answer when information is not available

🛡️ Zero hallucination by design

🎯 Key Features

🎬 Domain-specific Tollywood movie knowledge

📚 Curated, structured dataset

🔍 FAISS-based semantic retrieval

🤖 Offline HuggingFace LLM (no APIs)

❌ No external web search

🛡️ Strict hallucination prevention

🧾 Transparent source display

🎨 Clean, interactive Streamlit UI

🧠 How It Works (RAG Pipeline)
User Question
     ↓
Semantic Retrieval (FAISS)
     ↓
Relevant Movie Documents
     ↓
Strict Prompt + Validation
     ↓
Answer OR Explicit Denial

🔒 Hallucination Control

The system checks whether the queried movie exists in the dataset

If not, it responds with:

The requested information is not available in the provided dataset.

🗂️ Project Structure
tfi-chitravaani/
│
├── app.py                  # Streamlit UI
│
├── data/
│   └── movies.json         # Curated Tollywood dataset
│
├── rag/
│   ├── __init__.py
│   └── rag_pipeline.py     # RAG logic (retrieval + generation)
│
├── venv/
└── README.md

📊 Dataset Details

Each movie document contains structured fields such as:

Movie name

Director

Producer

Music director

Lyricist

Cast

Release year

Songs list

Awards

IMDb rating

Interesting facts

Wikipedia links

⚠️ The assistant never answers beyond this dataset.

🧪 Example Questions (Valid)

Who directed Baahubali: The Beginning?

List songs from Baahubali: The Beginning

What awards did Baahubali: The Beginning win?

Who composed the music for Baahubali: The Beginning?

❌ Invalid (Hallucination Test)

What is the budget of Avatar?

Who directed Titanic?

➡️ Correct response:

The requested information is not available in the provided dataset.

🛠️ Tech Stack
Component	Technology
Language	Python
UI	Streamlit
Vector DB	FAISS
Embeddings	sentence-transformers
LLM	HuggingFace (FLAN-T5)
Framework	LangChain (modern LCEL usage)
▶️ How to Run the Project
1️⃣ Create virtual environment
python3 -m venv venv
source venv/bin/activate

2️⃣ Install dependencies
pip install streamlit langchain langchain-core langchain-community \
            sentence-transformers transformers torch faiss-cpu

3️⃣ Run the app
streamlit run app.py


Open:

http://localhost:8501

🏆 Why This Project Stands Out

✅ Fully offline (no API dependency)

✅ Deterministic hallucination prevention

✅ Transparent source attribution

✅ Clean architecture

✅ Easy to extend with more movies

✅ Hackathon-friendly & explainable

🎤 One-Line Hackathon Pitch

“TFI ChitraVaani is a fully offline RAG-based assistant that answers Tollywood movie questions strictly from a curated dataset, eliminating hallucinations through deterministic validation.”

🚀 Future Enhancements

Add more Tollywood movies

Telugu language support

Movie info cards with posters

Confidence scoring for answers

Deployment on Streamlit Cloud

🙌 Team / Author

Lakshman Reddy Patlolla
Bhavitha 
B.Tech CSE | Hackathon-2 Participant