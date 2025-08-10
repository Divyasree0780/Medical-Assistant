# 🩺 Medical Chatbot with RAG (Retrieval-Augmented Generation)

This project implements a **Medical Assistant Chatbot** powered by **LLaMA 3-70B**, **PubMedBERT embeddings**, and **Qdrant Vector Database** using a **Retrieval-Augmented Generation (RAG)** pipeline. The chatbot retrieves contextually relevant medical information and generates factual, context-based answers without fabricating responses.

---

## 🚀 Features

* **Retrieval-Augmented Generation (RAG)** for accurate, context-aware answers.
* **PubMedBERT embeddings** for high-quality biomedical text understanding.
* **Qdrant** as a vector store for efficient semantic search.
* **LLaMA 3-70B** for large-scale, high-quality text generation via Groq API.
* **Streamlit UI** for interactive conversations.
* **Session memory** for chat continuity.
* Built-in safety:

  * Only answers based on retrieved context.
  * Avoids giving medical diagnoses or false information.

---

## 🛠️ Tech Stack

* **LangChain** – Orchestrating embeddings, retrieval, and LLM pipeline.
* **PubMedBERT** – Domain-specific embeddings for biomedical text.
* **Qdrant** – Vector database for storing and retrieving embeddings.
* **LLaMA 3-70B** – Large language model for answer generation (via Groq API).
* **Streamlit** – Web app interface.
* **Hugging Face Hub** – Embedding and model access.
* **Evaluation Metrics** – BLEU, ROUGE for text evaluation.

---

## 📦 Installation

### 1️⃣ Clone the repository

```bash
git clone https://github.com/Divyasree0780/Medical-Assistant.git
cd medical-chatbot-rag
```

### 2️⃣ Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
python -m nltk.downloader punkt
```

### 4️⃣ Set environment variables

set environment variables:

```env
qdrant_url=YOUR_QDRANT_URL
qdrant_api_key=YOUR_QDRANT_API_KEY
GROQ_API_KEY=YOUR_GROQ_API_KEY
HF_TOKEN=YOUR_HUGGINGFACE_TOKEN
```

---

## 📂 Project Structure

```
📦 medical-chatbot-rag
 ┣ 📜 app.py                 # Main Streamlit app
 ┣ 📜 requirements.txt       # Project dependencies
 ┣ 📜 README.md              # Documentation
 ┣ 📂 data                   # PDF of medical text data
 ┣ 📂 Research               # Research jupyter file
```

---

## ⚙️ How It Works

1. **Document Loading**

   * PDFs or medical text files are loaded using `PyPDFLoader` or `DirectoryLoader`.
2. **Text Processing**

   * Text is split into chunks with `RecursiveCharacterTextSplitter`.
3. **Embedding**

   * PubMedBERT converts chunks into vector embeddings.
4. **Vector Storage**

   * Qdrant stores embeddings for fast retrieval.
5. **Query Handling**

   * User queries are embedded and matched using cosine similarity.
6. **Context + LLaMA 3-70B**

   * Retrieved documents are fed into LLaMA for generating final answers.
7. **Streamlit Chat Interface**

   * Conversation is displayed and stored in session memory.

---

## ▶️ Run the Chatbot

```bash
streamlit run app.py
```

Open your browser and go to **[http://localhost:8501](http://localhost:8501)**.

---

## 📊 Evaluation

This project uses:

* **BLEU score** – Measures similarity between generated and reference text.
* **ROUGE score** – Measures recall and precision for text overlap.

---

## ⚠️ Disclaimer

> This chatbot is **not** a substitute for professional medical advice.
> It provides responses based only on the retrieved context and should be used for **educational purposes only**.

---

## 📜 License

MIT License © 2025 Divyasree Harikrishnan

---

