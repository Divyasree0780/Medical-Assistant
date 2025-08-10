from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List
from langchain.schema import Document
from langchain_huggingface  import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from huggingface_hub import InferenceClient
from datasets import load_dataset
import tqdm as notebook_tqdm
from langchain_qdrant import Qdrant
import time
import evaluate
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
import warnings
import numpy as np
import streamlit as st
from streamlit_chat import message
import time
import os 

qdrant_url = os.environ["qdrant_url"]
qdrant_api_key = os.environ["qdrant_api_key"]
GROQ_API_KEY = os.environ["GROQ_API_KEY"]
HF_TOKEN = os.environ["HF_TOKEN"]

#Download the Embeddings from Hugging Face
def download_hugging_face_embeddings():
    embeddings=HuggingFaceEmbeddings(model_name='NeuML/pubmedbert-base-embeddings')
    return embeddings

embeddings = download_hugging_face_embeddings()

# Initialise Qdrant Client
qdrant_client = QdrantClient(
    url=qdrant_url, 
    api_key=qdrant_api_key,
)

print(qdrant_client.get_collections())

# Configuration
collection_name = "vectordb"
embedding_dim = 768

# Check if collection exists
if not qdrant_client.collection_exists(collection_name):
    qdrant_client.recreate_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=embedding_dim,
            distance=models.Distance.COSINE
        )
    )
    print(f"Collection '{collection_name}' created.")
else:
    print(f"Collection '{collection_name}' already exists.")

# Adding documents to vector db

# Initialize LangChain Qdrant vectorstore
qdrant = Qdrant(
    client=qdrant_client,
    collection_name=collection_name,
    embeddings=embeddings
)

# Initialize your retriever (make sure you have this properly imported and configured)
retriever = qdrant.as_retriever(search_kwargs={"k": 3})

# Your LLaMA RAG query function (update GROQ_API_KEY accordingly)
def query_llama_rag(question: str, retriever):
    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="llama3-70b-8192"
    )

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
You are **Medical chatbot**, an AI-powered assistant trained to help users understand medical and health-related questions.

Your job is to provide clear, accurate, and helpful responses based **only on the provided context**.

---

**Context**:
{context}

**User Question**:
{question}

---

**Answer**:
- Respond in a calm, factual, and respectful tone.
- Use simple explanations when needed.
- If the context does not contain the answer, say: "I'm sorry, but I couldn't find relevant information in the provided documents."
- Do NOT make up facts.
- Do NOT give medical advice or diagnoses.
"""
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )

    result = qa_chain.invoke(question)
    return result['result'], result['source_documents']

# Streamlit app starts here
st.set_page_config(page_title="Medical Chatbot", page_icon="🤖", layout="centered")

if "history" not in st.session_state:
    st.session_state.history = []  # list of dicts with "role" and "content"

st.title("🤖 Medical Chatbot with Memory")

st.sidebar.header("About")
st.sidebar.info(
    """
    This Medical chatbot uses LLaMA-3-70B with Retrieval-Augmented Generation (RAG) pipeline combined with Vector Search using Qdrant DB.
    Your conversation is stored in memory during this session.
    """
)

if st.sidebar.button("Clear Chat"):
    st.session_state.history = []

with st.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_input("You:", placeholder="Type your question here...")
    submit_button = st.form_submit_button(label="Send")

if submit_button and user_input.strip():
    st.session_state.history.append({"role": "user", "content": user_input.strip()})

    with st.spinner("Bot is thinking..."):
        response, sources = query_llama_rag(user_input.strip(), retriever)
        st.session_state.history.append({"role": "bot", "content": response})

# Display chat messages
for chat in st.session_state.history:
    if chat["role"] == "user":
        message(chat["content"], is_user=True, avatar_style="big-smile")
    else:
        message(chat["content"], is_user=False, avatar_style="bottts") 