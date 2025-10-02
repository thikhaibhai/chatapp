# ==============================
# Install dependencies (run once in terminal before streamlit run)
# pip install streamlit sentence-transformers PyMuPDF langchain numpy google-generativeai pillow
# ==============================

import os
import fitz  # PyMuPDF
import numpy as np
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import google.generativeai as genai
import streamlit as st

# ==============================
# Load API Key from Streamlit Secrets
# ==============================
if "GOOGLE_API_KEY" not in st.secrets:
    st.error("❌ GOOGLE_API_KEY not found. Please add it in Streamlit Cloud → Settings → Secrets")
    st.stop()

GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
genai.configure(api_key=GOOGLE_API_KEY)

# ==============================
# Data Structures
# ==============================
class ChunkWithSource:
    def __init__(self, text: str, source: str):
        self.text = text
        self.source = source

# ==============================
# File Handling
# ==============================
def get_files(uploaded_files) -> Dict[str, List[str]]:
    # Ensure upload folder exists
    os.makedirs("uploaded_files", exist_ok=True)

    files = {'pdfs': [], 'images': []}
    for uploaded_file in uploaded_files:
        name = uploaded_file.name.lower()
        temp_path = os.path.join("uploaded_files", uploaded_file.name)

        # Save uploaded file
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.read())

        if name.endswith('.pdf'):
            files['pdfs'].append(temp_path)
        elif name.endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp')):
            files['images'].append(temp_path)

    return files

# ==============================
# PDF & Image Processing
# ==============================
def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page_num, page in enumerate(doc, start=1):
            text += page.get_text()
    return text

def process_image_with_gemini(image_path: str) -> str:
    try:
        with open(image_path, 'rb') as f:
            image_bytes = f.read()
        model = genai.GenerativeModel('gemini-2.0-flash')
        prompt = "Extract and return all text visible in this image. Format it clearly."
        response = model.generate_content([
            {"mime_type": f'image/{image_path.split(".")[-1].lower()}', "data": image_bytes},
            prompt
        ])
        return response.text
    except Exception as e:
        st.error(f"Error processing image with Gemini: {str(e)}")
        return ""

def split_text_into_chunks(text, chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", "!", "?", " ", ""]
    )
    return splitter.split_text(text)

# ==============================
# Embeddings
# ==============================
def sentence_encode(sentences):
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    embeddings = model.encode(sentences)
    return embeddings

def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# ==============================
# Conversation Memory
# ==============================
class ConversationMemory:
    def __init__(self, max_history: int = 5):
        self.history: List[Dict] = []
        self.max_history = max_history

    def add_interaction(self, query: str, response: str, context: str):
        self.history.append({"query": query, "response": response, "context": context})
        if len(self.history) > self.max_history:
            self.history.pop(0)

    def get_formatted_history(self) -> str:
        formatted = ""
        for interaction in self.history:
            formatted += f"Question: {interaction['query']}\n"
            formatted += f"Answer: {interaction['response']}\n"
        return formatted

# ==============================
# Main Document Processing
# ==============================
def process_documents(uploaded_files) -> List[ChunkWithSource]:
    chunks_with_sources = []
    files = get_files(uploaded_files)

    for pdf_path in files['pdfs']:
        filename = os.path.basename(pdf_path)
        text = extract_text_from_pdf(pdf_path)
        chunks = split_text_into_chunks(text)
        for chunk in chunks:
            chunks_with_sources.append(ChunkWithSource(chunk, filename))

    for image_path in files['images']:
        filename = os.path.basename(image_path)
        text = process_image_with_gemini(image_path)
        if text:
            chunks = split_text_into_chunks(text)
            for chunk in chunks:
                chunks_with_sources.append(ChunkWithSource(chunk, filename))

    return chunks_with_sources

# ==============================
# Streamlit UI
# ==============================
st.set_page_config(page_title="PDF & Image Chatbot", layout="wide")
st.title("📑 PDF & Image Q&A with Gemini")

uploaded_files = st.file_uploader("Upload PDFs or Images", type=["pdf", "png", "jpg", "jpeg"], accept_multiple_files=True)

if uploaded_files:
    st.success(f"{len(uploaded_files)} file(s) uploaded successfully.")
    chunks_with_sources = process_documents(uploaded_files)
    all_chunks = [chunk.text for chunk in chunks_with_sources]
    chunk_vectors = sentence_encode(all_chunks)
    memory = ConversationMemory()

    query = st.text_input("Ask a question about your documents:")

    if st.button("Get Answer") and query:
        query_vector = sentence_encode([query])
        similarities = []
        for idx, chunk_vec in enumerate(chunk_vectors):
            sim = cosine_similarity(chunk_vec, query_vector[0])
            similarities.append((sim, idx))

        top_chunks = sorted(similarities, reverse=True)[:3]
        top_indices = [idx for _, idx in top_chunks]

        new_context = ""
        for i in top_indices:
            new_context += all_chunks[i] + "\n"

        conversation_history = memory.get_formatted_history()
        prompt_template = f"""You are a helpful assistant with access to previous conversation context and the current question.

Previous Conversation:
{conversation_history}

Current Context (from {chunks_with_sources[top_indices[0]].source}):
{new_context}

Current Question: {query}

Please provide a coherent answer that takes into account both the conversation history and the current context. Also mention which PDF file(s) contained the relevant information."""

        try:
            model = genai.GenerativeModel('gemini-2.0-flash')
            response = model.generate_content(prompt_template)
            st.subheader("💡 Answer")
            st.write(response.text)
            memory.add_interaction(query, response.text, new_context)
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")
