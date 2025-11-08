# app.py

import streamlit as st
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Step 1 - PDF Extraction

def extract_pdf_text(pdf_file):
    doc = fitz.open(pdf_file)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# step 2 - Creation of Embeddings + Storage in FAISS db
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

def embed_text(text_chunks, embed_model):
    return embed_model.encode(text_chunks, convert_to_numpy=True)

def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

def retrieve(query, index, text_chunks, embed_model, top_k=5):
    query_emb = embed_model.encode([query], convert_to_numpy=True)
    D, I = index.search(query_emb, k=top_k)
    return "\n\n".join([text_chunks[i] for i in I[0]])


#using free and open source model Hugging Face LLM

@st.cache_resource
def load_llm():
    model_name = "tiiuae/falcon-7b-instruct" 
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    nlp = pipeline("text-generation", model=model, tokenizer=tokenizer, max_length=512)
    return nlp

def answer_question(query, context, llm):
    prompt = f"""
Use ONLY the following text to answer the question. If answer not present, say "I don't know".

Context:
{context}

Question: {query}
Answer:
"""
    response = llm(prompt, max_length=300, do_sample=True)
    return response[0]['generated_text'].split("Answer:")[-1].strip()


# Building the User Interface using Streamlit UI

st.set_page_config(page_title="PDF RAG Chatbot")
st.title("📄Bajaj Finserv ChatBot")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    with st.spinner("Extracting text from PDF..."):
        text = extract_pdf_text(uploaded_file)
        st.success(f"PDF processed! {len(text.split())} words extracted.")

        # Splitting the text into chunks
        chunk_size = 1000
        text_chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

        embed_model = load_embedding_model()
        embeddings = embed_text(text_chunks, embed_model)
        index = build_faiss_index(np.array(embeddings))

        llm = load_llm()

    query = st.text_input("Ask a question about the PDF:")

    if query:
        with st.spinner("Fetching answer..."):
            context = retrieve(query, index, text_chunks, embed_model)
            answer = answer_question(query, context, llm)
        st.markdown("**Answer:**")
        st.write(answer)
