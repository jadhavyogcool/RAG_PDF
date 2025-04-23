import fitz  
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

qa_pipeline = pipeline("text-generation", model="gpt2")

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def chunk_text(text, chunk_size=1000):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

def embed_chunks(chunks):
    return embedding_model.encode(chunks)

def create_vector_store(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

def retrieve_chunks(query, chunks, index, chunk_embeddings, k=3):
    query_vec = embedding_model.encode([query])
    D, I = index.search(query_vec, k)
    return [chunks[i] for i in I[0]]

def generate_answer(context, query):
    prompt = f"Context: {context}\nQuestion: {query}\nAnswer:"
    result = qa_pipeline(prompt, max_length=1000, do_sample=True)[0]['generated_text']
    return result.strip().split("Answer:")[-1].strip()

def process_pdf_and_query(pdf_path, query):
    text = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(text)
    chunk_embeddings = embed_chunks(chunks)
    index = create_vector_store(np.array(chunk_embeddings))
    relevant_chunks = retrieve_chunks(query, chunks, index, chunk_embeddings)
    context = "\n".join(relevant_chunks)
    answer = generate_answer(context, query)
    return answer
