import os
import glob
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

def load_documents(doc_dir):
    """Loads all markdown files from the docs directory."""
    documents = []
    file_paths = glob.glob(os.path.join(doc_dir, "*.md"))
    for file_path in file_paths:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            documents.append((file_path, content))
    return documents

def chunk_text(text, chunk_size=500, overlap=100):
    """Splits text into chunks with a given size and overlap."""
    tokens = text.split()
    chunks = []
    i = 0
    while i < len(tokens):
        chunk = " ".join(tokens[i:i+chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks

def build_index(documents, model_name="all-MiniLM-L6-v2"):
    """Builds a FAISS index from document chunks."""
    model = SentenceTransformer(model_name)
    texts = []
    metadata = []
    for file_path, content in documents:
        chunks = chunk_text(content)
        for chunk in chunks:
            texts.append(chunk)
            metadata.append({"file": file_path})
    print("Generating embeddings...")
    embeddings = model.encode(texts, show_progress_bar=True)
    embeddings = np.array(embeddings).astype("float32")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index, texts, metadata

if __name__ == "__main__":
    doc_dir = "../docs"
    documents = load_documents(doc_dir)
    index, texts, metadata = build_index(documents)
    # Save the FAISS index and supporting data for retrieval
    faiss.write_index(index, "faiss_index.idx")
    with open("texts.pkl", "wb") as f:
        pickle.dump(texts, f)
    with open("metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)
    print("Index built and saved!")
