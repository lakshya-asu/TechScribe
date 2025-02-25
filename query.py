import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer

def load_index():
    """Loads the saved FAISS index and supporting data."""
    index = faiss.read_index("faiss_index.idx")
    with open("texts.pkl", "rb") as f:
        texts = pickle.load(f)
    with open("metadata.pkl", "rb") as f:
        metadata = pickle.load(f)
    return index, texts, metadata

def search(query, model, index, texts, k=5):
    """Encodes the query and retrieves the top k similar text chunks."""
    query_embedding = model.encode([query])
    query_embedding = np.array(query_embedding).astype("float32")
    distances, indices = index.search(query_embedding, k)
    results = []
    for i in indices[0]:
        results.append(texts[i])
    return results

if __name__ == "__main__":
    index, texts, metadata = load_index()
    model = SentenceTransformer("all-MiniLM-L6-v2")
    query_input = input("Enter your query: ")
    results = search(query_input, model, index, texts, k=5)
    for i, res in enumerate(results):
        print(f"Result {i+1}:\n{res}\n{'-'*40}")
