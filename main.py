import requests
from query import load_index, search
from sentence_transformers import SentenceTransformer

def construct_prompt(query, contexts):
    """Constructs a prompt with retrieved context for the LLM."""
    context_text = "\n\n".join(contexts)
    prompt = (
        f"Using the following technical documentation excerpts:\n"
        f"{context_text}\n\n"
        f"Answer the following question:\n{query}\n"
    )
    return prompt

def query_ollama(prompt):
    """
    Sends the prompt to your local Ollama/llama3 7b model.
    Adjust the URL, model name, and payload as needed.
    """
    url = "http://localhost:11434/api/generate"  # Update this if needed.
    payload = {
        "model": "llama3-7b",
        "prompt": prompt,
        "max_tokens": 200,
    }
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        return response.json().get("response", "")
    else:
        return f"Error: {response.status_code} - {response.text}"

if __name__ == "__main__":
    # Load the FAISS index and texts.
    index, texts, metadata = load_index()
    model = SentenceTransformer("all-MiniLM-L6-v2")
    query_input = input("Enter your technical query: ")
    # Retrieve context from documentation.
    contexts = search(query_input, model, index, texts, k=5)
    # Build the prompt.
    prompt = construct_prompt(query_input, contexts)
    print("Generated Prompt:\n", prompt)
    # Query your local LLM via Ollama.
    answer = query_ollama(prompt)
    print("\nAnswer from LLM:\n", answer)
