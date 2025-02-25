# TechScribe

TechScribe is a Retrieval-Augmented Generation (RAG) based technical documentation assistant. It leverages a local language model (llama3 7b via Ollama) along with a FAISS-based vector retrieval system to provide precise, context-aware answers by sourcing relevant information from technical documentation.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)

## Overview

TechScribe is designed to help developers and technical writers quickly locate and synthesize information from a large collection of documentation files. It works by:

1. **Indexing Documentation:** Breaking down documents into manageable chunks, generating embeddings for each chunk, and storing these in a FAISS index.
2. **Query Retrieval:** When a query is submitted, the system retrieves the most relevant chunks from the index.
3. **LLM Integration:** The retrieved chunks are then combined with the query to form a prompt that is sent to a local LLM (llama3 7b via Ollama), generating a detailed answer.

## Features

- **Document Ingestion:** Supports markdown files and can be extended to other formats.
- **Chunking & Embedding:** Splits documents into overlapping chunks and converts them into vector embeddings using Sentence Transformers.
- **Efficient Retrieval:** Utilizes FAISS for fast similarity search.
- **LLM-Powered Responses:** Integrates with a local LLM (llama3 7b via Ollama) to generate context-rich answers.
- **Extensible Design:** Easily adaptable to new types of documentation or other retrieval models.

## Architecture

TechScribe follows a modular design:

1. **Data Ingestion Module:**  
   - Loads and preprocesses documents from the `docs/` folder.
   - Splits documents into chunks with overlapping tokens.
   - Generates vector embeddings and builds a FAISS index.

2. **Query Module:**  
   - Accepts user queries.
   - Retrieves the most similar documentation chunks using the FAISS index.

3. **LLM Integration Module:**  
   - Constructs a prompt by combining retrieved context with the user query.
   - Sends the prompt to the local LLM via Ollama.
   - Returns the generated answer.


