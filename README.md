# RAG-based Equipment Maintenance App

This is a fully open-source Retrieval-Augmented Generation (RAG) app designed for querying equipment maintenance documents. Users can upload files (e.g., PDF, TXT, CSV), and the system will process them, store the information, and answer questions based on the uploaded documents.

## Features

- **Upload Files**: Supports PDFs, TXT, and CSV files.
- **Document Search**: Uses FAISS for vector similarity search.
- **Open-Source LLM**: Uses llama or any other HuggingFace-supported LLM for query responses.
- **Embeddings**: Leverages SentenceTransformers (`all-MiniLM-L6-v2`) for creating document embeddings.

## Installation

### Backend

1. Navigate to the project folder.
2. Install dependencies:

   ```bash
   pipenv run pip install -r requirements.txt
   pipenv run streamlit run app.py