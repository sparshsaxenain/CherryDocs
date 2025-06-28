# CherryDocs

CherryDocs is a simple document question answering demo built with [Streamlit](https://streamlit.io/) and [Chroma](https://www.trychroma.com/).  Uploaded text based files are converted to chunks, embedded with an Ollama model and stored in a persistent vector database.  A user can then ask questions in the chat interface and the system will retrieve the most relevant chunks and ask an LLM to generate a response.

## Features

- Supports uploading **PDF**, **CSV** and **TXT** files.
- Documents are automatically split into semantic chunks and stored in ChromaDB.
- Uses Ollama for embeddings and chat completions.
- Maintains chat history during the session.

## Installation

1. Install Python 3.10 or newer.
2. Install the required packages:

```bash
pip install streamlit chromadb pydantic python-dotenv pymupdf pandas scikit-learn numpy ollama
```

3. Create a `.env` file in the project directory and set the following variables:

```
DB_NAME=cherrydocs
CHROMA_DB_PATH=/path/to/chroma
EMBEDDING_MODEL=<name of ollama embedding model>
DB_QUERY_PROMPT=<system prompt used when generating search queries>
```

## Running

Launch the Streamlit app with:

```bash
streamlit run streamlit_app.py
```

A browser window will open where you can start chatting and upload files.

## Project Structure

- `streamlit_app.py` – main Streamlit interface.
- `utilities.py` – helper functions for text extraction, semantic chunking and interaction with the LLM and database.
- `.streamlit/config.toml` – Streamlit theme configuration.

## Usage

1. Run the application using the command above.
2. Type a question or upload documents in the chat input field. Multiple files can be uploaded at once.
3. Uploaded files appear in the sidebar under **Sources**.
4. Ask questions about the uploaded content. The assistant will retrieve relevant passages and answer using the LLM.

## License

This project is provided as-is for demonstration purposes.

