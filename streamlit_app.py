"""CherryDocs main Streamlit application.

This app allows a user to upload text based files, store them in a ChromaDB
vector store and chat with an LLM over the uploaded documents.
"""

import os
import uuid
import shutil
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
import time

from utilities import (
    extract_text_from_pdf,
    show_chat_history,
    create_chroma_client,
    get_sha512_hash,
    semantic_chunking,
    OllamaEmbeddingFunction,
    generate_llm_response,
)

# Load environment
# Load environment variables from a .env file. The application expects at least
# the following variables:
#   DB_NAME        - Name of the Chroma collection.
#   CHROMA_DB_PATH - Filesystem path where the persistent database is stored.
load_dotenv()

db_name = os.getenv("DB_NAME")
print(f"Using database: {db_name}")
db_path = os.getenv("CHROMA_DB_PATH")

# Create (or connect to) the Chroma persistent client and collection. The
# OllamaEmbeddingFunction wraps the `ollama embed` API so that Chroma can
# generate embeddings for new documents automatically.
db_client = create_chroma_client(path=db_path)
embed_fn = OllamaEmbeddingFunction()
db = db_client.get_or_create_collection(
    name=db_name, embedding_function=embed_fn
)

# Configure the Streamlit page.
st.set_page_config(
    page_title = "CherryDocs",
    page_icon = ":cherries:",
    layout = "wide",
    initial_sidebar_state = "expanded",
)

st.title("CherryDocs :cherries:")

# Keep chat history between interactions so the model has context.
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Store the list of uploaded files so that they can be shown in the sidebar.
if "sources" not in st.session_state:
    st.session_state.sources = []

# Input area at the bottom of the page. Users can upload multiple files and
# optionally provide a text prompt.
prompt = st.chat_input(
    "Say something and/or attach a file to get started!",
    accept_file="multiple",
    file_type=["txt", "pdf", "csv"],
)

# List of extracted texts from uploaded files. These texts will be chunked and
# stored in the vector database.
documents = []

# Display previous conversation so the user can see the context.
if st.session_state.get("chat_history") is not None:
    show_chat_history(st.session_state.chat_history)

# Show a list of uploaded source files in the sidebar.
if st.session_state.get("sources") is not None:
    st.sidebar.title("Sources:")
    with st.sidebar:
        for source in st.session_state.sources:
            st.write(source)

# When files are uploaded process each of them and store the extracted text in
# the database.
if prompt and prompt["files"]:
    with st.sidebar:
        for file in prompt["files"]:
            st.session_state.sources.append(f"- {file.name} ({file.type})")
            st.write(f"- {file.name} ({file.type})")

    # Show a progress bar while the files are processed.
    chunking_start_time = time.time()
    percent_complete = 0
    my_bar = st.progress(percent_complete, text="Processing files...")
    for file in prompt["files"]:
        if file.type == "application/pdf":
            text = extract_text_from_pdf(file)
            documents.append(text)

        elif file.type == "text/plain":
            text = file.getvalue().decode("utf-8")
            documents.append(text)

        elif file.type == "text/csv":
            df = pd.read_csv(file)
            text = df.to_string(index=False)
            documents.append(text)

        # Update progress for each processed file
        my_bar.progress(percent_complete + 1, text=f"Processing {file.name}...")
    my_bar.empty()

    # Break large texts into semantically meaningful chunks and store them
    # in the Chroma collection.
    chunks = semantic_chunking(documents)
    chunking_end_time = time.time()
    chunking_duration = chunking_end_time - chunking_start_time
    ids = [f"{get_sha512_hash(chunk)}-{uuid.uuid4()}" for chunk in chunks]
    db.add(documents=chunks, ids=ids)
    st.write(f"Processed {len(documents)} files in {chunking_duration:.2f} seconds.")
    if prompt["text"]:
        st.chat_message("user").write(prompt["text"])
        st.session_state.chat_history.append({
            "role": "user",
            "content": prompt["text"]
        })

        # Generate a response using the uploaded documents as additional context
        response = generate_llm_response(
            user_query = prompt["text"],
            chat_history = st.session_state.chat_history,
            db = db
        )

        st.chat_message("assistant").write(response)
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": response
        })

elif prompt and prompt["text"]:
    # If no files were uploaded just generate a response from the existing
    # knowledge base.
    st.chat_message("user").write(prompt["text"])
    st.session_state.chat_history.append({
        "role": "user",
        "content": prompt["text"]
    })
    response = generate_llm_response(
        user_query=prompt["text"],
        chat_history=st.session_state.chat_history,
        db=db,
    )

    st.chat_message("assistant").write(response)
    st.session_state.chat_history.append({
        "role": "assistant",
        "content": response
    })
