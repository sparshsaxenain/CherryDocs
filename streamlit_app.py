import streamlit as st
from utilities import extract_text_from_pdf, show_chat_history, create_chroma_client, get_sha256_hash, semantic_chunking, generate_embeddings, OllamaEmbeddingFunction
import pandas as pd
from ollama import generate
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
# api_key = os.getenv("API_KEY")
db_name = os.getenv("DB_NAME")
print(f"Using database: {db_name}")
db_path = os.getenv("CHROMA_DB_PATH")
db_client = create_chroma_client(path=db_path)
embed_fn = OllamaEmbeddingFunction()
db = db_client.get_or_create_collection(name=db_name, embedding_function=embed_fn)

st.set_page_config(
    page_title = "CherryDocs",
    page_icon = ":cherries:",
    layout = "wide",
    initial_sidebar_state = "expanded",
)

st.title("CherryDocs :cherries:")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "sources" not in st.session_state:
    st.session_state.sources = []

prompt = st.chat_input(
    "Say something and/or attach a file to get started!",
    accept_file = "multiple",
    file_type = ["txt", "pdf", "csv"],
)

documents = []

if st.session_state.get("chat_history") is not None:
    show_chat_history(st.session_state.chat_history)
    # st.write(db.peek())

if st.session_state.get("sources") is not None:
    st.sidebar.title("Sources:")
    with st.sidebar:
        for source in st.session_state.sources:
            st.write(source)

if prompt and prompt["text"]:
    st.chat_message("user").write(prompt["text"])
    st.session_state.chat_history.append({
        "role": "user",
        "content": prompt["text"]
    })

if prompt and prompt["files"]:
    with st.sidebar:
        for file in prompt["files"]:
            st.session_state.sources.append(f"- {file.name} ({file.type})")
            st.write(f"- {file.name} ({file.type})")
    percent_complete = 0
    my_bar = st.progress(percent_complete, text= "Processing files...")
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
        my_bar.progress(percent_complete + 1, text= f"Processing {file.name}...")
    my_bar.empty()
    chunks = semantic_chunking(documents)
    db.add(documents=chunks, ids=[get_sha256_hash(chunks[i]) for i in range(len(chunks))])

    if prompt["text"]:
        st.chat_message("user").write(prompt["text"])
        st.session_state.chat_history.append({
            "role": "user",
            "content": prompt["text"]
        })
        # Generate response using Ollama