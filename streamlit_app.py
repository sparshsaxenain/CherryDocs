import streamlit as st
from utilities import extract_text_from_pdf, OllamaEmbeddingFunction
import pandas as pd
from ollama import generate


st.set_page_config(
    page_title="CherryDocs",
    page_icon=":cherries:",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("CherryDocs :cherries:")

prompt = st.chat_input(
    "Say something and/or attach a file to get started!",
    accept_file="multiple",
    file_type=["txt", "pdf", "csv"],
)

documents = []

if prompt and prompt["files"]:
    with st.sidebar:
        st.title("Sources:")
        for file in prompt["files"]:
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