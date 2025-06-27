import fitz
from ollama import embed
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from io import BytesIO
import streamlit as st
import hashlib
import re
from sklearn.metrics.pairwise import cosine_distances
import numpy as np
import time

def extract_text_from_pdf(file):
    file_bytes = file.read()
    if not file_bytes:
        raise ValueError("Uploaded file is empty.")
    
    pdf_stream = BytesIO(file_bytes)
    
    text = ""
    with fitz.open(stream=pdf_stream, filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text

def show_chat_history(chat_history):
    for message in chat_history:
        if message["role"] == "user":
            st.chat_message("user").write(message["content"])
        elif message["role"] == "assistant":
            st.chat_message("assistant").write(message["content"])

def create_chroma_client(path=None):
    chroma_client = chromadb.PersistentClient(path=path)
    return chroma_client

def get_sha256_hash(text: str) -> str:
    """Why use SHA-256?
    1. Ensures uniqueness of each document ID.
    2. Avoids collisions better than simple counters.
    3. Can help deduplicate if the same document is added again (ChromaDB may skip or overwrite based on implementation)."""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

def generate_embeddings(sentence):
    response = embed(
        model="bge-m3:latest",
        input=sentence,
    )

    return response

class OllamaEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        response = embed(model='bge-m3', input=input)
        return response['embeddings']

def semantic_chunking(documents, buffer_size=2, breakpoint_percentile_threshold=80):
    """
    Splits a list of documents into semantically coherent chunks based on changes in semantic similarity
    between consecutive sentences.

    Args:
        documents (list): A list of full document texts (strings).
        buffer_size (int): Number of surrounding sentences to include when computing context (default = 2).
        breakpoint_percentile_threshold (int): Percentile threshold to decide where semantic breaks occur (default = 80).

    Returns:
        list: List of chunked document strings, each containing semantically grouped sentences.
    """

    # Split all documents into sentences using punctuation as delimiters
    sentences = []
    for docs in documents:
        sentences += re.split(r'(?<=[.?!])\s+', docs)

    print(f"{len(sentences)} sentences were found")

    # Store sentence and its index
    sentences_indexed = [{'sentence': x, 'index': i} for i, x in enumerate(sentences)]

    # Create a buffer around each sentence for better context during embedding
    for i in range(len(sentences_indexed)):
        combine_sentences = ''
        for j in range(i - buffer_size, i + buffer_size + 1):
            if 0 <= j < len(sentences_indexed):
                combine_sentences += sentences_indexed[j]['sentence'] + ' '
        sentences_indexed[i]['combined_sentences'] = combine_sentences

    # Generate embeddings for each combined sentence using semantic similarity model
    batch_response = []
    progress_text = "Performing Magic. Please wait."
    my_bar = st.progress(0, text=progress_text)
    for x in sentences_indexed:
        response = generate_embeddings(x['combined_sentences'])
        batch_response.append(response.embeddings)
        if len(batch_response)%50 == 0:
            percent_complete = (len(batch_response) / len(sentences_indexed))
            my_bar.progress(percent_complete, text=f"{progress_text} {percent_complete:.2f}% completed")
    # Store embeddings back into the indexed sentence structure
    for i, sentence in enumerate(sentences_indexed):
        sentences_indexed[i]['combined_sentence_embedding'] = batch_response[i][0]

    # Calculate semantic distances between each sentence and the next one
    distances = []
    for i in range(len(sentences_indexed) - 1):
        distance = cosine_distances(
            [sentences_indexed[i]['combined_sentence_embedding']],
            [sentences_indexed[i + 1]['combined_sentence_embedding']]
        )[0][0]
        distances.append(distance)
        sentences_indexed[i]['distance_to_next'] = distance

    # Determine threshold for breaking into chunks
    breakpoint_distance_threshold = np.percentile(distances, breakpoint_percentile_threshold)
    indices_above_thresh = [i for i, x in enumerate(distances) if x > breakpoint_distance_threshold]

    # Group sentences into chunks based on high distance breakpoints
    start_index = 0
    chunks = []

    for index in indices_above_thresh:
        end_index = index
        group = sentences_indexed[start_index:end_index + 1]
        combined_text = ' '.join([d['sentence'] for d in group])
        chunks.append(combined_text)
        start_index = index + 1

    # Add the last remaining chunk if there are leftover sentences
    if start_index < len(sentences_indexed):
        combined_text = ' '.join([d['sentence'] for d in sentences_indexed[start_index:]])
        chunks.append(combined_text)
    my_bar.empty()

    return chunks