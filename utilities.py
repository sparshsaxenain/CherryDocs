"""Helper utilities for the CherryDocs application."""

import os
import json
import hashlib
import re
from io import BytesIO

import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
import fitz
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from ollama import embed, chat
from pydantic import BaseModel
from sklearn.metrics.pairwise import cosine_distances

load_dotenv()

embedding_model = os.getenv("EMBEDDING_MODEL")
db_query_prompt = os.getenv("DB_QUERY_PROMPT")

class SearchQueries(BaseModel):
    """Schema used when requesting search queries from the language model."""

    search_query1: str
    search_query2: str
    search_query3: str
    search_query4: str
    search_query5: str

def extract_text_from_pdf(file):
    """Return the textual content of a PDF uploaded through Streamlit."""

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
    """Render a list of chat messages in the Streamlit interface."""

    for message in chat_history:
        if message["role"] == "user":
            st.chat_message("user").write(message["content"])
        elif message["role"] == "assistant":
            st.chat_message("assistant").write(message["content"])

def create_chroma_client(path: str | None = None):
    """Return a persistent ChromaDB client stored at *path*."""

    chroma_client = chromadb.PersistentClient(path=path)
    return chroma_client

def get_sha512_hash(text: str) -> str:
    """Return a SHA-512 hash of *text* used for unique document IDs."""

    return hashlib.sha512(text.encode("utf-8")).hexdigest()

def generate_embeddings(sentence: str):
    """Generate an embedding vector for a sentence using Ollama."""

    response = embed(model=embedding_model, input=sentence)
    return response

class OllamaEmbeddingFunction(EmbeddingFunction):
    """Adapter that allows ChromaDB to use Ollama for embeddings."""

    def __call__(self, input: Documents) -> Embeddings:
        response = embed(model=embedding_model, input=input)
        return response["embeddings"]

def semantic_chunking(
    documents: list,
    buffer_size: int = 1,
    breakpoint_percentile_threshold: int = 80,
) -> list:
    """Split documents into semantically coherent chunks.

    Sentences are embedded and compared for semantic distance. Whenever the
    distance crosses the configured percentile threshold a new chunk begins.

    Parameters
    ----------
    documents : list[str]
        Input documents as plain strings.
    buffer_size : int, optional
        Number of neighbouring sentences to include when calculating
        embeddings. Defaults to 1.
    breakpoint_percentile_threshold : int, optional
        Threshold percentile used to determine chunk boundaries. Defaults to 80.

    Returns
    -------
    list[str]
        List of chunked document strings.
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
            my_bar.progress(percent_complete, text=f"{progress_text} {percent_complete*100:.2f}% completed")
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

def generate_db_queries(user_prompt: str, chat_history) -> list:
    """Ask the LLM to produce search queries for retrieving documents."""
    response = chat(
        messages=[
            {
                'role': 'system',
                'content': db_query_prompt
            },
            *chat_history[min(-10, -len(chat_history)):],  # Use the last 10 messages from chat history
            {
                'role': 'user',
                'content': user_prompt
            }
        ],
        model = 'llama3.1:8b',
        format = SearchQueries.model_json_schema(),
    )

    queries = json.loads(response.message.content)
    query_list = [queries[key] for key in queries.keys() if queries[key]]
    return query_list

def retrieve_documents_from_db(
    db,
    user_query: str,
    chat_history,
    n_results: int = 5,
) -> list:
    """Return a set of documents from the vector store that match the query."""
    if not user_query:
        return []

    queries = generate_db_queries(user_query, chat_history)
    relevant_docs = db.query(query_texts = queries, n_results = n_results)
    # Step 1: Flatten the list
    flattened = [item for sublist in relevant_docs['documents'] for item in sublist]

    # Step 2: Remove duplicates while maintaining order
    unique_items = list(dict.fromkeys(flattened))

    return unique_items

def generate_llm_response(user_query: str, chat_history, db) -> str:
    """Generate a response from the LLM using retrieved documents as context."""
    if not user_query:
        return "Please provide a query to generate a response."
    
    query_oneline = user_query.replace("\n", " ")

    relevant_docs = retrieve_documents_from_db(db, user_query, chat_history)
    passages = ""
    for passage in relevant_docs:
        passages += f"PASSAGE: {passage}\n"
    
    if not relevant_docs:
        return "No relevant documents found for the given query."
    
    user_query = f"# **PASSAGES**:\n {passages} \n\n# **USER QUERY**: " + query_oneline
    print(user_query)
    response = chat(
        messages=[
            {
                'role': 'system',
                'content': "You are a helpful and informative assistant, Your job is to answer questions as accurately and helpfully as possible using the provided passages (DON'T MENTION ANYTHING ABOUT PASSAGES GIVEN). Always prefer being helpful over being literal. If the passages are irrelevant, IGNORE them completely and say 'I don't know'."
            },
            *chat_history[min(-10, -len(chat_history)):],  # Use the last 10 messages from chat history
            {
                'role': 'user',
                'content': user_query
            }
        ],
        model = 'llama3.1:8b',
    )

    return response.message.content
