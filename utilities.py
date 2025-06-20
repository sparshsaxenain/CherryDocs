import fitz
from ollama import embed
from chromadb import Documents, EmbeddingFunction, Embeddings
from io import BytesIO

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

class OllamaEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        response = embed(model='bge-m3', input=input)
        return response['embeddings']