from enum import Enum

# Path dell'intero progetto
PROJECT_PATH = "/content/Pipeline_RAG"

# File contenente tutte le dipendenze
REQUIREMENTS_PATH = "/content/Pipeline_RAG/requirements.txt"

# Nome del modello usato per l'Embedding
class Embedding_Model(Enum):
  SEQ_128 = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" 
  SEQ_256 = "sentence-transformers/all-MiniLM-L6-v2"
  SEQ_384 = "sentence-transformers/all-mpnet-base-v2"
  SEQ_512 = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"

# Nome del modello per l'image captioning
CAPTIONING_MODEL = "Salesforce/blip-image-captioning-base"

# Nome del modello usato per l'LLM
LLM_NAME = "mistralai/Mistral-7B-Instruct-v0.3"

# Marcatori consigliati da LangChain per dividere il testo in chunk
MARKDOWN_SEPARATORS = [
  ("#", "Titolo"),
  ("##", "Titolo"),
  ("###", "Titolo"),
]

# Definizione del template per il prompt italiano
PROMT_TEMPLATE_ITA = """
Rispondi alla domanda basandoti sulla tua conoscenza e sul seguente contesto.

Contesto:
{context}

Domanda:
{question}

Risposta:
"""

# Definizione del template per il prompt inglese
PROMT_TEMPLATE_ENG ="""
Answer the question based on your knowledge and the following context.

Context:
{context}

Question:
{question}

Answer:
"""
