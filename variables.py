from enum import Enum

# Path dell'intero progetto
PROJECT_PATH = "/content/Pipeline_RAG"

# File contenente tutte le dipendenze
REQUIREMENTS_PATH = "/content/Pipeline_RAG/requirements.txt"

# Nome del modello usato per l'Embedding
class Embedding_Model(Enum):
  ST_128 = ("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",128)
  ST_256 = ("sentence-transformers/all-MiniLM-L6-v2", 256)
  ST_384 = ("sentence-transformers/all-mpnet-base-v2", 384)
  ST_512 = ("sentence-transformers/multi-qa-MiniLM-L6-cos-v1", 512)
  GPT = ("text-embedding-ada-002", 8192)

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
Sei un assistente virtuale che ha lo scopo di fornire per ognuno dei Document che trovi nel Contesto
un riassunto del perché quel bando rispecchia le caratteristiche fornite dall'utente nella sezione
Domanda. Inoltre, prima di ogni riassuto metti il nome del bando che trovi nei metadata. 

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
