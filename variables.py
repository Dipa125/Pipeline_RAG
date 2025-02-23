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

class Prompt_Template(Enum):
  NO_CONTEXT = """
Sei un assistente virtuale che ha lo scopo di rispondere in maniera semplice
e chiara alla domanda che trovi qui sotto

Domanda:
{question}

Risposta:
"""
  DOCUMENT_CONTEXT = """
Sei un assistente virtuale che ha lo scopo di fornire, tra tutti i Document che trovi
nel Contesto, un riassunto del perché il bando rispecchia le caratteristiche fornite
dall'utente che trovi nella sezione Domanda. Inoltre, prima del riassuto metti il
nome del bando che hai scelto come il più pertinente.

Contesto:
{context}

Domanda:
{question}

Risposta:
"""
  CHAT_CONTEXT ="""
Sei un assistente virtuale impiegato in una chat in cui devi aiutare l'utente ad
estrarre informazioni utili da dei bandi pubblici. Ti verrà fornita la memoria della
conversazione e dei documetni da usare come contesto. Rispondi alla Domanda in maniera
corretta secondo le infoamzioni fornite.

Domanda:
{question}

Risposta:
"""


