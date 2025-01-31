# Path dell'intero progetto
PROJECT_PATH = "/content/Pipeline_RAG"

# File contenente tutte le dipendenze
REQUIREMENTS_PATH = "/content/Pipeline_RAG/requirements.txt"

# Nome del modello usato per l'Embedding e Tokenizzatore
EMBEDDING_MODEL_NAME_L6 = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_MODEL_NAME_L12 = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


# Nome del modello usato per l'LLM
MODEL_NAME_FACEBOOK = "facebook/opt-1.3b"

# Marcatori consigliati da LangChain per dividere il testo in chunk
MARKDOWN_SEPARATORS = [
    "\n#{1,6} ",
    "```\n",
    "\n\\*\\*\\*+\n",
    "\n---+\n",
    "\n___+\n",
    "\n\n",
    "\n",
    " ",
    "",
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
