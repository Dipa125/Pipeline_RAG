# Nome del modello usato per l'Embedding
EMBEDDING_MODEL_NAME = "dbmdz/bert-base-italian-xxl-cased"

# Nome del modello usato per il Tokenizer
TOKENIZER_MODEL_NAME = "dbmdz/bert-base-italian-xxl-cased"

# Nome del modello usato per l'LLM
MODEL_NAME = "sapienzanlp/modello-italia-9b"

# marcatori consigliati da LangChain per dividere il testo in chunk
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
