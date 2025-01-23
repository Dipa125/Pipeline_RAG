# Nome del modello usato per il Tokenizer
EMBEDDING_MODEL_NAME = "dbmdz/bert-base-italian-xxl-cased"

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
