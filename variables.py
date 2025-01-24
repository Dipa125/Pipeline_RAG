# Nome del modello usato per l'Embedding e Tokenizzatore
EMBEDDING_MODEL_NAME_L6 = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_MODEL_NAME_L12 = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


# Nome del modello usato per l'LLM
MODEL_NAME_BF32 = "sapienzanlp/modello-italia-9b"
MODEL_NAME_BF16 = "sapienzanlp/modello-italia-9b-bf16"

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

# Definizione del template per il prompt
PROMT_TEMPLATE ="""
<|system|>
Rispondi alla domanda basandoti sulla tua conoscenza. Aiutati con il seguente contesto:
{context}</s>
<|user|>
{question}</s>
<|assistant|>
 """
