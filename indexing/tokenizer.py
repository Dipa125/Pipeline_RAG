import sys
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import Optional, List, Tuple
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
from langchain.docstore.document import Document as LangchainDocument

sys.path.append('/content/Pipeline_RAG')
from variables import EMBEDDING_MODEL_NAME
from variables import MARKDOWN_SEPARATORS



class Tokenizer:

  def __init__(self):
    max_chunk = SentenceTransformer(EMBEDDING_MODEL_NAME).max_seq_length
    self.text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME),
        chunk_size=max_chunk,
        chunk_overlap=int(max_chunk / 10),
        add_start_index=True,
        strip_whitespace=True,
        separators=MARKDOWN_SEPARATORS,
    )
  
  def split_documents(self, docs):
    RAW_KNOWLEDGE_BASE = [
      LangchainDocument(page_content=docs[doc], metadata={}) for doc in docs
    ]
    docs_processed = []
    for doc in RAW_KNOWLEDGE_BASE:
        docs_processed += self.text_splitter.split_documents([doc])

    return docs_processed







