from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer
from langchain.docstore.document import Document as LangchainDocument

import sys
sys.path.append('/content/Pipeline_RAG')

from variables import TOKENIZER_MODEL_NAME
from variables import MARKDOWN_SEPARATORS



class Tokenizer:

  def __init__(self):
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL_NAME)
    max_chunk = tokenizer.model_max_length
    self.text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer,
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







