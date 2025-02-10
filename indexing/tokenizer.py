from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownTextSplitter
from transformers import AutoTokenizer
from langchain.docstore.document import Document as LangchainDocument
from sentence_transformers import SentenceTransformer

from variables import EMBEDDING_MODEL_NAME_L6
from variables import MARKDOWN_SEPARATORS

class Tokenizer:

  def __init__(self):
    max_chunk = SentenceTransformer(EMBEDDING_MODEL_NAME_L6).max_seq_length
    # self.text_splitter = RecursiveCharacterTextSplitter(
    #     chunk_size=max_chunk,
    #     chunk_overlap=int(max_chunk / 10),
    #     # add_start_index=True,
    #     strip_whitespace=True,
    #     separators=MARKDOWN_SEPARATORS,
    # )
    self.text_splitter = MarkdownTextSplitter(
        chunk_size=max_chunk,
        chunk_overlap=int(max_chunk / 10),
        # add_start_index=True,
    )
  
  def split_documents(self, langchain_docs: List[LangchainDocument]):
    docs_processed = []
    for doc in langchain_docs:
        docs_processed += self.text_splitter.split_documents([doc])
    return docs_processed







