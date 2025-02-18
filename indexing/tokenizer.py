from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownTextSplitter, TokenTextSplitter
from transformers import AutoTokenizer
from langchain.docstore.document import Document as LangchainDocument
from sentence_transformers import SentenceTransformer

from variables import EMBEDDING_MODEL_NAME_L6
from variables import MARKDOWN_SEPARATORS

import re

class Tokenizer:

  def __init__(self):
    max_chunk = SentenceTransformer(EMBEDDING_MODEL_NAME_L6).max_seq_length
    self.markdown_splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_chunk*5,
        chunk_overlap=int(max_chunk / 10),
        # add_start_index=True,
        strip_whitespace=True,
        separators=MARKDOWN_SEPARATORS
    )
    # self.text_splitter = MarkdownTextSplitter(
    #     chunk_size=max_chunk,
    #     chunk_overlap=int(max_chunk / 10),
    #     # add_start_index=True,
    # )

    self.token_splitter = TokenTextSplitter(
      chunk_size=max_chunk,
      chunk_overlap=int(max_chunk / 10),
      encoding_name="cl100k_base"
    )
    # self.markdown_splitter = MarkdownTextSplitter()
  
  def split_documents(self, langchain_docs: List[LangchainDocument]):
    docs_processed = []
    for doc in langchain_docs:
      #print(doc)
      docs_processed += self.markdown_splitter.split_documents([doc])
    return docs_processed







