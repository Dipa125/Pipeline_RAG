from typing import List, Dict, Any
from langchain.text_splitter import TokenTextSplitter, MarkdownHeaderTextSplitter
from transformers import AutoTokenizer
from langchain.docstore.document import Document as LangchainDocument

#from sentence_transformers import SentenceTransformer

from variables import Embedding_Model
from variables import MARKDOWN_SEPARATORS

class Tokenizer:

  def __init__(self, embedder:Embedding_Model, chunk_size=None):
    #max_chunk = SentenceTransformer(embedder.value).max_seq_length

    if chunk_size is None:
      self.chunk_size = embedder.value[1]
    elif (chunk_size > embedder.value[1]):
      raise ValueError("The chunk size cannot exceed the window size of the selected embedder.")
    else:
      self.chunk_size = chunk_size

    self.markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=MARKDOWN_SEPARATORS)
    
    self.token_splitter = TokenTextSplitter(
      chunk_size=self.chunk_size,
      chunk_overlap=int(self.chunk_size / 10),
      encoding_name="cl100k_base"
    )
  
  def _split_documents_from_markdown(self, name:str, markdown: str) -> List[LangchainDocument]:
    return [LangchainDocument(
        page_content = doc.page_content,
        metadata = doc.metadata | {"name" : name}
      ) 
      for doc in self.markdown_splitter.split_text(markdown)
    ]
  
  def tokenized_documents(self, dict_docs: Dict[str, str], markdown_chunking:bool=False) -> List[LangchainDocument]:
    docs_processed = []
    for doc in dict_docs:
      if markdown_chunking:
        split_docs = self._split_documents_from_markdown(name = doc, markdown = dict_docs[doc])
      else:
        split_docs = [LangchainDocument(page_content = doc.page_content,metadata = {"name" : doc})]
      tokenizer_docs = self.token_splitter.split_documents(split_docs)
      docs_processed.extend(tokenizer_docs)
    return docs_processed











