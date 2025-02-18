from typing import List, Dict
from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownTextSplitter, TokenTextSplitter, MarkdownHeaderTextSplitter
from transformers import AutoTokenizer
from langchain.docstore.document import Document as LangchainDocument
from sentence_transformers import SentenceTransformer

from variables import EMBEDDING_MODEL_NAME_L6
from variables import MARKDOWN_SEPARATORS

class Tokenizer:

  def __init__(self):
    max_chunk = SentenceTransformer(EMBEDDING_MODEL_NAME_L6).max_seq_length
    # self.markdown_splitter = RecursiveCharacterTextSplitter(
    #     chunk_size=max_chunk*5,
    #     chunk_overlap=int(max_chunk / 10),
    #     # add_start_index=True,
    #     strip_whitespace=True,
    #     separators=MARKDOWN_SEPARATORS
    # )

    # self.text_splitter = MarkdownTextSplitter(
    #     chunk_size=max_chunk,
    #     chunk_overlap=int(max_chunk / 10),
    #     # add_start_index=True,
    # )

    # self.markdown_splitter = MarkdownTextSplitter()

    self.markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=MARKDOWN_SEPARATORS)
    
    self.token_splitter = TokenTextSplitter(
      chunk_size=max_chunk,
      chunk_overlap=int(max_chunk / 10),
      encoding_name="cl100k_base"
    )  
  
  def _split_documents_from_markdown(self, name:str, markdown: str) -> List[LangchainDocument]:
    split_docs = [
      LangchainDocument(
        page_content = doc.page_content,
        metadata = doc.metadata | {"name" : name}
      ) 
      for doc in self.markdown_splitter.split_text(markdown)
    ]
    return split_docs

  def tokenized_documents(self, dict_docs: Dict[str, str]) -> List[LangchainDocument]:
    docs_processed = []
    for doc in dict_docs:
      split_docs = self._split_documents_from_markdown(name = doc, markdown = dict_docs[doc])
      tokenizer_docs = self.token_splitter.split_documents(split_docs)
      docs_processed.extend(tokenizer_docs)
    return docs_processed











