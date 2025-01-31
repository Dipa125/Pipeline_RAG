import os

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy

from variables import EMBEDDING_MODEL_NAME_L6

class VectorDB_Manager:

  def __init__(self, docs, load_path=None):
    self.embedding_model = HuggingFaceEmbeddings(
      model_name=EMBEDDING_MODEL_NAME_L6,
      multi_process=True,
      model_kwargs={"device": "cuda"},
      encode_kwargs={"normalize_embeddings": True},
      )

    if load_path is None:
      self.vectorDB = FAISS.from_documents(
        docs,
        self.embedding_model,
        distance_strategy=DistanceStrategy.COSINE
        )
    else:
      self.vectorDB = self._load_vectorDB(load_path)

# Si può rendere parametrico anche la scelta della similarità e il numero di documenti estratti
  def retrieval(self):
    return self.vectorDB.as_retriever(search_type="similarity", search_kwargs={"k": 4})

  def add_docs(self, new_docs):
    self.vectorDB.add_documents(new_docs)

  def clear_vectorDB(self, new_docs):
    self.vectorDB = FAISS.from_documents(
      new_docs,
      self.embedding_model,
      distance_strategy=DistanceStrategy.COSINE
      )

  def save_vectorDB(self, save_path):
    if self.vectorDB:
      os.makedirs(os.path.dirname(save_path), exist_ok=True)
      self.vectorDB.save_local(save_path)

  def _load_vectorDB(self, load_path):
    if os.path.exists(load_path):
      self.vectorDB = FAISS.load_local(load_path, self.embedding_model)



