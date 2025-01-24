from langchain.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy

from variables import EMBEDDING_MODEL_NAME_L6

class VectorDB_Manager:

  def __init__(self, docs):
    self.embedding_model = HuggingFaceEmbeddings(
      model_name=EMBEDDING_MODEL_NAME_L6,
      multi_process=True,
      model_kwargs={"device": "cuda"},
      encode_kwargs={"normalize_embeddings": True},
      )

    self.vectorDB = FAISS.from_documents(
      docs,
      self.embedding_model,
      distance_strategy=DistanceStrategy.COSINE
      )

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

# Esistono delle funzioni per caricare e salvare localmente il vectorDB


