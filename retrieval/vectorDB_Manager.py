import os
from typing import List

from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings

from variables import Embedding_Model

class VectorDB_Manager:

  def __init__(self, embedder:Embedding_Model, results=3, key_GPT=None, docs=None, load_path=None):

    #---Embedder---
    if embedder == Embedding_Model.GPT:
      if key_GPT is None:
        raise ValueError("To use an OpenAI embedder, a valid API key is required.")
      else:
        self.embedding_model = OpenAIEmbeddings(
          model = Embedding_Model.GPT.value[0],
          openai_api_key = key_GPT
        )      
    else:
      self.embedding_model = HuggingFaceEmbeddings(
        model_name = embedder.value[0],
        multi_process = True,
        model_kwargs = {"device": "cuda"},
        encode_kwargs = {"normalize_embeddings": True},
      )

    #---VectorDB---
    if load_path:
      self.vectorDB = self._load_vectorDB(load_path)
    elif docs:
      self.vectorDB = FAISS.from_documents(
        docs,
        self.embedding_model,
        distance_strategy=DistanceStrategy.COSINE
        )
    else:
      raise ValueError("path or document needed for initialization")
    
    #---Retrieval---
    self.retrieval = self.vectorDB.as_retriever(search_type="similarity", search_kwargs={"k": results})
  
  def get_similar_documents(self, query:str) -> List[str]:
    similar_docs = self.retrieval.invoke(query)
    return [doc.metadata["name"] for doc in similar_docs]

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
      os.makedirs(save_path, exist_ok=True)
      self.vectorDB.save_local(save_path)

  def _load_vectorDB(self, load_path):
    if not os.path.exists(os.path.join(load_path, "index.faiss")):
        raise FileNotFoundError(f"'index.faiss' not found in {load_path}")
    if not os.path.exists(os.path.join(load_path, "index.pkl")):
        raise FileNotFoundError(f"'index.pkl' non trovato in {load_path}")

    if os.path.exists(load_path):
      return FAISS.load_local(load_path, self.embedding_model, allow_dangerous_deserialization=True)



