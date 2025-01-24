from langchain.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy

import sys
sys.path.append('/content/Pipeline_RAG')

from variables import EMBEDDING_MODEL_NAME_L6

class Embedder:
  def __init__(self):
    self.embedding_model = HuggingFaceEmbeddings(
      model_name=EMBEDDING_MODEL_NAME_L6,
      multi_process=True,
      model_kwargs={"device": "cuda"},
      encode_kwargs={"normalize_embeddings": True},
      )
  
  def create_vectorDB(self, docs):
    return FAISS.from_documents(
      docs, self.embedding_model, distance_strategy=DistanceStrategy.COSINE
      )