from langchain.vectorstores import FAISS

class Retriever:

  def __new__(cls, vectorDB):
    return vectorDB.as_retriever(search_type="similarity", search_kwargs={"k": 4})