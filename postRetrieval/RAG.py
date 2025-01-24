from langchain_core.runnables import RunnablePassthrough

class RAG:
  def __new__(cls, retrieval, llm_chain):
    return {"context": retrieval, "question": RunnablePassthrough()} | llm_chain