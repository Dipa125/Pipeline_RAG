from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain.chains import ChatVectorDBChain
from langchain.memory import ConversationBufferMemory

from variables import Prompt_Template

class RAG:
  def __new__(cls, retrieval, llm_chain):
    return {"context": retrieval, "question": RunnablePassthrough()} | llm_chain

# Rivedere i prompt
class RAG_Builder:
  def __init__(self, llm, retrieval):
    self.llm = llm
    self.retrieval = retrieval

  def build_RAG_without_context(self):
    prompt = PromptTemplate(
      input_variables=["question"],
      template = Prompt_Template.NO_CONTEXT,
    )
    return prompt | self.llm | StrOutputParser()
  
  def build_RAG_with_context(self):
    prompt = PromptTemplate(
      input_variables=["question"],
      template = Prompt_Template.EXTENDED_CONTEXT,
    )
    return {"context": self.retrieval, "question": RunnablePassthrough()} | prompt | self.llm | StrOutputParser()

  def build_RAG_with_chat(self, use_memory=False):
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True) if use_memory else None
    prompt = PromptTemplate(
      input_variables=["question"],
      template = Prompt_Template.REDUCED_CONTEXT,
    )
    return ChatVectorDBChain.from_llm(self.llm, self.retrieval, memory=memory, question_prompt=prompt)
    

















