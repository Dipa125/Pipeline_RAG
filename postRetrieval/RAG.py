from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory

from variables import Prompt_Template

# Rivedere i prompt
class RAG_Builder:
  def __init__(self, llm, retrieval):
    self.llm = llm
    self.retrieval = retrieval
    print("Versione 5")

  #---NO CONTEXT---
  def build_RAG_without_context(self):
    prompt = PromptTemplate(
      input_variables=["question"],
      template = Prompt_Template.NO_CONTEXT.value,
    )
    return prompt | self.llm | StrOutputParser()

  #---CONTEXT--- 
  def build_RAG_with_context(self):
    prompt = PromptTemplate(
      input_variables=["context", "question"],
      template = Prompt_Template.DOCUMENT_CONTEXT.value,
    )
    return {
      "context": self.retrieval,
      "question": RunnablePassthrough()
    } | prompt | self.llm | StrOutputParser()

  #---MEMORY---
  def build_RAG_with_chat(self):
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        input_key="query",
        output_key="answer"
    )

    prompt = PromptTemplate(
      input_variables=["chat_history","context", "question"],
      template=Prompt_Template.CHAT_CONTEXT.value,
    )

    print_promt = {
      "chat_history": memory.load_memory_variables,
      "context": self.retrieval,
      "question": RunnablePassthrough()
    } | prompt

    print(print_promt)

    return {
      "chat_history": memory.load_memory_variables,
      "context": self.retrieval,
      "question": RunnablePassthrough()
    } | prompt | self.llm | StrOutputParser()


  

















