from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain.chains import ChatVectorDBChain
from langchain.memory import ConversationBufferMemory

from variables import Prompt_Template

# Rivedere i prompt
class RAG_Builder:
  def __init__(self, llm, vectorDB):
    self.llm = llm
    self.vectorDB = vectorDB

  def build_RAG_without_context(self):
    prompt = PromptTemplate(
      input_variables=["question"],
      template = Prompt_Template.NO_CONTEXT.value,
    )
    return prompt | self.llm | StrOutputParser()
  
  def build_RAG_with_context(self):
    prompt = PromptTemplate(
      input_variables=["question"],
      template = Prompt_Template.EXTENDED_CONTEXT.value,
    )
    return {"context": self.vectorDB.retrieval, "question": RunnablePassthrough()} | prompt | self.llm | StrOutputParser()

  def build_RAG_with_chat(self, use_memory=False):
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True) if use_memory else None
    prompt = PromptTemplate(
      input_variables=["question"],
      template = Prompt_Template.REDUCED_CONTEXT.value,
    )
    return ChatVectorDBChain.from_llm(self.llm, self.vectorDB, memory=memory, condense_question_prompt=prompt)
    

















