from langchain_core.runnables import RunnablePassthrough
from langchain.schema.runnable import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

from variables import Prompt_Template

class RAG_Builder:
  def __init__(self, llm, retrieval):
    self.llm = llm
    self.retrieval = retrieval

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

    class Chat:
      def __init__(self, retrieval, llm):
        
        self.memory = ConversationBufferMemory(
          memory_key="chat_history",
          return_messages=True,
          input_key="question",
          output_key="answer"
        )
        self.prompt = PromptTemplate(
          input_variables=["chat_history","context", "question"],
          template=Prompt_Template.CHAT_CONTEXT.value,
        )
        self.retrieval = retrieval
        self.llm = llm

        self.chain = {
          "chat_history": self.memory.load_memory_variables,
          "context": self.retrieval,
          "question": RunnablePassthrough()
        } | self.prompt | self.llm | StrOutputParser()

      def chat(self, query:str):
        response = self.chain.invoke(query)
        self.memory.save_context({"question": query}, {"answer": response})
        return response

    return Chat(
      retrieval=self.retrieval,
      llm=self.llm
    )


    # self.memory = ConversationBufferMemory(
    #     memory_key="chat_history",
    #     return_messages=True,
    #     input_key="question",
    #     output_key="answer"
    # )

    # prompt = PromptTemplate(
    #   input_variables=["chat_history","context", "question"],
    #   template=Prompt_Template.CHAT_CONTEXT.value,
    # )

    # def save_to_memory(inputs_and_outputs):
    #   query = inputs_and_outputs["question"]
    #   print("Query ->",query)
    #   response = inputs_and_outputs["answer"]
    #   print("Response ->",response)
    #   self.memory.save_context({"query": query}, {"answer": response})
    #   return response

    # # def save_to_memory(inputs, outputs):
    # #   self.memory.save_context({"query": inputs["question"]}, {"answer": outputs})
    # #   return outputs

    # return {
    #   "chat_history": self.memory.load_memory_variables,
    #   "context": self.retrieval,
    #   "question": RunnablePassthrough()
    # } | prompt | self.llm | StrOutputParser() | RunnableLambda(save_to_memory)


  

















