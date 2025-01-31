from transformers import pipeline

from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFacePipeline
from langchain_core.output_parsers import StrOutputParser

from variables import PROMT_TEMPLATE_ITA
from variables import PROMT_TEMPLATE_ENG

# Gestione diversa per la costruzione del prompt?
class LLM:
  def __new__(cls, model, tokenizer=None):
    if tokenizer is None:
      llm = model
    else:
      text_generation_pipeline = pipeline(
        model=model,
        tokenizer=tokenizer,
        task="text-generation",
        temperature=0.2,
        do_sample=True,
        repetition_penalty=1.1,
        return_full_text=True,
        max_new_tokens=300,
        pad_token_id=tokenizer.eos_token_id
        )
      llm = HuggingFacePipeline(pipeline=text_generation_pipeline)
      
    prompt = PromptTemplate(
      input_variables=["context", "question"],
      template = PROMT_TEMPLATE_ENG,
      )   
    
    return prompt | llm | StrOutputParser()
 




