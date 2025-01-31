from transformers import pipeline

from langchain_huggingface import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from variables import PROMT_TEMPLATE

# Gestione diversa per la costruzione del prompt?
class LLM:
  def __new__(cls, model, tokenizer=None):
    if tokenizer is None:
      # Stai usando GPT, non serve tokenizer e prompt
      print("")
    else:
      # Stai usando altri modelli di HuggingFace, usa tokenizer e prompt
      print("")
    
    # Creazione del prompt da adattare al modello di sapienza
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=prompt,
    )
    """
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
    """
    llm = model
    return prompt | llm | StrOutputParser()   





