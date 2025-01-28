from transformers import pipeline

from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from variables import PROMT_TEMPLATE

# Gestione diversa per la costruzione del prompt?
class LLM:
  def __new__(cls, model, tokenizer, prompt):
    # Creazione del prompt da adattare al modello di sapienza
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=prompt,
    )
    
    text_generation_pipeline = pipeline(
        model=model,
        tokenizer=tokenizer,
        task="text-generation",
        temperature=0.2,
        do_sample=True,
        repetition_penalty=1.1,
        return_full_text=True,
        max_new_tokens=400,
    )
    
    llm = HuggingFacePipeline(pipeline=text_generation_pipeline)
    
    return prompt | llm | StrOutputParser()   





