from transformers import pipeline

from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

import sys
sys.path.append('/content/Pipeline_RAG')
from variables import PROMT_TEMPLATE

class LLM:
  def __new__(cls, model, tokenizer):
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=PROMT_TEMPLATE,
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





