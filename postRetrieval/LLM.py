import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from transformers import pipeline
from langchain_core.output_parsers import StrOutputParser

import sys
sys.path.append('/content/Pipeline_RAG')

from variables import MODEL_NAME_BF16
from variables import PROMT_TEMPLATE

class LLM:
  def __init__(self):
    bnb_config = BitsAndBytesConfig(
      load_in_4bit=True,
      bnb_4bit_use_double_quant=True,
      bnb_4bit_quant_type="nf4",
      bnb_4bit_compute_dtype=torch.bfloat16
      )
    self.model = AutoModelForCausalLM.from_pretrained(MODEL_NAME_BF16, quantization_config=bnb_config)
    self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_BF16)
  
  def __prompt():
    return PromptTemplate(
      input_variables=["context", "question"],
      template=PROMT_TEMPLATE,
      )

  def create_pipelineLLM(self):
    text_generation_pipeline = pipeline(
      model=self.model,
      tokenizer=self.tokenizer,
      task="text-generation",
      temperature=0.2,
      do_sample=True,
      repetition_penalty=1.1,
      return_full_text=True,
      max_new_tokens=400,
      )
    llm = HuggingFacePipeline(pipeline=text_generation_pipeline)
    return self.__promt() | llm | StrOutputParser()





