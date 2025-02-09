from transformers import pipeline

from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from postRetrieval.loadModel import LoadModel

from variables import PROMT_TEMPLATE_ITA
from variables import PROMT_TEMPLATE_ENG

# Gestione diversa per la costruzione del prompt?
# Gestiore parametri diversi per LLM
class LLM:
  def __new__(cls, model_name, is_local = True, quantize = False, key_HF = None, key_GPT = None, lang = "en"):
    if not is_local and (key_HF is None and key_GPT is None):
        raise ValueError("To use a remote model, you must provide a valid key.")
    
    if is_local:
        model, tokenizer = LoadModel(model_name, quantize)
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
    else:
        if key_HF is not None:
            llm = HuggingFaceEndpoint(
                task="text-generation",
                repo_id = model_name,
                huggingfacehub_api_token = key_HF,
                temperature=0.2,
                do_sample=True,
                repetition_penalty=1.1,
                return_full_text=True,
                max_new_tokens=300,
            )
        else:
            llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=key_GPT)
      
    prompt = PromptTemplate(
      input_variables=["context", "question"],
      template = PROMT_TEMPLATE_ITA if lang=="ita" else PROMT_TEMPLATE_ENG,
      )   
    
    return prompt | llm | StrOutputParser()
 




