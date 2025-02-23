from transformers import pipeline

from langchain_huggingface import HuggingFacePipeline, HuggingFaceEndpoint
from langchain_openai import ChatOpenAI

from postRetrieval.loadModel import LoadModel

# Gestiore parametri diversi per LLM
class LLM:
  def __new__(cls, model_name = None, is_local = True, quantize = False, key_HF = None, key_GPT = None, return_prompt=False):
    if not is_local and (key_HF is None and key_GPT is None):
      raise ValueError("To use a remote model, you must provide a valid key.")
    
    if key_GPT is None and model_name is None:
      raise ValueError("A model_name is required when using a HuggingFace model")

    if is_local:
      model, tokenizer = LoadModel(model_name, quantize)
      text_generation_pipeline = pipeline(
        model=model,
        tokenizer=tokenizer,
        task="text-generation",
        return_full_text=return_prompt,
        temperature=0.2,
        do_sample=True,
        repetition_penalty=1.1,
        max_new_tokens=300,
        pad_token_id=tokenizer.eos_token_id
      )
      llm = HuggingFacePipeline(pipeline=text_generation_pipeline)
    else:
      if key_HF is not None:
        llm = HuggingFaceEndpoint(
          task="text-generation",
          return_full_text=return_prompt,
          repo_id = model_name,
          huggingfacehub_api_token = key_HF,
          temperature=0.2,
          do_sample=True,
          repetition_penalty=1.1,
          max_new_tokens=300,
        )
      else:
        llm = ChatOpenAI(
          model="gpt-3.5-turbo",
          openai_api_key = key_GPT,
          verbose=return_prompt)
    
    return llm
 




