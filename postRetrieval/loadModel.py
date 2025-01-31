import torch
from transformers import AutoTokenizer, AutoModelForCausalLM 
from transformers import BitsAndBytesConfig
from langchain_openai import ChatOpenAI

from variables import MODEL_NAME_FACEBOOK

class LoadModel():

  def __new__(cls, api_key=None, quantize=True):
    if api_key is not None:
      model = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=api_key)
      return model, None

    else:
      if quantize:
        bnb_config = cls._quantize_4bit()
        model = AutoModelForCausalLM.from_pretrained(
          MODEL_NAME_FACEBOOK,
          quantization_config=bnb_config
          )
      else:
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME_FACEBOOK)
        
      tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_FACEBOOK)
      return model, tokenizer

  def _quantize_4bit():
    return BitsAndBytesConfig(
          load_in_4bit=True,
          bnb_4bit_use_double_quant=True,
          bnb_4bit_quant_type="nf4",
          bnb_4bit_compute_dtype=torch.bfloat16
          )







