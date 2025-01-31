import torch
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM 
from transformers import BitsAndBytesConfig
from langchain_openai import ChatOpenAI

from variables import MODEL_NAME_GOOGLE

class LoadModel():

  def __new__(cls, api_key=None , quantize=True):
    if api_key is not None:
      model = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=api_key)
      return model, None

    else:
      if quantize:
        bnb_config = cls._quantize_4bit()
        model = AutoModelForSeq2SeqLM.from_pretrained(
          MODEL_NAME_GOOGLE,
          quantization_config=bnb_config
          )
      else:
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME_GOOGLE)
      tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_GOOGLE)
      return model, tokenizer

  def _quantize_4bit():
    return BitsAndBytesConfig(
          load_in_4bit=True,
          bnb_4bit_use_double_quant=True,
          bnb_4bit_quant_type="nf4",
          bnb_4bit_compute_dtype=torch.bfloat16
          )







