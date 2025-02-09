import torch
from transformers import AutoTokenizer, AutoModelForCausalLM 
from transformers import BitsAndBytesConfig

# Parametrizzare la quantizzazione?
class LoadModel():

  def __new__(cls, model_name, quantize):
    if quantize:
      bnb_config = cls._quantize_4bit()
      model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config
      )
    else:
      model = AutoModelForCausalLM.from_pretrained(model_name)
      
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

  def _quantize_4bit():
    return BitsAndBytesConfig(
      load_in_4bit=True,
      bnb_4bit_use_double_quant=True,
      bnb_4bit_quant_type="nf4",
      bnb_4bit_compute_dtype=torch.bfloat16
    )







