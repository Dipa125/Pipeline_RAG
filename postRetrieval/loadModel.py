import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from variables import MODEL_NAME_BF16

class LoadModel():
  def __new__(cls, quantize=True):
    if quantize:
      bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
        )
      model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME_BF16,
        quantization_config=bnb_config
        )
    else:
      model = AutoModelForCausalLM.from_pretrained(MODEL_NAME_BF16)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_BF16)
    return model, tokenizer