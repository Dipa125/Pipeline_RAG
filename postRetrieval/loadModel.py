import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

import sys
sys.path.append('/content/Pipeline_RAG')
from variables import MODEL_NAME_BF16

class LoadModel():
  def __new__(cls):
    bnb_config = BitsAndBytesConfig(
      load_in_4bit=True,
      bnb_4bit_use_double_quant=True,
      bnb_4bit_quant_type="nf4",
      bnb_4bit_compute_dtype=torch.bfloat16
      )
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME_BF16, quantization_config=bnb_config)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_BF16)
    return model, tokenizer