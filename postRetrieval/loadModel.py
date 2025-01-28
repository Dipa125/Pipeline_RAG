import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from variables import MODEL_NAME_BF16
from variables import PROMT_TEMPLATE_SHORT

class LoadModel():

  def __new__(cls, quantize=True):
    
    # Funzione per la creazione del prompt che si adatta al modello di Sapienza?
    def _prompt_model(tokenizer):
      messages = [
          {"role": "system", "content": PROMT_TEMPLATE_SHORT},
          {"role": "user", "content": "{question}"},
      ]
      tokenized_prompt = tokenizer.apply_chat_template(
          messages, tokenize=False, add_generation_prompt=False
      )
      return "".join(tokenized_prompt)


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
    prompt = _prompt_model(tokenizer)

    return model, tokenizer, prompt








