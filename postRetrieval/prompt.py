from variables import PROMT_TEMPLATE_SHORT
from variables import PROMT_TEMPLATE_GPT

# Funzione per la creazione del prompt che si adatta al modello di Sapienza?
def prompt_model(tokenizer):
  messages = [
      {"role": "system", "content": PROMT_TEMPLATE_SHORT},
      {"role": "user", "content": "{question}"},
  ]
  tokenized_prompt = tokenizer.apply_chat_template(
      messages, tokenize=False, add_generation_prompt=False
  )
  return "".join(tokenized_prompt)
