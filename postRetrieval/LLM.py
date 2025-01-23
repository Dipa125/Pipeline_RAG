import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

import sys
sys.path.append('/content/Pipeline_RAG')

from variables import MODEL_NAME

class LLM:
    