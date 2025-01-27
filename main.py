import sys
import subprocess
import argparse

from variables import PROGECT_PATH, REQUIREMENTS_PATH

def setup_environment():
  subprocess.run(["pip", "install", "-q", "-r", REQUIREMENTS_PATH], check=True)
  if PROGECT_PATH not in sys.path:
      sys.path.append(PROGECT_PATH)

def import_class():
  global Recognition, Tokenizer, VectorDB_Manager, LoadModel, LLM, RAG
  from indexing.recognition import Recognition
  from indexing.tokenizer import Tokenizer
  from retrieval.vectorDB_Manager import VectorDB_Manager
  from postRetrieval.loadModel import LoadModel
  from postRetrieval.LLM import LLM
  from postRetrieval.RAG import RAG

# Funzione per l'avvio direttamente da Colab
def create_chains(pathDoc=None):
  if pathDoc is None:
    raise ValueError("Il parametro 'pathDoc' è obbligatorio")

  setup_environment()
  import_class()

  # Istanziamento dell'OCR e del Tokenizzatore
  recognition = Recognition()
  tokenizer = Tokenizer()

  # Processamento dei file all'interno di una cartella
  docs = recognition.extractText(args.pathDoc)
  docs_processed = tokenizer.split_documents(docs)

  # Istanziamento della classe che gestisce il VectorDB
  vectorDB = VectorDB_Manager(docs_processed)

  # Creazione del retrieval associandolo al Vector DB
  retrieval = vectorDB.retrieval()

  # Import del modello e tokenizzatore per LLM
  model, tokenizer = LoadModel(quantize=False)

  # Creazione della catena per LLM
  llm_chain = LLM(model, tokenizer)

  # Creazione della catena completa per la Pipeline RAG
  rag_chain = RAG(retrieval, llm_chain)

  return llm_chain, rag_chain

# Funzione main per l'avvio (per una gestione interattiva)
def main():
  parser = argparse.ArgumentParser(description="--pathDoc per inserire il path contenente i documenti")
  parser.add_argument("--pathDoc", type=str, required=True, help="path della cartella con i documenti")
  args = parser.parse_args()

  setup_environment()
  import_class()

  # Istanziamento dell'OCR e del Tokenizzatore
  recognition = Recognition()
  tokenizer = Tokenizer()

  # Processamento dei file all'interno di una cartella
  docs = recognition.extractText(args.pathDoc)
  docs_processed = tokenizer.split_documents(docs)

  # Istanziamento della classe che gestisce il VectorDB
  vectorDB = VectorDB_Manager(docs_processed)

  # Creazione del retrieval associandolo al Vector DB
  retrieval = vectorDB.retrieval()

  # Import del modello e tokenizzatore per LLM
  model, tokenizer = LoadModel(quantize=False)

  # Creazione della catena per LLM
  llm_chain = LLM(model, tokenizer)

  # Creazione della catena completa per la Pipeline RAG
  rag_chain = RAG(retrieval, llm_chain)

  return llm_chain, rag_chain

if __name__ == "__main__":
  main()
