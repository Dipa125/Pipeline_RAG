import os
import fitz
from paddleocr import PaddleOCR
from PIL import Image
from io import BytesIO
import numpy as np

class Recognition:

  def __init__(self):
    self.paddleOCR = PaddleOCR(use_angle_cls=True, lang="it")

#-Concatena il testo rilevato in ogni pagina del PDF
  def __pdf2text(self, reader):
    pages_with_text = [page.get_text("text") for page in reader.pages() if page.get_text("text").strip()]
    return " ".join(pages_with_text)

#-Trasforma immagini in testo con PaddleOCR
  def __images2text(self, images):
    text = ""
    for image in images:
      paddle_result = self.paddleOCR.ocr(np.array(image), cls=True)
      for line in paddle_result[0]:
          text += line[1][0] + " "
      text += "\n"
    return text

#-Gestisce l'accesso ai file di una cartella ed avvia la giusta conversione
  def extractText(self, folder_path):
    docs={}
    for file_name in os.listdir(folder_path):

      file_path = os.path.join(folder_path, file_name)
      reader = fitz.open(file_path)
      images=[]

      if file_path.endswith(".pdf"):
        text = self.__pdf2text(reader)
        if text.strip():
          docs[file_name]=text
        else:
          images = []
          for page_num in range(len(reader)):
              page = reader.load_page(page_num)
              pix = page.get_pixmap(dpi=150)
              img = Image.open(BytesIO(pix.tobytes("png"))).convert("RGB")
              images.append(img)
          docs[file_name] = self.__images2text(images)

      elif file_path.endswith((".png", ".jpg", ".jpeg", ".bmp")):
        img = Image.open(file_path).convert('RGB')
        docs[file_name] = self.__images2text([img])

    return docs







