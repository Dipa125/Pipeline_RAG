'''
Questa classe si occupa del recupero dei file da una cartella
e di convertirli in testo.
'''

import os
import fitz
from paddleocr import PaddleOCR
from PIL import Image
from io import BytesIO

class Recognition:

  def __init__(self, lang):
    self.docs = {}
    self.paddleOCR = PaddleOCR(use_angle_cls=True, lang=lang)

#-Concatena il testo rilevato in ogni pagina del PDF
  def __pdf2text(self, reader):
    return " ".join([page.get_text("text") for page in reader.pages()])

#-Trasforma immagini in testo con PaddleOCR
  def __images2text(self, images):
    text = ""
    for image in images:
      paddle_result = self.paddleOCR.ocr(image, cls=True)
      for line in paddle_result[0]:
          text += line[1][0] + " "
      text += "\n"
    return text

  def extractText(self, folder_path):
    for file_name in os.listdir(folder_path):

      file_path = os.path.join(folder_path, file_name)
      reader = fitz.open(file_path)
      images=[]

      if file_path.endswith(".pdf"):
        text = self.__pdf2text(reader)
        if(text != ""):
          self.docs[file_name]=text
        else:
          images = []
          for page_num in range(len(reader)):
              page = reader.load_page(page_num)
              pix = page.get_pixmap(dpi=150)
              img = Image.open(BytesIO(pix.tobytes("png")))
              img = img.convert("RGB")
              images.append(img)
          self.docs[file_name] = self.__images2text(images)

      elif file_path.endswith((".png", ".jpg", ".jpeg", ".bmp")):
        img = Image.open(file_path).convert("RGB")
        self.docs[file_name] = self.__images2text([img])

    return self.docs


























