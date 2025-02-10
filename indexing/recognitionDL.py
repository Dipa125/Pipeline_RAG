import os
from enum import Enum
from typing import List

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document as LangchainDocument

from docling_core.types.doc.base import ImageRefMode
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat

from indexing.imageCaptioning import ImageCaptioningFromBase64

class ExportFormat(Enum):
    JSON = "json"
    MARKDOWN = "markdown"

# Gestione delle variabili di istanza?
class DoclingPDFLoader(BaseLoader):
  def __init__(self, path: str | list[str], do_captioning = False):
    self.do_captioning = do_captioning
    self.file_paths = path if isinstance(path, list) else [path]
    self.docs_markdown={}
    self.docs_json={}
    self.converter = DocumentConverter(
      format_options={
        InputFormat.PDF: PdfFormatOption(
          pipeline_options=PdfPipelineOptions(
            generate_picture_images = do_captioning,
          )
        )
      }
    )

  def load(self, export_format: ExportFormat):
    for source in self.file_paths:
      file_name = os.path.splitext(os.path.basename(source))[0]
      docling_doc = self.converter.convert(source).document

      if export_format == ExportFormat.MARKDOWN:
        text = docling_doc.export_to_markdown(
          image_mode = ImageRefMode.EMBEDDED if self.do_captioning else ImageRefMode.PLACEHOLDER
        )
        if self.do_captioning:
          text = self._captioningFromBase64(text)
        self.docs_markdown[file_name] = text
      
      elif export_format == ExportFormat.JSON: #Da capire ancora le immagini
        json_data = docling_doc.export_to_dict()
        self.docs_json[file_name] = json_data
      else:
        raise ValueError(f"Unsupported export format: {export_format}. Use ExportFormat.JSON or ExportFormat.MARKDOWN")
    # return self.docs # Potrebbe non serive più essendo diventata una variabile di istanza
  
  def langchainDocument_from_markdown(self)->List[LangchainDocument]:
    langchain_docs_markdown = [
      LangchainDocument(page_content=self.docs_markdown[doc], metadata={"name":doc}) for doc in self.docs_markdown
    ]
    return langchain_docs_markdown

  def langChainDocument_from_json(self) -> List[LangchainDocument]:
    print("La funzione non è ancora implementata.")
    return []

  def _captioningFromBase64(self, text):
    captioning = ImageCaptioningFromBase64()
    pattern_image = r'!\[Image\]\((data:image/png;base64,[^\)]+)\)'
    return captioning.replace_base64_images_with_captions(text, pattern_image)






