import os
import re
import base64
from io import BytesIO
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document as LangchainDocument

from docling_core.types.doc.base import ImageRefMode
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat


from variables import CAPTIONING_MODEL

class ImageCaptioningFromBase64:
  def __init__(self) -> None:
      self.processor = BlipProcessor.from_pretrained(CAPTIONING_MODEL)
      self.model = BlipForConditionalGeneration.from_pretrained(CAPTIONING_MODEL)

  def _base64_to_image(self, base64_string):
    image_data = base64.b64decode(base64_string.split(",")[1])
    return Image.open(BytesIO(image_data))

  def _image_to_caption(self, image):
    inputs = self.processor(image, return_tensors="pt")
    output = self.model.generate(**inputs)
    caption = self.processor.decode(output[0], skip_special_tokens=True)
    return caption

  def replace_base64_images_with_captions(self, markdown_text, pattern_base64):
    
    def replace_match(match):
        base64_string = match.group(1)
        image = self._base64_to_image(base64_string)
        caption = self._image_to_caption(image)
        return f"<|image|>{caption}</s>"

    updated_text = re.sub(pattern_base64, replace_match, markdown_text)
    return updated_text



class DoclingPDFLoader(BaseLoader):
  def __init__(self, path: str | list[str], do_captioning = False):
    self.do_captioning = do_captioning
    self.file_paths = path if isinstance(path, list) else [path]
    self.converter = DocumentConverter(
      format_options={
        InputFormat.PDF: PdfFormatOption(
          pipeline_options=PdfPipelineOptions(
            generate_picture_images = do_captioning,
          )
        )
      }
    )

  def load(self): # -> Iterator[LangchainDocument]
    docs={}
    for source in self.file_paths:
      file_name = os.path.splitext(os.path.basename(source))[0]
      docling_doc = self.converter.convert(source).document

      text = docling_doc.export_to_markdown(
        image_mode = ImageRefMode.EMBEDDED if self.do_captioning else ImageRefMode.PLACEHOLDER
      )
      if self.do_captioning:
        captioning = ImageCaptioningFromBase64()
        pattern_image = r'!\[Image\]\((data:image/png;base64,[^\)]+)\)'
        text = captioning.replace_base64_images_with_captions(text, pattern_image)
      docs[file_name] = text
      # docs.append(LangchainDocument(page_content=updated_text, metadata={"name":file_name}))
      # yield LangchainDocument(page_content=updated_text)
    return docs





