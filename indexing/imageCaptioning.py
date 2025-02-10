import re
import base64
from io import BytesIO
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

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

    