import inspect
from typing import List, Optional, Union

import numpy as np
import torch

import inspect
from typing import List, Optional, Union

import numpy as np
import torch
import PIL
from PIL import Image, ImageFilter
import gradio as gr
from diffusers import StableDiffusionInpaintPipeline
import sys

device = "cuda"
model_path = "runwayml/stable-diffusion-inpainting"

pipe = StableDiffusionInpaintPipeline.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
).to(device)


#add aspect ratio, num_samples as parameters later
def outpaint_image(image):
  generator = torch.Generator(device="cuda").manual_seed(0) # change the seed to get different results
  generation_size = 512
  ## empty prompt for random generation
  prompt = ""
  seed = 51
  guidance_scale= 7.5
  num_samples = 1
  # negative prompt to steer away the random generation from generating frames/fonts/writings
  negative = """grass, frame, album cover, document photo, portrait, picture frame, incoherents, collage, type design, magazine cover, with text, cover, painting, wall mural, poster on wall,    poster, screenshot, awful, wallpaper, grid, collage, text, writing, with writing, painting on the wall, poster, movie poster, logo, logos, watermark, plate, border, edge, wood, table, fabric, plate, pattern"""

  if type(image) == str:  # input_image is a URL
    image = Image.open(image)
  else:  # input_image is a numpy ndarray # for gradio
    image = Image.fromarray(image)

  #get image with added whitespace and mask
  input_images = get_masks(image)
  input_image = input_images[0]
  mask_image = input_images[1]

  images = pipe(
      prompt=prompt,
      image=input_image,
      mask_image= mask_image,
      guidance_scale=guidance_scale,
      generator=generator,
      num_images_per_prompt=num_samples,
      width = generation_size,
      height = generation_size,
      negative_prompt=negative).images
  images.insert(0, input_image)
  outpainted = images[1]
  result = paste_original(image, outpainted)
  result.save('/content/Fork-Human-Centric-Image-Cropping/results_cropping/outpainted.png')
  return result

def get_masks(input_image):
    # Get the original dimensions
    width, height = input_image.size
    # Calculate the size of the new square image
    new_size = max(width, height)
    # Create a new blank square image
    blank = Image.new("RGB", (new_size, new_size), (255, 255, 255))
    # Paste the original image into the center of the new image
    blank.paste(input_image, ((new_size - width) // 2, (new_size - height) // 2))
    square_image = blank.resize((512, 512))

    mask_image = input_image.convert('L')
    mask_image = mask_image.point(lambda x: 0, '1')
    # Create a new blank square image
    blank = Image.new("L", (new_size, new_size), (255))
    # Paste the original image into the center of the new image
    blank.paste(mask_image, ((new_size - width) // 2, (new_size - height) // 2))
    mask_image = blank.resize((512, 512))

    return square_image, mask_image

def resize_image(image):
    # get the original image size
    width, height = image.size

    # determine which side is longer
    if width >= height:
        # resize based on width
        new_width = 512
        new_height = round((512 / width) * height)
    else:
        # resize based on height
        new_height = 512
        new_width = round((512 / height) * width)

    # resize the image while preserving aspect ratio
    resized_image = image.resize((new_width, new_height))

    # return the resized image
    return resized_image

def paste_original(original_image, outpainted):
  # resize original
  original_resized = resize_image(original_image)
  # Get the dimensions
  width, height = original_resized.size
  new_size = max(width, height)
  
  #paste the resized original image on top of the outpaint
  outpainted.paste(original_resized, ((new_size - width) // 2, (new_size - height) // 2))
  return outpainted

if __name__ == '__main__':
  argument_list = sys.argv
  image = f"'{argument_list[1]}'"
  outpaint_image(image)
  #outpaint_image('/content/Fork-Human-Centric-Image-Cropping/results_cropping/original.png')

