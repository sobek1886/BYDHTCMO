import inspect
from typing import List, Optional, Union

import numpy as np
import torch
import PIL
from PIL import Image, ImageFilter
import gradio as gr
from diffusers import StableDiffusionInpaintPipeline
from SSIM_PIL import compare_ssim
from config_GAICD import cfg

class OutpaintingFeature:
    def __init__(self, image, num_samples = 1):

        self.desired_size = cfg.image_size
        self.input_image = image
        if type(self.input_image) == str:  # input_image is a URL
            self.input_image = Image.open(self.input_image)
        elif isinstance(self.input_image, Image.Image): # input_image is already a PIL Image object
            pass
        else:  # input_image is a numpy ndarray # for gradio
            self.input_image = Image.fromarray(self.input_image)
        # resize image
        #self.input_image.thumbnail((512,512))
        print(f'input to outpaint.py {self.input_image.size}')
        self.input_image.thumbnail(self.desired_size)
        print(f'after thumbnail in outpaint init {self.input_image.size}')

        # get width and size
        self.image_width, self.image_height = self.input_image.size
        # Calculate the size of the new square image
        self.new_size = max(self.image_width, self.image_height)

        # coordinates used to paste original image back
        self.x = (self.new_size - self.image_width) // 2
        self.y = (self.new_size - self.image_height) // 2

        # used to get areas for SSIM
        if self.image_width > self.image_height:
          self.ratio = self.desired_size[0]/self.image_width
          self.wider = True
        else:
          self.ratio = self.desired_size[0]/self.image_height
          self.wider = False

        self.device = "cuda"
        self.model_path = "runwayml/stable-diffusion-inpainting"
        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
        ).to(self.device)
        self.generator = torch.Generator(device="cuda").manual_seed(0) # change the seed to get different results
        self.generation_size = self.desired_size[0]
        ## empty prompt for random generation
        self.prompt = ""
        self.seed = 51
        self.guidance_scale= 7.5
        self.num_samples = num_samples
        # negative prompt to steer away the random generation from generating frames/fonts/writings
        self.negative = '''frame, album cover, document photo, portrait, picture frame, incoherents, collage, type design, magazine cover, with text, cover, painting, wall mural, poster on wall, poster, screenshot, awful, wallpaper, grid, collage, text, writing, with writing, painting on the wall, poster, movie poster, logo, logos, watermark, plate, border, edge, wood, table, fabric, plate, pattern, lowres, error, cropped, worst quality, low quality, jpeg artifacts, out of frame, watermark, signature, deformed, ugly, mutilated, disfigured, text, extra limbs, face cut, head cut, extra fingers, extra arms, poorly drawn face, mutation, bad proportions, cropped head, malformed limbs, mutated hands, fused fingers, long neck'''
        """frame, album cover, document photo, portrait, picture frame, incoherents, collage, type design, magazine cover, with text, cover, painting, wall mural, poster on wall, poster, screenshot, awful, wallpaper, grid, collage, text, writing, with writing, painting on the wall, poster, movie poster, logo, logos, watermark, plate, border, edge, wood, table, fabric, plate, pattern"""

    def outpaint_image(self):

        #get image with added whitespace and mask
        squared_image, mask_image = self.get_masks()
        print(f'whitespaced: {squared_image.size} \n mask{mask_image.size}')
        # generate outpaints
        images = self.pipe(
            prompt = self.prompt,
            image = squared_image,
            mask_image = mask_image,
            guidance_scale = self.guidance_scale,
            generator = self.generator,
            num_images_per_prompt = self.num_samples,
            width = self.generation_size,
            height = self.generation_size,
            negative_prompt = self.negative).images
        
        # get areas for SSIM comparison
        new_areas = [self.get_areas_new(outpainted_image) for outpainted_image in images]
        og_A, og_B = self.get_areas_original()
        best_A, idx_best_B = self.get_best_outpaints(og_A, og_B, new_areas)

        #combine the original with the best outpaints
        result = images[idx_best_B]
        result.paste(best_A, (0, 0))
        result.paste(self.input_image, (self.x, self.y))
        #result.save('/content/Fork-Human-Centric-Image-Cropping/results_cropping/outpainted.png')
        return result

    def get_masks(self):
        # Create a new blank square image
        blank = Image.new("RGB", (self.new_size, self.new_size), (255, 255, 255))
        # Paste the original image into the center of the new image
        blank.paste(self.input_image, (self.x, self.y))
        squared_image = blank

        mask_image = self.input_image.copy().convert('L')
        mask_image = mask_image.point(lambda x: 0, '1')
        # Create a new blank square image
        blank = Image.new("L", (self.new_size, self.new_size), (255))
        # Paste the mask
        blank.paste(mask_image, (self.x, self.y))
        mask_image = blank

        return squared_image, mask_image
    
    def get_areas_new(self, outpainted_image):

        outpainted_image_width, outpainted_image_height = outpainted_image.size
        if self.wider:

            crop_coordinates_new_A = [coord * self.ratio for coord in [0, 0, outpainted_image_width, self.y]]
            new_area_A = outpainted_image.crop(crop_coordinates_new_A)

            crop_coordinates_new_B = [coord * self.ratio for coord in [0, outpainted_image_height - self.y, outpainted_image_width, outpainted_image_height]]
            new_area_B = outpainted_image.crop(crop_coordinates_new_B)  


        else:
            crop_coordinates_new_A = [coord * self.ratio for coord in [0, 0, self.x, self.image_height]]
            new_area_A = outpainted_image.crop(crop_coordinates_new_A)

            crop_coordinates_new_B = [coord * self.ratio for coord in [outpainted_image_width - self.x, 0, outpainted_image_width, outpainted_image_height]]
            new_area_B = outpainted_image.crop(crop_coordinates_new_B)  

        return [new_area_A, new_area_B]

    def get_areas_original(self):
      
      if self.wider:
        crop_coordinates_A = [coord * self.ratio for coord in [0, 0, self.image_width, self.y]]
        original_area_A = self.input_image.crop(crop_coordinates_A)

        crop_coordinates_B = [coord * self.ratio for coord in [0, self.image_height - self.y, self.image_width, self.image_height]]
        original_area_B = self.input_image.crop(crop_coordinates_B)

      else:
          crop_coordinates_A = [coord * self.ratio for coord in [0, 0, self.x, self.image_height]]
          original_area_A = self.input_image.crop(crop_coordinates_A)

          crop_coordinates_B = [coord * self.ratio for coord in [self.image_width - self.x, 0, self.image_width, self.image_height]]
          original_area_B = self.input_image.crop(crop_coordinates_B)

      return original_area_A, original_area_B
    
    def get_best_outpaints(self, og_A, og_B, new_areas):
      SSIM_values_A = []
      SSIM_values_B = []

      for area in new_areas:
        print(f'\n og_A: {og_A.size}\n area0: {area[0].size}')
        value_A = compare_ssim(og_A, area[0], GPU=False)
        SSIM_values_A.append(value_A)

        value_B = compare_ssim(og_B, area[1], GPU=False)
        SSIM_values_B.append(value_B)

      idx_A = np.argmax(SSIM_values_A)
      idx_B = np.argmax(SSIM_values_B)

      return new_areas[idx_A][0], idx_B

if __name__ == '__main__':
    img_url = "/content/Fork-Human-Centric-Image-Cropping/GAIC_280712.jpg"
    Outpainter = OutpaintingFeature(img_url, num_samples = 3)
    output = Outpainter.outpaint_image()
    output.show()

