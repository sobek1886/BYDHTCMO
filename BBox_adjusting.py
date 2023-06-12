import numpy as np
from PIL import Image
from super_gradients.common.object_names import Models
from super_gradients.training import models
import torch

class BoundingBox:
    def __init__(self, coordinates):
        self.coordinates = np.array(coordinates, dtype=np.float32)

    def calculate_area(self):
        width = self.coordinates[2] - self.coordinates[0]
        height = self.coordinates[3] - self.coordinates[1]
        return width * height

    def get_coordinates(self):
        return self.coordinates

class RegionDetector:
    def __init__(self, image, make_square):
         self.input_image = image
         self.make_square = make_square
         if type(image) == str:  # input_image is a URL
            self.image = Image.open(image).convert('RGB')
         elif isinstance(image, Image.Image): # input_image is already a PIL Image object
            self.image = image
         else:  # input_image is a numpy ndarray # for gradio
            self.image = Image.fromarray(image)

         self.image_width, self.image_height = self.image.size

    def YOLO_prediction(self):
        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        model = models.get("yolo_nas_l", pretrained_weights="coco").to(device)
        image_prediction = list(model.predict(self.input_image, conf=0.2))[0]
        return image_prediction

    def detect_objects(self, YOLO_predictions):
        detected_objects = []
        image_prediction = YOLO_predictions
        class_names = image_prediction.class_names
        labels = image_prediction.prediction.labels
        confidence = image_prediction.prediction.confidence
        bboxes = image_prediction.prediction.bboxes_xyxy

        for i, (label, conf, bbox) in enumerate(zip(labels, confidence, bboxes)):
          detected_objects.append(BoundingBox(bbox))

        return detected_objects

    def determine_objects_in_region(self, detected_objects, most_important_region):
        objects_in_region = []
        for obj in detected_objects:
            intersection_area = self.calculate_intersection_area(obj, most_important_region)
            object_area = obj.calculate_area()

            if intersection_area >= 0.3 * object_area:
                objects_in_region.append(obj)
        return objects_in_region

    def calculate_intersection_area(self, obj, region):
        intersection = np.maximum(0, np.minimum(obj.coordinates, region.coordinates))
        return BoundingBox(intersection).calculate_area()
      
    def adjust_most_important_region(self, objects_in_region, most_important_region):
      if len(objects_in_region) == 0:
        min_x, min_y, max_x, max_y = most_important_region.coordinates
      else:
        min_x = min(*([obj.coordinates[0] for obj in objects_in_region]))
        min_y = min(*([obj.coordinates[1] for obj in objects_in_region]))
        max_x = max(*([obj.coordinates[2] for obj in objects_in_region]))
        max_y = max(*([obj.coordinates[3] for obj in objects_in_region]))
      
      if self.make_square:
        width = max_x - min_x
        height = max_y - min_y
        print('\r')

        if width == height:
          adjusted_region = np.array([
          max(0, min_x),
          max(0, min_y),
          min(max_x, self.image_width),
          min(max_y, self.image_height)
          ], dtype=np.float32)

        if width > height:
          gap = width - height
          top_available = min_y
          bot_available = (self.image_height - max_y)
          if gap >= (top_available + bot_available):
              adjusted_region = np.array([min_x, 0, max_x, self.image_height], dtype=np.float32)
              '''if new_height != new_width
                  outpaint'''
          else:
              top_available = min_y
              bot_available = (self.image_height - max_y)
              if (top_available < gap/2):
                  print(f"width: {width} height: {height} gap: {gap} top_available: {top_available} bot_availableee: {bot_available}  ")
                  adjusted_region = np.array([min_x, 0, max_x, (self.image_height - (gap - top_available))], dtype=np.float32)
              elif (bot_available < gap/2):
                  adjusted_region = np.array([min_x, (gap - bot_available), max_x, (self.image_height)], dtype=np.float32)
              else:
                  adjusted_region = np.array([min_x, (min_y - gap/2), max_x, (max_y + gap/2)], dtype=np.float32)

        else:
          gap = height - width
          top_available = min_y
          bot_available = (self.image_height - max_y)
          if gap >= (top_available + bot_available):
              adjusted_region = np.array([min_x, 0, max_x, self.image_height], dtype=np.float32)
              '''if new_height != new_width
                  outpaint'''
          else:
              top_available = min_y
              bot_available = (self.image_height - max_y)
              if (top_available < gap/2):
                  print(f"width: {width} height: {height} gap: {gap} top_available: {top_available} bot_availableee: {bot_available}  ")
                  adjusted_region = np.array([min_x, 0, max_x, (self.image_height - (gap - top_available))], dtype=np.float32)
              elif (bot_available < gap/2):
                  adjusted_region = np.array([min_x, (gap - bot_available), max_x, (self.image_height)], dtype=np.float32)
              else:
                  adjusted_region = np.array([min_x, (min_y - gap/2), max_x, (max_y + gap/2)], dtype=np.float32)

      else:
        adjusted_region = np.array([
        max(0, min_x),
        max(0, min_y),
        min(max_x, self.image_width),
        min(max_y, self.image_height)
        ], dtype=np.float32)

      return adjusted_region
        
if __name__ == '__main__':
  # Usage example:
  most_important_region = [29, 43, 656, 811]
  image = "/content/GAIC_280712.jpg"
  make_square = True

  region_detector = RegionDetector(image, make_square)
  YOLO_predictions = region_detector.YOLO_prediction()
  detected_objects = region_detector.detect_objects(YOLO_predictions)
  most_important_region = BoundingBox(most_important_region)
  objects_in_region = region_detector.determine_objects_in_region(detected_objects, most_important_region)
  adjusted_region = region_detector.adjust_most_important_region(objects_in_region, most_important_region)
  print("Adjusted region:", adjusted_region)
