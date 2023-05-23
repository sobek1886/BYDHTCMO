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
    def __init__(self, image):
         self.input_image = image
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

    def expand_most_important_region(self, objects_in_region, most_important_region):
        min_x = min(most_important_region.coordinates[0], *([obj.coordinates[0] for obj in objects_in_region]))
        min_y = min(most_important_region.coordinates[1], *([obj.coordinates[1] for obj in objects_in_region]))
        max_x = max(most_important_region.coordinates[2], *([obj.coordinates[2] for obj in objects_in_region]))
        max_y = max(most_important_region.coordinates[3], *([obj.coordinates[3] for obj in objects_in_region]))

        expanded_region = np.array([
            max(0, min_x),
            max(0, min_y),
            min(max_x, self.image_width),
            min(max_y, self.image_height)
        ], dtype=np.float32)
        
        return expanded_region

        
if __name__ == '__main__':
  # Usage example:
  most_important_region = [29, 43, 656, 811]
  image = "/content/Fork-Human-Centric-Image-Cropping/GAIC_280712.jpg"

  region_detector = RegionDetector(image)
  YOLO_predictions = region_detector.YOLO_prediction()
  detected_objects = region_detector.detect_objects(YOLO_predictions)
  most_important_region = BoundingBox(most_important_region)
  objects_in_region = region_detector.determine_objects_in_region(detected_objects, most_important_region)
  expanded_region = region_detector.expand_most_important_region(objects_in_region, most_important_region)
  print(expanded_region)
