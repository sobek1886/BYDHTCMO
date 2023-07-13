import os
import cv2
import numpy as np
import pickle
import lmdb
import datetime
import torch
from PIL import Image, ImageOps, ImageDraw
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import json
import matplotlib.pyplot as plt
import math
import random
from config_GAICD import cfg
from cropping_dataset import generate_partition_mask, generate_target_size_crop_mask, rescale_bbox
from test_pretrained import get_pdefined_anchor
from cropping_model import HumanCentricCroppingModel
from outpaint import outpaint_image
import sys

device = torch.device('cuda:{}'.format(cfg.gpu_id))
torch.cuda.set_device(cfg.gpu_id)
MOS_MEAN = 2.95
MOS_STD  = 0.8
IMAGE_NET_MEAN = [0.485, 0.456, 0.406]
IMAGE_NET_STD = [0.229, 0.224, 0.225]

def crop_image(image):
    keep_aspect = False
    crop_mask_downsample = 4
    human_mask_downsample = 16
    image_transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGE_NET_MEAN, std=IMAGE_NET_STD)])

    model = HumanCentricCroppingModel(loadweights=False, cfg=cfg)
    model.load_state_dict(torch.load('/content/Fork-Human-Centric-Image-Cropping/experiments/30epochs/checkpoints/best-srcc.pth'))
    model = model.eval().to(device)
    
    #image_name = self.image_list[index]
    #image_file = os.path.join(self.image_dir, image_name)
    #image_file = image_path
    #assert os.path.exists(image_file), image_file
    if type(image) == str:  # input_image is a URL
      image = Image.open(image).convert('RGB')
    elif isinstance(image, Image.Image): # input_image is already a PIL Image object
      pass
    else:  # input_image is a numpy ndarray # for gradio
      image = Image.fromarray(image)
    image.save('/content/Fork-Human-Centric-Image-Cropping/results_cropping/original.png')

    im_width, im_height = image.size
    if keep_aspect:
        scale = float(cfg.image_size[0]) / min(im_height, im_width)
        h = round(im_height * scale / 32.0) * 32
        w = round(im_width * scale / 32.0) * 32
    else:
        h = cfg.image_size[1]
        w = cfg.image_size[0]
    resized_image = image.resize((w, h), Image.ANTIALIAS)
    im = image_transformer(resized_image)
    #print(im)
    rs_width, rs_height = resized_image.size
    ratio_h = float(rs_height) / im_height
    ratio_w = float(rs_width) / im_width
    '''if image_name in self.human_bboxes:
        hbox = self.human_bboxes[image_name]
        hbox = rescale_bbox(hbox, ratio_w, ratio_h)
    else:'''
    hbox = np.array([[-1, -1, -1, -1]]).astype(np.float32)
    #hbox = np.array([[1, 28, 486, 549]]).astype(np.float32)
    #hbox = rescale_bbox(hbox, ratio_w, ratio_h)
    

    part_mask = generate_partition_mask(hbox, rs_width, rs_height,
                                                 human_mask_downsample)

    #crop = self.annotations[image_name]
    #x,y,w,h = crop
    #crop = torch.tensor([x,y,x+w,y+h])

    pdefined_anchors = get_pdefined_anchor() # n,4, (x1,y1,x2,y2)
    crop = np.zeros((len(pdefined_anchors), 4), dtype=np.float32)
    crop[:, 0::2] = pdefined_anchors[:, 0::2] * im.shape[-1]
    crop[:, 1::2] = pdefined_anchors[:, 1::2] * im.shape[-2]
    crop_mask = generate_target_size_crop_mask(crop, im.shape[-1], im.shape[-2], 64, 64)

    crop = torch.from_numpy(crop).unsqueeze(0).to(device)  # 1,n,4
    crop_mask = torch.from_numpy(crop_mask).unsqueeze(0).to(device)
    im_deviced = im.unsqueeze(0).to(device)
    #print(im_deviced.size())
    hbox_deviced = torch.from_numpy(hbox).unsqueeze(0).to(device)
    part_mask_deviced = torch.from_numpy(part_mask).unsqueeze(0).to(device)

    ############## get prediction from the model
    part_feat, heat_map, scores = model(im_deviced, crop, hbox_deviced, crop_mask, part_mask_deviced)

    # get best crop
    scores = scores.reshape(-1).cpu().detach().numpy()
    idx = np.argmax(scores)
    pred_x1 = int(pdefined_anchors[idx][0] * im_width)
    pred_y1 = int(pdefined_anchors[idx][1] * im_height)
    pred_x2 = int(pdefined_anchors[idx][2] * im_width)
    pred_y2 = int(pdefined_anchors[idx][3] * im_height)

    image_copy = image.copy()
    # Create a draw object
    #draw = ImageDraw.Draw(image_copy)
    # Draw a rectangle on the image copy
    #draw.rectangle((pred_x1, pred_y1, pred_x2, pred_y2), outline='red')
    #image_copy.save('/content/Fork-Human-Centric-Image-Cropping/results_cropping/crop_visualized.png')
    # Display the image copy
    #print('crop visualized')
    #image_copy.show()

    # Crop the image
    cropped_image = image_copy.crop((pred_x1, pred_y1, pred_x2, pred_y2))
    # Save the cropped image
    cropped_image.save('/content/Fork-Human-Centric-Image-Cropping/results_cropping/visualise_crop.jpg')
    
    make_square = False
    if make_square:
      squared_image = outpaint_image(cropped_image)
      squared_image.save('/content/Fork-Human-Centric-Image-Cropping/results_cropping/visualise_crop.jpg')
      return squared_image
    else:
      return cropped_image

def visualize_heat_map(im, pre_heat):
    im = im.squeeze()
    pre_heat = pre_heat.squeeze()
    im = im.detach().cpu()
    pre_heat = pre_heat.detach().cpu()
    im = im * torch.tensor(IMAGE_NET_STD).view(3,1,1) + torch.tensor(IMAGE_NET_MEAN).view(3,1,1)
    trans_fn = transforms.ToPILImage()
    im = trans_fn(im).convert('RGB')
    width, height = im.size
    pre_heat = trans_fn(pre_heat).convert('RGB').resize((width,height))
    ver_band = (np.ones((height,5,3)) * 255).astype(np.uint8)
    cat_im   = np.concatenate([np.asarray(im), ver_band, np.asarray(pre_heat)], axis=1)
    cat_im   = Image.fromarray(cat_im)
    fig_dir  = os.path.join(cfg.exp_path, 'visualization')
    os.makedirs(fig_dir,exist_ok=True)
    cat_im.save('/content/Fork-Human-Centric-Image-Cropping/results_cropping/heatmap2.png')

if __name__ == '__main__':
    from config_GAICD import cfg
    cfg.use_partition_aware = True
    cfg.partition_aware_type = 9
    cfg.use_content_preserve = True
    cfg.content_preserve_type = 'gcn'
    cfg.only_content_preserve = False

    
    #print(sys.argv[1:])
    #for image_path in sys.argv[1:]:
    image = '/content/Fork-Human-Centric-Image-Cropping/visual_results/GAICD-MS-naive-PA_repeat1/originals/10.png'
    
    image = Image.open(image)
    image = np.array(image)
    print(type(image))
    #image = f"'{image_path}'"
    #print(image)
    crop_image(image)
