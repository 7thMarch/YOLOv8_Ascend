import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import random
random.seed(0)

import glob
import cv2
from pathlib import Path
from ultralytics import YOLO

# Load Images
dataroot_path = r'D:\datasets\VisDrone\VisDrone2019-DET-test-dev\images'
labelroot_path = r'D:\datasets\VisDrone\VisDrone2019-DET-test-dev\labels'
image_paths = glob.glob(dataroot_path+'\\*')

for img in image_paths:
    img = Path(img)
    