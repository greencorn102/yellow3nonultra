"""
Detection for YOLOv3
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt

# FILE = Path(__file__).resolve()
# ROOT = FILE.parents[0]  # root directory
# if str(ROOT) not in sys.path:
#     sys.path.append(str(ROOT))  # add ROOT to PATH
# ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

#from models.common import DetectMultiBackend
from models.experimental import attempt_load
#from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
#from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
#                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.general import non_max_suppression

#from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync

from timeit import default_timer as timer

import glob
import numpy as np



conf_thres=0.25  # confidence threshold
iou_thres=0.45  # NMS IOU threshold
max_det=1000  # maximum detections per image

view_img=False  # show results
save_txt=False # save results to *.txt
save_conf=False  # save confidences in --save-txt labels
save_crop=False  # save cropped prediction boxes
nosave=False  # do not save images/videos
classes=None  # filter by class: --class 0, or --class 0 2 3
agnostic_nms=False  # class-agnostic NMS
augment=False  # augmented inference
visualize=False  # visualize features
update=False  # update all models
###project=ROOT / 'runs/detect'  # save results to project/name
name='exp'  # save results to project/name
exist_ok=False # existing project/name ok, do not increment
line_thickness=3  # bounding box thickness (pixels)
hide_labels=False  # hide labels
hide_conf=False  # hide confidences
half=False # use FP16 half-precision inference
dnn=False



weights='runs/train/exp10/weights/best.pt'
device='cuda:0'
device = select_device(device)
model = attempt_load(weights, device=device, inplace=True, fuse=True)
model = model.half()


### Predict loop for all images in a folder

root_dir = 'data/test'


f=0
for filename in glob.iglob(root_dir + '**/*.jpg', recursive=True):
    start=timer()
    print(filename)
    img_file = cv2.imread(filename)
    
    img=img_file.transpose(2,0,1).reshape(1,3,640,640) ### MUST BE A MULTIPLE OF 64 !!!
    im = torch.from_numpy(img).to(device)
    im = im.half()
    im /= 255
    pred0=model(im)
    pred = non_max_suppression(pred0)
    
    cdts1=pred[0].cpu().detach().numpy()
    print('cdts1',cdts1)
    cc=[]
    for ii in range(len(cdts1)):
        p=cdts1[ii][0:4]
        for j in p:
            cc.append(j)
    print(cc)
    #print(len(cc))
    qqqa=[]
    for kk in range((int(len(cc)/4))):
        qq = cc[(0+(kk*4)):(4+(kk*4))]
        qqqa.append(qq)
    print(qqqa)
    #print(len(qqqa))
    for k in range(len(qqqa)):
        w=abs(int(qqqa[k][0]))
        x=abs(int(qqqa[k][1]))
        y=abs(int(qqqa[k][2]))
        z=abs(int(qqqa[k][3]))
        print(w,x,y,z)
        cv2.rectangle(
            img_file, (w, x), (y, z), (0, 0, 255), 4, cv2.LINE_AA
        )

        #print(k)    
    cv2.imwrite('X/'+filename, img_file)
    f=f+1
    
    vtime = timer() - start
    print(vtime)  


 
