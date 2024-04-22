import torch
import numpy as np
import cv2
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_boxes
from utils.BaseDetector import baseDet
from utils.torch_utils import select_device
from utils.datasets import letterbox
from datetime import datetime

class Detector(baseDet):

    def __init__(self):
        super(Detector, self).__init__()
        self.init_model()
        self.build_config()
        self.last_detection_times = {}  # 初始化最后检测时间的字典
        self.occluded_flags = {}  # 初始化遮挡标志的字典，用于跟踪每个人物ID是否被遮挡

    def init_model(self):
        self.weights = 'weights/yolov5s.pt'
        self.device = '0' if torch.cuda.is_available() else 'cpu'
        self.device = select_device(self.device)
        model = attempt_load(self.weights)  # Remove map_location here
        model.to(self.device).eval()  # This line already correctly assigns the model to the device
        model.float()
        self.m = model
        self.names = model.module.names if hasattr(model, 'module') else model.names

    def preprocess(self, img):

        img0 = img.copy()
        img = letterbox(img, new_shape=self.img_size)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.float()  # 半精度
        img /= 255.0  # 图像归一化
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        return img0, img

    def detect(self, im):

        im0, img = self.preprocess(im)

        pred = self.m(img, augment=False)[0]
        pred = pred.float()
        pred = non_max_suppression(pred, self.threshold, 0.4)

        pred_boxes = []
        for det in pred:

            if det is not None and len(det):
                det[:, :4] = scale_boxes(
                    img.shape[2:], det[:, :4], im0.shape).round()

                for *x, conf, cls_id in det:
                    lbl = self.names[int(cls_id)]
                    if not lbl in ['person', 'car', 'truck']:
                        continue
                    x1, y1 = int(x[0]), int(x[1])
                    x2, y2 = int(x[2]), int(x[3])
                    pred_boxes.append(
                        (x1, y1, x2, y2, lbl, conf))

        return im, pred_boxes

