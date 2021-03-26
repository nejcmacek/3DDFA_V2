# coding: utf-8

import os
import pathlib

import imageio
import numpy as np
import yaml
from tqdm import tqdm
import cv2

from FaceBoxes.FaceBoxes_ONNX import FaceBoxes_ONNX
from TDDFA_ONNX import TDDFA_ONNX
from utils.pose import P2sRt, matrix2angle


class Gaze:

    def __init__(self, root, config):
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
        os.environ['OMP_NUM_THREADS'] = '4'

        cfg = yaml.load(open(os.path.join(root, config)), Loader=yaml.SafeLoader)
        cfg["checkpoint_fp"] = os.path.join(root, cfg["checkpoint_fp"])
        cfg["bfm_fp"] = os.path.join(root, cfg["bfm_fp"])

        self.face_boxes = FaceBoxes_ONNX()
        self.tddfa = TDDFA_ONNX(**cfg)

        reader = imageio.get_reader("<video0>")
        self.frames = iter(tqdm(enumerate(reader)))
        self.pre_ver = None
        self.has_boxes = False

    def read_next(self):
        _, frame = next(self.frames)
        frame_bgr = frame[..., ::-1]  # RGB->BGR

        if not self.has_boxes:
            # the first frame, detect face, here we only use the first face, you can change depending on your need
            boxes = self.face_boxes(frame_bgr)

            if not boxes:
                return None

            boxes = [boxes[0]]
            param_lst, roi_box_lst = self.tddfa(frame_bgr, boxes)
            ver = self.tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=False)[0]

            # refine
            param_lst, roi_box_lst = self.tddfa(frame_bgr, [ver], crop_policy='landmark')
            ver = self.tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=False)[0]

            self.has_boxes = True
        else:
            param_lst, roi_box_lst = self.tddfa(frame_bgr, [self.pre_ver], crop_policy='landmark')
            roi_box = roi_box_lst[0]

            # potential improvement: add confidence threshold to judge the tracking is failed
            if abs(roi_box[2] - roi_box[0]) * abs(roi_box[3] - roi_box[1]) < 2020:
                boxes = self.face_boxes(frame_bgr)

                if not boxes:
                    self.has_boxes = False
                    return None
                
                boxes = [boxes[0]]
                param_lst, roi_box_lst = self.tddfa(frame_bgr, boxes)

            ver = self.tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=False)[0]

        self.pre_ver = ver

        # compute rotation angles
        param = param_lst[0]
        P = param[:12].reshape(3, -1) # camera matrix
        _, R, t3d = P2sRt(P)
        P = np.concatenate((R, t3d.reshape(3, -1)), axis=1)  # without scale
        pose = matrix2angle(R)
        return pose


if __name__ == '__main__':
    root = pathlib.Path(__file__).parent
    gaze = Gaze(root, 'configs/mb1_120x120.yml')
    for i in range(500):
        transform = gaze.read_next()
        transform = [p / np.pi * 180 for p in transform] if transform else None
        print(transform)
