

from __future__ import division
from functools import lru_cache
from typing import Tuple, Optional
# -*- coding: utf-8 -*-
# @Organization  : insightface.ai
# @Author        : Jia Guo
# @Time          : 2021-05-04
# @Function      :
# from ekyc_services.face_utils.utils import Face
# from utils import label_dict
import numpy as np
import onnxruntime
import os
import os.path as osp
import cv2

onnxruntime.set_default_logger_severity(3)


cur_dir = os.path.dirname(os.path.abspath(__file__))


def softmax(z):
    assert len(z.shape) == 2
    s = np.max(z, axis=1)
    s = s[:, np.newaxis]  # necessary step to do broadcasting
    e_x = np.exp(z - s)
    div = np.sum(e_x, axis=1)
    div = div[:, np.newaxis]  # dito
    return e_x / div


def distance2bbox(points: np.ndarray, distance: np.ndarray, max_shape: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """
    Decode distance prediction to bounding box.

    Args:
        points: A 2D tensor of shape (n, 2) representing the coordinates of the points.
        distance: A 2D tensor of shape (n, 4) representing the distance from each point to the four boundaries of a bounding box (left, top, right, bottom).
        max_shape: An optional tuple representing the shape of the image. If provided, the bounding boxes will be clamped within the image boundaries.

    Returns:
        A tensor of shape (n, 4) representing the decoded bounding boxes.
    """
    # print(points.shape)
    # print(distance.shape)
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    if max_shape is not None:
        x1 = np.clip(x1, 0, max_shape[1])
        y1 = np.clip(y1, 0, max_shape[0])
        x2 = np.clip(x2, 0, max_shape[1])
        y2 = np.clip(y2, 0, max_shape[0])
    return np.stack([x1, y1, x2, y2], axis=-1)


def distance2kps(points, distance, max_shape=None):
    """Decode distance prediction to keypoints.

    Args:
        points (np.ndarray): Shape (n, 2), [x, y].
        distance (np.ndarray): Distance from the given point to keypoints.
        max_shape (tuple): Shape of the image.

    Returns:
        np.ndarray: Decoded keypoints.
    """
    px = points[:, 0, np.newaxis] + distance[:, 0::2]
    py = points[:, 1, np.newaxis] + distance[:, 1::2]
    if max_shape is not None:
        px = np.clip(px, 0, max_shape[1])
        py = np.clip(py, 0, max_shape[0])
    preds = np.stack((px, py), axis=-1)
    return preds.reshape(points.shape[0], -1)


class SCRFD:
    def __init__(self, model_file=os.path.join(cur_dir, 'weights', 'scrfd_10g.onnx'), session=None, topK=5000):
        self.model_file = model_file
        self.session = session
        self.batched = False
        if self.session is None:
            assert self.model_file is not None
            assert osp.exists(self.model_file)
            self.session = onnxruntime.InferenceSession(self.model_file, None)
        self.center_cache = {}
        self.nms_thresh = 0.3
        self.det_thresh = 0.3
        self._topK = topK
        self._init_vars()

    def set_config(self, confThreshold=0.6, nmsThreshold=0.3, topK=5000):
        self.det_thresh = confThreshold
        self.nms_thresh = nmsThreshold
        self._topK = topK

    def _init_vars(self):
        input_cfg = self.session.get_inputs()[0]
        input_shape = input_cfg.shape
        print(input_shape)
        # print(input_shape)
        self.input_size = None
        # print('image_size:', self.image_size)
        input_name = input_cfg.name
        self.input_shape = input_shape
        outputs = self.session.get_outputs()
        # if len(outputs[0].shape) == 3:
        #     self.batched = True
        output_names = []
        for o in outputs:
            output_names.append(o.name)
        self.input_name = input_name
        self.output_names = output_names
        self.input_mean = 127.5
        self.input_std = 128.0
        print(f"input_name: {self.input_name}")
        print(f"output_names: {self.output_names}")
        self.use_kps = False
        self._anchor_ratio = 1.0
        self._num_anchors = 1
        self.fmc = 3
        self._feat_stride_fpn = [8, 16, 32]
        self._num_anchors = 2
        self.use_kps = True
        print('----')
        print(self.fmc)
        print(self._feat_stride_fpn)
        print(self._num_anchors)
        print(self.use_kps)
        print('----')
        print(self.batched)

    def prepare(self, ctx_id, **kwargs):
        if ctx_id < 0:
            self.session.set_providers(['CPUExecutionProvider'])
        nms_thresh = kwargs.get('nms_thresh', None)
        if nms_thresh is not None:
            self.nms_thresh = nms_thresh
        det_thresh = kwargs.get('det_thresh', None)
        if det_thresh is not None:
            self.det_thresh = det_thresh
        input_size = kwargs.get('input_size', None)
        if input_size is not None:
            if self.input_size is not None:
                print('warning: det_size is already set in scrfd model, ignore')
            else:
                self.input_size = input_size

    @lru_cache(maxsize=100)
    def get_anchor_centers(self, height, width, stride):
        anchor_centers = np.stack(
            np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)

        anchor_centers = (anchor_centers * stride).reshape((-1, 2))
        if self._num_anchors > 1:
            anchor_centers = np.stack(
                [anchor_centers]*self._num_anchors, axis=1).reshape((-1, 2))
        return anchor_centers

    def create_blob(self, img):
        input_size = tuple(img.shape[0:2][::-1])
        blob = cv2.dnn.blobFromImage(img, 1.0/self.input_std, input_size,
                                     (self.input_mean, self.input_mean, self.input_mean), swapRB=True)
        return blob

    def forward(self, img, threshold):
        scores_list, bboxes_list, kpss_list = [], [], []
        blob = self.create_blob(img)
        net_outs = self.session.run(self.output_names, {self.input_name: blob})
        print(len(net_outs))
        print(type(net_outs))
        input_height = blob.shape[2]
        input_width = blob.shape[3]
        fmc = self.fmc
        for idx, stride in enumerate(self._feat_stride_fpn):
            if self.batched:
                scores = net_outs[idx][0]
                bbox_preds = net_outs[idx + fmc][0]
                print('bbox_preds', bbox_preds.shape)
                bbox_preds = bbox_preds * stride
                print('bbox_preds', bbox_preds.shape)
                if self.use_kps:
                    kps_preds = net_outs[idx + fmc * 2][0] * stride
            else:
                scores = net_outs[idx]
                bbox_preds = net_outs[idx + fmc]
                print('bbox_preds', bbox_preds.shape)
                bbox_preds = bbox_preds * stride
                print('bbox_preds', bbox_preds.shape)
                if self.use_kps:
                    kps_preds = net_outs[idx + fmc * 2] * stride

            height = input_height // stride
            width = input_width // stride
            # print('sx---xs')
            # print(height)
            # print(width)
            # print('sx---xs')
            anchor_centers = self.get_anchor_centers(height, width, stride)

            pos_inds = np.where(scores >= threshold)[0]
            # print('0----0')
            # print(bbox_preds.shape)
            # print(anchor_centers.shape)
            # print('0----0')
            bboxes = distance2bbox(anchor_centers, bbox_preds)
            pos_scores = scores[pos_inds]
            pos_bboxes = bboxes[pos_inds]
            scores_list.append(pos_scores)
            bboxes_list.append(pos_bboxes)
            if self.use_kps:
                kpss = distance2kps(anchor_centers, kps_preds)
                kpss = kpss.reshape((kpss.shape[0], -1, 2))
                pos_kpss = kpss[pos_inds]
                kpss_list.append(pos_kpss)
        return scores_list, bboxes_list, kpss_list

    def detect(self, img, input_size=None, max_num=None, metric='default'):
        """
        Detect objects in an image.

        Args:
            img (numpy.ndarray): The input image.
            input_size (list, optional): The size of the input image. Defaults to None.
            max_num (int, optional): The maximum number of objects to detect. Defaults to None.
            metric (str, optional): The metric used for object selection. Defaults to 'default'.

        Returns:
            numpy.ndarray: The detected objects.
            numpy.ndarray: The keypoints of the detected objects.
        """
        input_size = input_size or self.input_size

        if input_size is None:
            raise ValueError(
                "input_size is not provided and self.input_size is not set")

        img_height = img.shape[0]
        img_width = img.shape[1]
        im_ratio = float(img_height) / img_width
        model_ratio = float(input_size[1]) / input_size[0]

        if im_ratio > model_ratio:
            new_height = input_size[1]
            new_width = int(new_height / im_ratio)
        else:
            new_width = input_size[0]
            new_height = int(new_width * im_ratio)

        det_scale = float(new_height) / img_height
        resized_img = cv2.resize(img, (new_width, new_height))
        det_img = np.zeros((input_size[1], input_size[0], 3), dtype=np.uint8)
        det_img[:new_height, :new_width, :] = resized_img

        scores_list, bboxes_list, kpss_list = self.forward(
            det_img, self.det_thresh)

        scores = np.vstack(scores_list)
        scores_ravel = scores.ravel()
        order = scores_ravel.argsort()[::-1]
        bboxes = np.vstack(bboxes_list) / det_scale

        if self.use_kps:
            kpss = np.vstack(kpss_list) / det_scale
        else:
            kpss = None

        pre_det = np.hstack((bboxes, scores)).astype(np.float32, copy=False)
        pre_det = pre_det[order, :]
        keep = self.nms(pre_det)
        det = pre_det[keep, :]

        det[:, 0:4] /= np.array(
            [img_width, img_height, img_width, img_height],
            dtype=np.float32
        )

        if self.use_kps:
            kpss = kpss[order, :, :]
            kpss = kpss[keep, :, :]
            kpss[:, :, 0] /= float(img_width)
            kpss[:, :, 1] /= float(img_height)

        else:
            kpss = None

        if max_num is not None and max_num > 0 and det.shape[0] > max_num:
            area = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
            img_center = img_height // 2, img_width // 2
            offsets = np.vstack([
                (det[:, 0] + det[:, 2]) / 2 - img_center[1],
                (det[:, 1] + det[:, 3]) / 2 - img_center[0]
            ])
            offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)

            if metric == 'max':
                values = area
            else:
                values = area - offset_dist_squared * 2.0

            bindex = np.argsort(values)[::-1]
            bindex = bindex[0:max_num]
            det = det[bindex, :]

            if kpss is not None:
                kpss = kpss[bindex, :]

        return det, kpss

    def nms(self, dets):
        thresh = self.nms_thresh
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scores = dets[:, 4]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = np.array([], dtype=int)
        while order.size > 0:
            i = order[0]
            keep = np.append(keep, i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            # inds = np.where(ovr <= thresh)[0]
            inds = np.nonzero(ovr <= thresh)[0]
            order = order[inds + 1]

        return keep

    def inference(self, img, input_size=[320, 320], max_num=0, metric='default'):
        total_boxes = np.zeros((0, 5), dtype=np.float32)
        total_landmark = np.zeros((0, 5, 2), dtype=np.float32)
        ax = np.arange(5, dtype=np.float32)
        bboxes, lmks = self.detect(
            img,
            input_size=input_size,
            max_num=max_num,
            metric='default')
        if bboxes.shape[0] != 0:
            total_boxes = np.vstack(
                [total_boxes, np.hstack((bboxes[:, :4], bboxes[:, 4:5]))])
            total_landmark = np.vstack(
                [total_landmark, lmks.reshape(-1, 5, 2)])
        return total_boxes, total_landmark


if __name__ == '__main__':
    scrfd = SCRFD(
        '/home/misa/Workshop/MISA.MFace3/mface_services/models/scrfd/weights/scrfd_10g.onnx')
    img = cv2.imread(
        '/home/misa/Downloads/PTom_other_1.jpg')
    print(img.shape)
    print('HAHLKDSAFNLAKNFLKANFDS')
    res = scrfd.inference(img, [320, 320])
    # print(res[0].shape, res[1].shape)
    # print(res[0])
    # print(res[1])
    # draw
    # scale res to w, h
    res[0][:, 0] *= img.shape[1]
    res[0][:, 1] *= img.shape[0]
    res[0][:, 2] *= img.shape[1]
    res[0][:, 3] *= img.shape[0]
    res[1][:, :, 0] *= img.shape[1]
    res[1][:, :, 1] *= img.shape[0]

    cv2.rectangle(img, (int(res[0][0][0]), int(res[0][0][1])),
                  (int(res[0][0][2]), int(res[0][0][3])), (0, 255, 0), 2)
    for i in range(5):
        cv2.circle(img, (int(res[1][0][i][0]), int(
            res[1][0][i][1])), 2, (125, 125, 125), -1)
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
