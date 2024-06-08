import sys, os
import numpy as np
import torch

from .deep.feature_extractor import Extractor, FastReIDExtractor
from .sort.nn_matching import NearestNeighborDistanceMetric
from .sort.preprocessing import non_max_suppression
from .sort.detection import Detection
from .sort.tracker import Tracker

__all__ = ['DeepSort']

class DeepSort(object):
    def __init__(self, cfg: dict, gpu:bool =True):
        
        self.model_path = cfg['reid_ckpt']              # path to the reid checkpoint (model weights)
        self.model_config= cfg['reid_cfg']              # path to the reid config file (yaml)
        self.max_cosine_distance = cfg['max_dist']
        self.min_confidence = cfg['min_confidence']     # ignore detections if confidence lower than this value
        self.nms_max_overlap = cfg['nms_max_overlap']
        self.max_iou_distance = cfg['max_iou_distance'] 
        self.max_age = cfg['max_age']           # if a track is not matched in max_age frames, it will be deleted
        self.n_init = cfg['n_init']             # number of detections before creating a track
        self.nn_budget = cfg['nn_budget']
        
        if self.model_config is None:
            self.extractor = Extractor(os.path.join(os.path.dirname(__file__), 'deep/checkpoint/', self.model_path)
                                       , use_cuda=gpu)   # default extractor
        else:
            cfg_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../third_party/fast-reid/configs/', self.model_config))
            ckpt_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../third_party/fast-reid/checkpoint/', self.model_path))
            self.extractor = FastReIDExtractor(cfg_path, ckpt_path, use_cuda=gpu)    # fast-reid extractor

        metric = NearestNeighborDistanceMetric("cosine", self.max_cosine_distance, self.nn_budget)
        self.tracker = Tracker(metric, max_iou_distance=self.max_iou_distance, max_age=self.max_age, n_init=self.n_init)

    def update(self, bbox_xywh, confidences, ori_img):
        self.height, self.width = ori_img.shape[:2]

        # generate detections
        features = self._get_features(bbox_xywh, ori_img)   # extract features for bboxes
        bbox_tlwh = self._xywh_to_tlwh(bbox_xywh)           # convert bbox from xc_yc_w_h to left top width height/width
        detections = [Detection(bbox_tlwh[i], conf, features[i]) 
                        for i, conf in enumerate(confidences) if conf>self.min_confidence] # ignore low confidence bboxes

        # run on non-maximum supression
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = non_max_suppression(boxes, self.nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # update tracker
        self.tracker.predict()              # predict the location of the bboxes
        self.tracker.update(detections)     # update tracks by location and appearance

        # output bbox identities
        outputs = []
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            box = track.to_tlwh()
            x1,y1,x2,y2 = self._tlwh_to_xyxy(box)
            track_id = track.track_id
            outputs.append(np.array([x1,y1,x2,y2,track_id], dtype=int))
        if len(outputs) > 0:
            outputs = np.stack(outputs,axis=0)
        return outputs

    """
    TODO:
        Convert bbox from xc_yc_w_h to xtl_ytl_w_h
    Thanks JieChen91@github.com for reporting this bug!
    """
    @staticmethod
    def _xywh_to_tlwh(bbox_xywh):
        if isinstance(bbox_xywh, np.ndarray):
            bbox_tlwh = bbox_xywh.copy()
        elif isinstance(bbox_xywh, torch.Tensor):
            bbox_tlwh = bbox_xywh.clone()
        bbox_tlwh[:,0] = bbox_xywh[:,0] - bbox_xywh[:,2]/2.
        bbox_tlwh[:,1] = bbox_xywh[:,1] - bbox_xywh[:,3]/2.
        return bbox_tlwh


    def _xywh_to_xyxy(self, bbox_xywh):
        x,y,w,h = bbox_xywh
        x1 = max(int(x-w/2),0)
        x2 = min(int(x+w/2),self.width-1)
        y1 = max(int(y-h/2),0)
        y2 = min(int(y+h/2),self.height-1)
        return x1,y1,x2,y2

    def _tlwh_to_xyxy(self, bbox_tlwh):
        """
        TODO:
            Convert bbox from xtl_ytl_w_h to xc_yc_w_h
        Thanks JieChen91@github.com for reporting this bug!
        """
        x,y,w,h = bbox_tlwh
        x1 = max(int(x),0)
        x2 = min(int(x+w),self.width-1)
        y1 = max(int(y),0)
        y2 = min(int(y+h),self.height-1)
        return x1,y1,x2,y2

    def _xyxy_to_tlwh(self, bbox_xyxy):
        x1,y1,x2,y2 = bbox_xyxy

        t = x1
        l = y1
        w = int(x2-x1)
        h = int(y2-y1)
        return t,l,w,h
    
    def _get_features(self, bbox_xywh, ori_img):
        im_crops = []
        for box in bbox_xywh:
            x1,y1,x2,y2 = self._xywh_to_xyxy(box)
            im = ori_img[y1:y2,x1:x2]
            im_crops.append(im)
        if im_crops:
            features = self.extractor(im_crops)
        else:
            features = np.array([])
        return features


