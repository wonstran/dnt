# ---------------------------------------------------------
# IOU Tracker
# Copyright (c) 2017 TU Berlin, Communication Systems Group
# Licensed under The MIT License [see LICENSE for details]
# Written by Erik Bochinski
# ---------------------------------------------------------

from unittest import result
import numpy as np
import csv
import os, pathlib
import pandas as pd
from tqdm import tqdm
import math
import multiprocessing as mp
import datetime
import cv2

visdrone_classes = {'car': 4, 'bus': 9, 'truck': 6, 'pedestrian': 1, 'van': 5}

def load_class_dict(name_file:str=None)->dict:
    if not name_file:
        lib_root = pathlib.Path(__file__).resolve().parents[1]
        data_path = os.path.join(lib_root, 'shared/data')
        name_file = os.path.join(data_path, 'coco.names')

    class_dict = pd.read_csv(name_file,header=None).to_dict()
    class_dict_reverse = dict(map(reversed, class_dict.items()))

    print(class_dict_reverse)
    input('...')
    return class_dict_reverse

def load_classes(name_file=None)->list:
    if not name_file:
        lib_root = pathlib.Path(__file__).resolve().parents[1]
        data_path = os.path.join(lib_root, 'shared/data')
        name_file = os.path.join(data_path, 'coco.names')

    my_file = open(name_file, "r") 
    data = my_file.read() 
    results = data.split("\n")
    my_file.close()

    return results

def load_mot16(detections, nms_overlap_thresh=None, with_classes=True, nms_per_class=False, class_filename=""):
    """
    Loads detections stored in a mot-challenge like formatted CSV or numpy array (fieldNames = ['frame', 'id', 'x', 'y',
    'w', 'h', 'score']).

    Args:
        detections (str, numpy.ndarray): path to csv file containing the detections or numpy array containing them.
        nms_overlap_thresh (float, optional): perform non-maximum suppression on the input detections with this thrshold.
                                              no nms is performed if this parameter is not specified.
        with_classes (bool, optional): indicates if the detections have classes or not. set to false for motchallange.
        nms_per_class (bool, optional): perform non-maximum suppression for each class separately

    Returns:
        list: list containing the detections for each frame.
    """

    class_dict = load_class_dict(class_filename)

    if nms_overlap_thresh:
        assert with_classes, "currently only works with classes available"

    data = []
    if type(detections) is str:
        raw = np.genfromtxt(detections, delimiter=',', dtype=float)
        if np.isnan(raw).all():
            raw = np.genfromtxt(detections, delimiter=' ', dtype=float)

    else:
        # assume it is an array
        assert isinstance(detections, np.ndarray), "only numpy arrays or *.csv paths are supported as detections."
        raw = detections.astype(float)

    end_frame = int(np.max(raw[:, 0]))
    pbar = tqdm(total=end_frame, unit=" frames")
    pbar.set_description_str("Pre-processing frames")
    for i in range(1, end_frame+1):
        idx = raw[:, 0] == i
        bbox = raw[idx, 2:6]
        bbox[:, 2:4] += bbox[:, 0:2]  # x1, y1, w, h -> x1, y1, x2, y2
        bbox -= 1  # correct 1,1 matlab offset
        scores = raw[idx, 6]

        if with_classes:
            classes = raw[idx, 7]
            bbox_filtered = None
            scores_filtered = None
            classes_filtered = None
            for coi in class_dict:
                cids = classes==class_dict[coi]
                if nms_per_class and nms_overlap_thresh:
                    bbox_tmp, scores_tmp = nms(bbox[cids], scores[cids], nms_overlap_thresh)
                else:
                    bbox_tmp, scores_tmp = bbox[cids], scores[cids]

                if bbox_filtered is None:
                    bbox_filtered = bbox_tmp
                    scores_filtered = scores_tmp
                    classes_filtered = [coi]*bbox_filtered.shape[0]
                elif len(bbox_tmp) > 0:
                    bbox_filtered = np.vstack((bbox_filtered, bbox_tmp))
                    scores_filtered = np.hstack((scores_filtered, scores_tmp))
                    classes_filtered += [coi] * bbox_tmp.shape[0]

            if bbox_filtered is not None:
                bbox = bbox_filtered
                scores = scores_filtered
                classes = classes_filtered

            if nms_per_class is False and nms_overlap_thresh:
                bbox, scores, classes = nms(bbox, scores, nms_overlap_thresh, np.array(classes))

        else:
            classes = ['car']*bbox.shape[0]

        pbar.update()
        #print("Preprocessing frame: {} of {}".format(i, end_frame), end='\r')
    
        dets = []
        for bb, s, c in zip(bbox, scores, classes):
            dets.append({'bbox': (bb[0], bb[1], bb[2], bb[3]), 'score': s, 'class': c})
        
        data.append(dets)
    
    pbar.close()
    return data

def process_data(raw_data, start_frame, end_frame):
    #start_frame = int(np.min(raw_data[:, 0]))
    #end_frame = int(np.max(raw_data[:, 0]))
    results = []
        
    for i in range(start_frame, end_frame+1):
        idx = raw_data[:, 0] == i
        bbox = raw_data[idx, 2:6]
        bbox[:, 2:4] += bbox[:, 0:2]  # x1, y1, w, h -> x1, y1, x2, y2
        bbox -= 1  # correct 1,1 matlab offset
        scores = raw_data[idx, 6]

        if with_classes_g:
            classes = raw_data[idx, 7]
            bbox_filtered = None
            scores_filtered = None
            classes_filtered = None
            for coi in class_dict_g:
                cids = classes==class_dict_g[coi]
                if nms_per_class_g and nms_overlap_thresh_g:
                    bbox_tmp, scores_tmp = nms(bbox[cids], scores[cids], nms_overlap_thresh_g)
                else:
                    bbox_tmp, scores_tmp = bbox[cids], scores[cids]

                if bbox_filtered is None:
                    bbox_filtered = bbox_tmp
                    scores_filtered = scores_tmp
                    classes_filtered = [coi]*bbox_filtered.shape[0]
                elif len(bbox_tmp) > 0:
                    bbox_filtered = np.vstack((bbox_filtered, bbox_tmp))
                    scores_filtered = np.hstack((scores_filtered, scores_tmp))
                    classes_filtered += [coi] * bbox_tmp.shape[0]

            if bbox_filtered is not None:
                bbox = bbox_filtered
                scores = scores_filtered
                classes = classes_filtered

            if nms_per_class_g is False and nms_overlap_thresh_g:
                bbox, scores, classes = nms(bbox, scores, nms_overlap_thresh_g, np.array(classes))

        else:
            classes = ['car']*bbox.shape[0]

        dets = []
        for bb, s, c in zip(bbox, scores, classes):
            dets.append({'bbox': (bb[0], bb[1], bb[2], bb[3]), 'score': s, 'class': c})
        
        results.append({'frame': i, 'dets':dets})

    return results

def load_mot16_par(detections, nms_overlap_thresh=None, with_classes=True, nms_per_class=False, class_filename="", nWorkers=1, strata_size=100, video_count=0, total_num=0):
    """
    Loads detections stored in a mot-challenge like formatted CSV or numpy array (fieldNames = ['frame', 'id', 'x', 'y',
    'w', 'h', 'score']).

    Args:
        detections (str, numpy.ndarray): path to csv file containing the detections or numpy array containing them.
        nms_overlap_thresh (float, optional): perform non-maximum suppression on the input detections with this thrshold.
                                              no nms is performed if this parameter is not specified.
        with_classes (bool, optional): indicates if the detections have classes or not. set to false for motchallange.
        nms_per_class (bool, optional): perform non-maximum suppression for each class separately

    Returns:
        list: list containing the detections for each frame.
    """
    global with_classes_g
    global class_dict_g
    global nms_per_class_g
    global nms_overlap_thresh_g
    global class_dict_g
    
    nms_overlap_thresh_g = nms_overlap_thresh
    with_classes_g = with_classes
    nms_per_class_g = nms_per_class
    class_dict_g = load_class_dict(class_filename)

    data = []
    if type(detections) is str:
        raw = np.genfromtxt(detections, delimiter=',', dtype=float)
        if np.isnan(raw).all():
            raw = np.genfromtxt(detections, delimiter=' ', dtype=float)
    else:
        # assume it is an array
        assert isinstance(detections, np.ndarray), "only numpy arrays or *.csv paths are supported as detections."
        raw = detections.astype(float)

    pool = mp.Pool(processes=nWorkers)

    def update(*a):
        pbar.update()

    end_frame = int(np.max(raw[:, 0]))
    #nStrata = math.ceil(end_frame/nWorkers)
    nStrata = math.ceil(end_frame/strata_size)

    if (video_count>0) and (total_num>0):
        pbar = tqdm(total=nStrata, desc="Pre-processing {} of {}".format(video_count, total_num), unit=" strata")
    else:
        pbar = tqdm(total=nStrata, desc="Pre-processing", unit=" strata")

    res_list = []
    for strata in range(nStrata):
        start_frame_strata = strata*strata_size+1
        end_frame_strata = min((strata+1)*strata_size, end_frame)
        rows = raw[(raw[:, 0] >= start_frame_strata) & (raw[:, 0] <= end_frame_strata), :]
        res = pool.apply_async(process_data, args=(rows, start_frame_strata, end_frame_strata, ), callback=update)
        res_list.append(res)

    pool.close()
    pool.join()
    pool.terminate()
    pbar.close()
    data = []

    for res in res_list:
        data.extend(res.get())
    data = sorted(data, key = lambda d: d['frame'])

    result = []
    for d in data:
        result.append(d['dets'])

    return result

def load_mot(detections, nms_overlap_thresh=None, with_classes=True, nms_per_class=False):
    """
    Loads detections stored in a mot-challenge like formatted CSV or numpy array (fieldNames = ['frame', 'id', 'x', 'y',
    'w', 'h', 'score']).

    Args:
        detections (str, numpy.ndarray): path to csv file containing the detections or numpy array containing them.
        nms_overlap_thresh (float, optional): perform non-maximum suppression on the input detections with this thrshold.
                                              no nms is performed if this parameter is not specified.
        with_classes (bool, optional): indicates if the detections have classes or not. set to false for motchallange.
        nms_per_class (bool, optional): perform non-maximum suppression for each class separately

    Returns:
        list: list containing the detections for each frame.
    """
    if nms_overlap_thresh:
        assert with_classes, "currently only works with classes available"

    data = []
    if type(detections) is str:
        raw = np.genfromtxt(detections, delimiter=',', dtype=np.float32)
        if np.isnan(raw).all():
            raw = np.genfromtxt(detections, delimiter=' ', dtype=np.float32)

    else:
        # assume it is an array
        assert isinstance(detections, np.ndarray), "only numpy arrays or *.csv paths are supported as detections."
        raw = detections.astype(np.float32)

    end_frame = int(np.max(raw[:, 0]))

    for i in range(1, end_frame+1):
        idx = raw[:, 0] == i
        bbox = raw[idx, 2:6]
        bbox[:, 2:4] += bbox[:, 0:2]  # x1, y1, w, h -> x1, y1, x2, y2
        bbox -= 1  # correct 1,1 matlab offset
        scores = raw[idx, 6]

        if with_classes:
            classes = raw[idx, 7]

            bbox_filtered = None
            scores_filtered = None
            classes_filtered = None
            for coi in visdrone_classes:
                cids = classes==visdrone_classes[coi]
                if nms_per_class and nms_overlap_thresh:
                    bbox_tmp, scores_tmp = nms(bbox[cids], scores[cids], nms_overlap_thresh)
                else:
                    bbox_tmp, scores_tmp = bbox[cids], scores[cids]

                if bbox_filtered is None:
                    bbox_filtered = bbox_tmp
                    scores_filtered = scores_tmp
                    classes_filtered = [coi]*bbox_filtered.shape[0]
                elif len(bbox_tmp) > 0:
                    bbox_filtered = np.vstack((bbox_filtered, bbox_tmp))
                    scores_filtered = np.hstack((scores_filtered, scores_tmp))
                    classes_filtered += [coi] * bbox_tmp.shape[0]

            if bbox_filtered is not None:
                bbox = bbox_filtered
                scores = scores_filtered
                classes = classes_filtered

            if nms_per_class is False and nms_overlap_thresh:
                bbox, scores, classes = nms(bbox, scores, nms_overlap_thresh, np.array(classes))

        else:
            classes = ['car']*bbox.shape[0]

        dets = []
        for bb, s, c in zip(bbox, scores, classes):
            dets.append({'bbox': (bb[0], bb[1], bb[2], bb[3]), 'score': s, 'class': c})
        data.append(dets)

    return data

def nms(boxes, scores, overlapThresh, classes=None):
    """
    perform non-maximum suppression. based on Malisiewicz et al.
    Args:
        boxes (numpy.ndarray): boxes to process
        scores (numpy.ndarray): corresponding scores for each box
        overlapThresh (float): overlap threshold for boxes to merge
        classes (numpy.ndarray, optional): class ids for each box.

    Returns:
        (tuple): tuple containing:

        boxes (list): nms boxes
        scores (list): nms scores
        classes (list, optional): nms classes if specified
    """
    # # if there are no boxes, return an empty list
    # if len(boxes) == 0:
    #     return [], [], [] if classes else [], []

    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    if scores.dtype.kind == "i":
        scores = scores.astype("float")

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    #score = boxes[:, 4]
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(scores)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(overlap > overlapThresh)[0])))

    if classes is not None:
        return boxes[pick], scores[pick], classes[pick]
    else:
        return boxes[pick], scores[pick]


def save_to_csv(out_path, tracks, viou=False):
    """
    Saves tracks to a CSV file.

    Args:
        out_path (str): path to output csv file.
        tracks (list): list of tracks to store.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline='') as ofile:
        if viou == False:
            field_names = ['frame', 'id', 'x', 'y', 'w', 'h', 'score', 'wx', 'wy', 'wz']
        else:
            field_names = ['frame', 'id', 'x', 'y', 'w', 'h', 'score', 'object_category', 'truncation', 'occlusion']

        odict = csv.DictWriter(ofile, field_names)
        id_ = 1
        for track in tracks:
            for i, bbox in enumerate(track['bboxes']):
                row = {'id': id_,
                       'frame': track['start_frame'] + i,
                       'x': bbox[0]+1,
                       'y': bbox[1]+1,
                       'w': bbox[2] - bbox[0],
                       'h': bbox[3] - bbox[1],
                       'score': track['max_score']}
                if viou == False:
                    row['wx'] = -1
                    row['wy'] = -1
                    row['wz'] = -1
                else:
                    row['object_category'] = visdrone_classes[track['class']]
                    row['truncation'] = -1
                    row['occlusion'] = -1

                odict.writerow(row)
            id_ += 1


def iou(bbox1, bbox2):
    """
    Calculates the intersection-over-union of two bounding boxes.

    Args:
        bbox1 (numpy.array, list of floats): bounding box in format x1,y1,x2,y2.
        bbox2 (numpy.array, list of floats): bounding box in format x1,y1,x2,y2.

    Returns:
        int: intersection-over-onion of bbox1, bbox2
    """

    bbox1 = [float(x) for x in bbox1]
    bbox2 = [float(x) for x in bbox2]

    (x0_1, y0_1, x1_1, y1_1) = bbox1
    (x0_2, y0_2, x1_2, y1_2) = bbox2

    # get the overlap rectangle
    overlap_x0 = max(x0_1, x0_2)
    overlap_y0 = max(y0_1, y0_2)
    overlap_x1 = min(x1_1, x1_2)
    overlap_y1 = min(y1_1, y1_2)

    # check if there is an overlap
    if overlap_x1 - overlap_x0 <= 0 or overlap_y1 - overlap_y0 <= 0:
        return 0

    # if yes, calculate the ratio of the overlap to each ROI size and the unified size
    size_1 = (x1_1 - x0_1) * (y1_1 - y0_1)
    size_2 = (x1_2 - x0_2) * (y1_2 - y0_2)
    size_intersection = (overlap_x1 - overlap_x0) * (overlap_y1 - overlap_y0)
    size_union = size_1 + size_2 - size_intersection

    return size_intersection / size_union

def get_files(folder, extention=None, prefix_fodler = False):
    all_files = os.listdir(folder)
    if extention:
        all_files = [fname for fname in all_files if fname.endswith(extention)]
    if prefix_fodler:
        all_files = [os.path.join(folder, fname) for fname in all_files]
    return all_files

def hhmmss2seconds(hhmmss):
    h,m,s = hhmmss.split(':')
    return datetime.timedelta(hours=int(h),minutes=int(m),seconds=float(s)).total_seconds()

def get_fps(video_file):
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        raise IOError("Couldn't open webcam or video")
    video_fps    = int(cap.get(cv2.CAP_PROP_FPS))      
    return video_fps

def convert_review(ious, fps):
    if fps>0:
        ious[ious.shape[1]] = ious[0].apply(lambda f: f/fps)
        
    else:
        ious[ious.shape[1]] = 0

    results = pd.DataFrame()
    results[0] = ious[8]
    results[1] = ious[7]
    results[2] = ious[6]

    return results

def frame_min_max(df):
    s = df.stack()
    return s.min(), s.max()

def get_fps(input_video):
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print('Failed to open the video!')
    fps = cap.get(cv2.CAP_PROP_FPS)
    return fps
