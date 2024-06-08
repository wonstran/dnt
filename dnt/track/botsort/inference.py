import os, sys
import cv2
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from bot_tracker.mc_bot_sort import BoTSORT
from config import Config

def track(video_file:str, det_file:str, out_file:str = None, cfg:dict = Config.get_cfg_botsort(), 
          video_index:int = None, total_videos:int = None)->pd.DataFrame:

    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        raise IOError("Couldn't open webcam or video")
    
    if video_file is not None:
        fps = cap.get(cv2.CAP_PROP_FPS)
    else:
        fps = cfg['fps']

    botsort = BoTSORT(cfg, frame_rate=fps)
    
    detections = pd.read_csv(det_file, header=None).to_numpy()
    start_frame = int(min(detections[:,0]))
    end_frame = int(max(detections[:,0]))
    tot_frames = max((end_frame - start_frame + 1), 0)

    pbar = tqdm(total=tot_frames, desc='Tracking {}'.format(os.path.basename(video_file)), unit = 'frames')
    if video_index and total_videos:
        pbar.desc = 'Tracking {} of {}'.format(video_index, total_videos)

    results = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    while cap.isOpened():

        pos_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        ret, frame = cap.read()

        if (not ret) or (pos_frame>end_frame):
            break
        
        im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_dets = detections[np.where(detections[:,0] == pos_frame)]
        nrows, ncols = frame_dets.shape

        if nrows > 0:
            dets = []
            for det in frame_dets:
                dets.append([det[2], det[3], det[2]+det[4], det[3]+det[5], det[6], det[7]]) # x1, y1, x2, y2, score, class
            dets = np.array(dets)
            '''
            for det in frame_dets:
                bbox_xywh.append([det[2]+det[4]/2, det[3]+det[5]/2, det[4], det[5]])
            bbox_xywh = np.array(bbox_xywh)
            conf_score = np.array(frame_dets[:,6])
            '''
            outputs = botsort.update(dets, im)

            if len(outputs) > 0:
                for t in outputs:
                    tlwh = t.tlwh
                    tlbr = t.tlbr
                    tid = t.track_id
                    tcls = t.cls
                    score = t.score
                    vertical = tlwh[2] / tlwh[3] > cfg['aspect_ratio_thresh']
                    if tlwh[2] * tlwh[3] > cfg['min_box_area'] and not vertical:
                        results.append([pos_frame, tid, int(tlwh[0]), int(tlwh[1]), int(tlwh[2]), int(tlwh[3]), round(score,1), int(tcls), -1, -1])

        pbar.update()
    cap.release()
    pbar.close()

    df = pd.DataFrame(results)
    if out_file:
        if cfg['output_header']:
            header = ['frame', 'track_id', 'x', 'y', 'w', 'h', 'score', 'class', 'r3','r4']
        else:
            header = None
        df.to_csv(out_file, index=False, header=header)

    return df

if __name__ == "__main__":

    video_file = "/mnt/d/videos/ped2stage/videos/gh1293.mp4"
    iou_file = "/mnt/d/videos/ped2stage/dets/gh1293_iou_ped.txt"
    out_file = "/mnt/d/videos/ped2stage/tracks/gh1293_ped_track_2.txt"

    track(video_file, iou_file, out_file)
    