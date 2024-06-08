import cv2, os
import pandas as pd
from pathlib import Path
import numpy as np
from tqdm import tqdm
from PIL import Image
from typing import List
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.models.resnet import ResNet18_Weights
from shared.download import download_file
from matplotlib import pyplot as plt

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.resnet18 = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.resnet18.fc = nn.Linear(512, 2)

    def forward(self, x):
        return self.resnet18(x)
    
class SignalDetector:
    def __init__(self, det_zones:list, model:str='ped', weights:str=None, batchsz:int=64):
        '''
        Detect traffic signal status

        Parameters:
        - det_zones: cropped zones for detection list[(x, y, w, h)]
        - model: detection model, default is 'ped', 'custom'
        - weights: path of weights, default is None
        - batchsz: the batch size for prediction, default is 64
        '''
        
        self.det_zones = det_zones

        cwd = Path(__file__).parent.absolute()
        if not weights:
            if (model == 'ped'):
                weights = os.path.join(cwd, 'weights', 'ped_signal.pt')            

        if not os.path.exists(weights):
            url = 'https://its.cutr.usf.edu/alms/downloads/ped_signal.pt'
            download_file(url, weights)    
        
        self.model = Model()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.load_state_dict(torch.load(weights))
        self.model.to(self.device)

        self.batchsz = batchsz
    
    def detect(self, input_video:str, det_file:str=None,
               video_index:int=None, video_tot:int=None):
        '''
        Parameters:
        - input_video: the video path
        - det_file: the file name for output
        - video_index: the index of video
        - video_tot: the total number of videos

        Return:
        - a dataframe with prediction results, ['frame', 'zone0', 'zone1', ...]
        '''

        # initialize the result array
        results = np.array([])    #np.full((0, len(self.det_zones)), -1)
        frames = np.array([])
        zones = np.array([])
        batch = []
        temp_frames=[]

        # open input video
        cap = cv2.VideoCapture(input_video)
        if not cap.isOpened():
            print('Failed to open the video!')

        tot_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        pbar = tqdm(total=tot_frames, unit=' frame')
        if video_index and video_tot: 
            pbar.set_description_str("Detecting signals {} of {}".format(video_index, video_tot))
        else:
            pbar.set_description_str("Detecting signals ")
        
        while cap.isOpened():
            pos_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            ret, frame = cap.read()
            if not ret:
                break
        
            crop_img = self.crop_zone(frame)
            batch.append(crop_img)
            temp_frames.append(pos_frame)

            if ((pos_frame+1) % self.batchsz == 0) or (pos_frame >= tot_frames-1):
                #batch_pred = self.predict(batch).reshape(-1, len(self.det_zones))
                #results = np.append(results, batch_pred, axis=0)

                batch_pred = self.predict(batch).flatten()
                results = np.append(results, batch_pred, axis=0)
                zones = np.append(zones, np.tile(np.array(list(range(len(self.det_zones)))), self.batchsz), axis=0)
                frames = np.append(frames, np.repeat(np.array(temp_frames), len(self.det_zones)), axis=0)

                batch=[]
                temp_frames = []

            pbar.update()

        pbar.close()
        cap.release()

        df = pd.DataFrame(list(zip(frames, zones, results)), columns=['frame', 'signal', 'detection'])

        if det_file:
            df.to_csv(det_file, index=False)

        return df

    def gen_ped_interval(self, dets:pd.DataFrame, input_video:str, walk_interval:int, countdown_interval:int, 
                         out_file:str, factor:float=0.75, video_index:int=None, video_tot:int=None):
        '''
        Parameters:
        - dets: the dataframe for signal detections
        - input_video: the video path
        - walk_interval: the pedestrian walking interval 4s to 7s
        - countdown_interval: the pedestrian countdown interval, cross-length(ft)/4(ft/s)
        - factor: the factor to identify a walking signal (0, 1), 
                    default is 0.75 (75% of frames in a sliding window is walking, then walking is identified)

        Return:
        - a dataframe of ped sigal intervals ['signal', 'walk_beg', 'walk_end', 'countdown_beg', 'countdown_end']  
        '''

         # open input video
        cap = cv2.VideoCapture(input_video)
        if not cap.isOpened():
            print('Failed to open the video!')
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        
        results = []
        for i in range(len(self.det_zones)):
            dets_i = dets[dets['signal']==i]
            results.append(self.scan_walk_interval(dets_i, int(fps*walk_interval), factor, i, fps))

        df = pd.concat(results, axis=0)
        if out_file:
            df.to_csv(out_file, index=False)

        return df

    def scan_walk_interval(self, dets:pd.DataFrame, window:int, factor:float, zone:int, fps:int=30,
                           video_index:int=None, video_tot:int=None) -> pd.DataFrame:

        sequence = dets['detection'].to_numpy()

        if len(sequence) == 0:
            return []    
        
        frame_intervals = []
        pre_walk = False
        tmp_cnt = 0
        
        pbar = tqdm(total=len(sequence)-window, unit=' frame')
        if video_index and video_tot: 
            pbar.set_description_str("Scanning intervals for signal {}, {} of {}".format(zone, video_index, video_tot))
        else:
            pbar.set_description_str("Scanning intervals for signal {}".format(zone))

        for i in range(len(sequence) - window):
            count = sum(sequence[i:i+window])

            # check if the current frame can be a start of green light
            if count >= factor * window:
                is_walking = True
            else:
                is_walking = False

            # if the current is green
            # 1) if prev status is green, update the latest interval
            # 2) if prev status is not green, append a new interval
            if is_walking:
                if not pre_walk:
                    frame_intervals.append([i, i + window])
                    tmp_cnt = 0
                else:
                    if count > tmp_cnt:
                        tmp_cnt = count
                        frame_intervals[-1] = [i, i + window]
            
            pre_walk = is_walking

            pbar.update()
        pbar.close()

        results = []
        for start, end in frame_intervals:
            results.append([zone, 1, int(dets['frame'].iloc[start]), int(dets['frame'].iloc[end])]) 
            results.append([zone, 2, int(dets['frame'].iloc[end])+1, int(dets['frame'].iloc[end]+int(10*fps))])
    
        df = pd.DataFrame(results, columns=['signal', 'status', 'beg_frame', 'end_frame'])
        return df

    def crop_zone(self, frame:np.ndarray)->list[np.ndarray]:
        '''
        Parameters:
        - frame: frame of the video

        Return:
        - list of cropped zones
        '''
        crop_regions = []
        for i, region in enumerate(self.det_zones):
            x, y, w, h = region
            cropped = frame[y:y + h, x:x + w]
            cropped = Image.fromarray(cropped)
            crop_regions.append(cropped)
        return crop_regions
    
    def predict(self, batch:list):
        """
        Parameters:
            batch: the cropped batch of images
        Returns:
            pred_array: the corresponding traffic signal predictions with format (zone0_pred, zone1_pred, ...)
        """
        self.model.eval()

        batchsz = len(batch)
        num_group = len(batch[0])

        # define the image transform
        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Transform all images and stack to different groups(corresponding light)
        transformed_crops = [[transform(image) for image in crop] for crop in batch]
        transformed_batch = [torch.stack([transformed_crops[j][i] for j in range(batchsz)]) for i in range(num_group)]

        # send to model and make predictions zone by zone
        batch_pred = []
        for group in transformed_batch:
            # with torch.no_grad():
            outputs = self.model(group.to(self.device))
            _, y_pred = torch.max(outputs, 1)
            batch_pred.append(y_pred.data.cpu().numpy())          

        pred_array = np.array(batch_pred).T
        return pred_array

    def generate_labels(self, signals:pd.DataFrame, input_video:str, label_file:str,
                        size_factor:float=1.5, thick:int=1, video_index:int=None, video_tot:int=None)->pd.DataFrame:

         # open input video
        cap = cv2.VideoCapture(input_video)
        if not cap.isOpened():
            print('Failed to open the video!')
        tot_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        pbar = tqdm(total= tot_frames)
        if video_index and video_tot:
            pbar.set_description_str("Generating signal labeles for {} of {}".format(video_index, video_tot))
        else:
            pbar.set_description_str("Generating signal labels")
        
        results = []
        for i in range(tot_frames):
            for j in range(len(self.det_zones)):
                status = 0      # no walking
                selected = signals[(signals['beg_frame']<=i) & (signals['end_frame']>=i) & (signals['signal']==j)]
                if len(selected) > 0:
                    status = selected['status'].iloc[0]

                x, y, w, h = self.det_zones[j]
                cx = int(x + w/2)
                cy = int(y + h/2)
                r = int(max(w, h) * size_factor)
    
                if status == 0:
                    results.append([i, 'circle', [(cx, cy)], (0, 0, 255), r, thick, ''])
                elif status == 1:
                    results.append([i, 'circle', [(cx, cy)], (0, 255, 0), r, thick, ''])
                elif status == 2:
                    results.append([i, 'circle', [(cx, cy)], (0, 255, 255), r, thick, ''])

                pbar.update()

        df = pd.DataFrame(results, columns=['frame','type','coords','color','size','thick','desc'])
        df.sort_values(by='frame')
        
        if label_file:
            df.to_csv(label_file, index=False)

        return df

