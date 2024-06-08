import yaml
import os, io

class Config:
    def __init__(self, cfg:dict):
        pass

    @staticmethod
    def get_cfg_sort()->dict:
        '''
        method: str - Tracking method, 'sort'
        max_age: int - Maximum age of a track, default is 1
        min_inits: int - Number of frames to wait before activate a track, default is 3
        iou_threshold: float - IOU threshold, default is 0.3
        '''
        return {'method':'sort', 
                'max_age':1, 
                'min_inits':3, 
                'iou_threshold':0.3}

    @staticmethod
    def get_cfg_dsort(model:str="default")->dict:
        '''
        Parameters:
            model: str - DeepSORT model, 'default', 'bot_R50', 'bot_R50-ibn', 'bot_S50', 'bot_R101-ibn', 'veh_bot_R50-ibn'
        Contents:
            method: str - Tracking method, 'dsort'
            reid_cfg: str - ReID config file path
            reid_ckpt: str - ReID weights file path
            max_dist: float - Maximum cosine distance, default is 0.2
            min_confidence: float - Minimum detection confidence, default is 0.3
            nms_max_overlap: float - Maximum overlap in NMS, default is 0.5
            max_iou_distance: float - Maximum IOU distance, default is 0.7
            max_age: int - Maximum age of a track, default is 70
            n_init: int - Number of frames to wait before activate a track, default is 3
            nn_budget: int - Maximum number of nearest neighbors, default is 100
        '''
        cfg = { 'method': 'dsort',
                'reid_cfg': None,
                'reid_ckpt': 'ckpt.t7', 
                'max_dist': 0.2, 
                'min_confidence': 0.3,
                'nms_max_overlap': 0.5, 
                'max_iou_distance': 0.7, 
                'max_age': 70, 
                'n_init': 3, 
                'nn_budget': 100}

        if model == 'bot_R50':
            cfg['reid_cfg'] = 'Market1501/bagtricks_R50.yml'
            cfg['reid_ckpt'] = 'market_bot_R50.pth'
        elif model == 'bot_R50-ibn':
            cfg['reid_cfg'] = 'Market1501/bagtricks_R50-ibn.yml'
            cfg['reid_ckpt'] = 'market_bot_R50-ibn.pth'
        elif model == 'bot_S50':
            cfg['reid_cfg'] = 'Market1501/bagtricks_S50.yml'
            cfg['reid_ckpt'] = 'market_bot_S50.pth'
        elif model == 'bot_R101-ibn':
            cfg['reid_cfg'] = 'Market1501/bagtricks_R101-ibn.yml'
            cfg['reid_ckpt'] = 'market_bot_R101-ibn.pth'
        elif model == 'veh_bot_R50-ibn':
            cfg['reid_cfg'] = 'VehicleID/bagtricks_R50-ibn.yml'
            cfg['reid_ckpt'] = 'vehicleid_bot_R50-ibn.pth'
            
        return cfg

    @staticmethod
    def get_cfg_botsort()->dict:
        '''
        method: str - Tracking method, 'botsort'
        device: str - Calucation device, default is 'cuda'
        half: bool - Half precision, default is False
        track_high_thresh: float - Tracking confidence threshold, default is 0.5
        track_low_thresh: float - Lowest detection threshold, default is 0.1
        new_track_thresh: float - New track thresh, default is 0.6
        track_buffer: int - The frames for keep lost tracks, default is 30
        fps: int - Video fps, default is 30
        match_thresh: float - Matching threshold for tracking, default is 0.8
        aspect_ratio_thresh: float - Threshold for filtering out boxes of which aspect ratio are above the given value, default is 1.6
        min_box_area: int - Filter out tiny boxes, default is 10
        fuse_score: bool - Fuse score and iou for association, default is False
        mot20: str - MOT20 dataset, default is None
        cmc_method: str - CMC method, 'sparseOptFlow', 'orb', 'ecc', default is 'sparseOptFlow'
        ablation: bool - Ablation, default is False
        name: str - Benchmark name, default is 'None'
        with_reid: bool - With ReID module, default is True
        fast_reid_config: str - ReID config file path, default is 'Market1501/sbs_S50.yml'
        fast_reid_weights: str - ReID config file path, default is 'market_bot_S50.pth'
        proximity_thresh: float - Threshold for rejecting low overlap reid matches, default is 0.5
        appearance_thresh: float - Threshold for rejecting low appearance similarity reid matches, default is 0.25
        '''
        return { 'method': 'botsort',
                'device': 'cuda',               # calucation device: cpu, cuda
                'half': False,                  # half precision: True, False
                'output_header': False,         # add header in the output file

                 # Track
                'track_high_thresh': 0.5,       # tracking confidence threshold
                'track_low_thresh': 0.1,        # lowest detection threshold
                'new_track_thresh': 0.6,        # new track thresh
                'track_buffer': 30,             # the frames for keep lost tracks
                'fps': 30,                      # video fps
                'match_thresh': 0.8,            # matching threshold for tracking
                'aspect_ratio_thresh': 1.6,     # threshold for filtering out boxes of which aspect ratio are above the given value.
                'min_box_area': 10,             # filter out tiny boxes
                'fuse_score': False,            # fuse score and iou for association
                'mot20': None,                  # mot20 dataset
                
                # CMC
                'cmc_method': 'sparseOptFlow',  # cmc method: files (Vidstab GMC) | sparseOptFlow |orb | ecc
                'ablation': False,              # ablation
                'name': 'None',                 # benchmark name
             
                # ReID  
                'with_reid': True,                             # with ReID module.
                'fast_reid_config': 'Market1501/sbs_S50.yml',        # reid config file path
                'fast_reid_weights': 'market_bot_S50.pth',       # reid config file path
                'proximity_thresh': 0.5,                        # threshold for rejecting low overlap reid matches
                'appearance_thresh': 0.25                       # threshold for rejecting low appearance similarity reid matches
            }