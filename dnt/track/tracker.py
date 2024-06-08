import os, sys
sys.path.append(os.path.dirname(__file__))

from dsort import track as track_dsort
from sort import track as track_sort
from botsort import track as track_botsort
from config import Config
import pandas as pd
from tqdm import tqdm
import numpy as np
from ..engine import interpolate_bbox, interpolate_bboxes, cluster_by_gap

class Tracker(object):
    def __init__(self, cfg: dict = None, gpu:bool=True):
        '''
        Parameters:
            cfg: dict - Configuration for tracking method
            gpu: bool - Use GPU for tracking
        '''
        if cfg is None:
            self.cfg = Config.get_cfg_dsort()
        else:
            self.cfg = cfg
        self.gpu = gpu 

    def track(self, det_file:str, out_file:str, video_file:str=None, video_index:int=None, total_videos:int=None):
        '''
        Parameters:
        - det_file:     Detection file path
        - out_file:     Output track file path
        - video_file:   Video file path
        - video_index:  Index of video in the batch
        - total_videos: Total number of videos in the batch
        '''

        if self.cfg['method'] == 'sort':
            track_sort(det_file, out_file, self.cfg['max_age'], self.cfg['min_inits'], 
                       self.cfg['iou_threshold'], video_index, total_videos)
        elif self.cfg['method'] == 'dsort':
            if video_file:
                track_dsort(video_file=video_file, det_file=det_file, out_file=out_file, gpu=self.gpu, cfg=self.cfg,
                            video_index=video_index, total_videos=total_videos)
            else:
                print('No video file exists!')
        elif self.cfg['method'] == 'botsort':
            track_botsort(video_file=video_file, det_file=det_file, out_file=out_file, cfg=self.cfg,
                          video_index=video_index, total_videos=total_videos)
        else:
            print('Invalid tracking method!')
    
    def track_batch(self, det_files=list[str], video_files=list[str], output_path:str=None, 
                    is_overwrite:bool=False, is_report:bool=True)->list[str]:
        results = []
        total_videos = len(det_files)
        count=0
        for det_file in det_files:
            count+=1

            base_filename = os.path.splitext(os.path.basename(det_file))[0].replace("_iou", "")
            if output_path:
                if not os.path.exists(output_path):
                    os.mkdir(output_path)
                track_file = os.path.join(output_path, base_filename+"_track.txt")

            if not is_overwrite:
                if os.path.exists(track_file):
                    if is_report:
                        results.append(track_file)    
                    continue 
            
            video_file = None
            if self.cfg['method']=="dsort":
                video_file = video_files[count-1]

            self.track(det_file=det_file, out_file=track_file, video_file=video_file, video_index=count, total_videos=total_videos)

            results.append(track_file)

        return results
        
    @staticmethod
    def export_track_header():
        return ['frame', 'track', 'x', 'y', 'w', 'h', 'score', 'cls', 'infill', 'cluster']
    
    @staticmethod
    def infill_frames(tracks:pd.DataFrame, ids:list[int] = None, method:str='cubic', inplace:bool=True, info:str='', 
                      video_index:int=None, video_tot:int=None, verbose:bool=True) -> pd.DataFrame:
        '''
        Paramters:
        - tracks: the track dataframe need to be fill
        - ids: track ids need to be fill, if None, all ids will be processed
        - method: interpolation method, 'cubic' (default), 'linear', 'nearest'
        - inplace: if combine filled and raw frames in the output, default is True.
        Return:
        - A dataframe contains infilled frames (inplace=False), or infilled+raw (inplace=True)
        '''
        results = []

        tracks.columns = Tracker.export_track_header()
        if ids is None:
            ids = tracks['track'].unique()
        
        pbar = tqdm(total=len(ids), unit=' tracks')
        if video_index and video_tot: 
            pbar.set_description_str("Infliing frames for {} tracks, {} of {}".format(info, video_index, video_tot))
        else:
            pbar.set_description_str("Inflling frames for {} tracks".format(info))
        for id in ids:
            frames = tracks[tracks['track']==id].copy().sort_values(by='frame')
            f_min = frames['frame'].min()
            f_max = frames['frame'].max()

            if (f_max-f_min+1) > len(frames):
                raw_fids = frames['frame'].values.tolist()
                missed_fids = Tracker.__find_missing_number(raw_fids)

                raw_bboxes = frames[['x', 'y', 'w', 'h']].to_numpy()
                infilled_bboxes = interpolate_bboxes(boxes = raw_bboxes, frames = np.array(raw_fids), target_frames=np.array(missed_fids), 
                                                     method=method)

                d = {'frame':missed_fids, 'track':id, 'x': infilled_bboxes[:,0], 'y':infilled_bboxes[:,1], 
                                          'w':infilled_bboxes[:,2], 'h':infilled_bboxes[:,3], 'score':frames['score'].iloc[0], 
                                          'cls':frames['cls'].iloc[0], 'infill':1, 'cluster':frames['cluster'].iloc[0]}
                df = pd.DataFrame(d)
                df = pd.concat([frames, df], ignore_index=True).sort_values(by='frame')
                results.append(df)

            if verbose:
                pbar.update()         

        pbar.close()

        if len(results)>0:
            df = pd.concat(results, ignore_index=True)
            if inplace:
                df = pd.concat([tracks, df], ignore_index=True).sort_values(by=['frame', 'track'])
            return df
        else:
            if inplace:
                return tracks
            else:
                return None
    
    @staticmethod
    def cluster_frames(tracks:pd.DataFrame, ids:list[int] = None, gap_thres:int=100, keep_thres:int=25, inplace:bool=True, 
                      video_index:int=None, video_tot:int=None, verbose:bool=True) -> pd.DataFrame:
        '''
        Paramters:
        - tracks: a dataframe of tracks need to be clustered by empty frames
        - ids: track ids need to be clustered, if None, all track ids
        - gap_thres: if empty frames > gap_thres, the track will be cut-off, default is 100
        - keep_thres: if the frames of a clustered track is >= keep_thres, the track will be kept, default is 25 
        - inplace: if combine filled and raw frames in the output, default is True.
        Return:
        - A dataframe contains infilled frames (inplace=False), or infilled+raw (inplace=True)
        '''
        results = []
        del_ids = []

        tracks.columns = Tracker.export_track_header()
        if ids is None:
            ids = tracks['track'].unique()
        tracks_grouped = tracks.groupby('track')

        id_index = max(ids)

        pbar = tqdm(total=len(ids), unit=' track')
        if video_index and video_tot: 
            pbar.set_description_str("Clustering frames {} of {}".format(video_index, video_tot))
        else:
            pbar.set_description_str("Clustering frames")
        for id in ids:
            #frames = tracks[tracks['track']==id].copy().sort_values(by='frame')
            frames = tracks_grouped.get_group(id)
            fids = frames['frame'].values
            clusters = cluster_by_gap(fids, gap_thres)

            if len(clusters) > 1:
                for cluster in clusters:
                    frame_length = max(cluster) - min(cluster) + 1
                    if frame_length >= keep_thres:
                        #d = tracks[(tracks['track']==id) & (tracks['frame'].isin(cluster))].copy()
                        d = frames[frames['frame'].isin(cluster)].copy()
                        id_index += 1
                        d['track'] = id_index
                        d['cluster'] = id
                        results.append(d)
                del_ids.append(id)

            if verbose:
                pbar.update()
    
        pbar.close()
        
        df = pd.concat(results, ignore_index=True)
        if inplace:
            tracks_del = tracks[~tracks['track'].isin(del_ids)].copy()
            df = pd.concat([tracks_del, df], ignore_index=True).sort_values(by=['frame', 'track'])

        return df
    
    @staticmethod
    def del_short_tracks(tracks:pd.DataFrame, keep_thres:int=25,
                      video_index:int=None, video_tot:int=None, verbose:bool=True) -> pd.DataFrame:
        '''
        Paramters:
        - tracks: a dataframe of tracks need to be clustered by empty frames
        - ids: track ids need to be clustered, if None, all track ids
        - gap_thres: if empty frames > gap_thres, the track will be cut-off, default is 100
        - keep_thres: if the frames of a clustered track is >= keep_thres, the track will be kept, default is 25 
        - inplace: if combine filled and raw frames in the output, default is True.

        Return:
        - A dataframe contains infilled frames (inplace=False), or infilled+raw (inplace=True)
        '''
        del_ids = []

        tracks.columns = Tracker.export_track_header()
        ids = tracks['track'].unique()
        tracks_grouped = tracks.groupby('track')

        pbar = tqdm(total=len(ids), unit=' track')
        if video_index and video_tot: 
            pbar.set_description_str("Scan short tracks {} of {}".format(video_index, video_tot))
        else:
            pbar.set_description_str("Scan short tracks")
        for id in ids:
            frames = tracks_grouped.get_group(id)
            #frames = tracks[tracks['track']==id].copy().sort_values(by='frame')
            frame_max = max(frames['frame'].values)
            frame_min = min(frames['frame'].values)
            frame_length = frame_max - frame_min + 1
            if frame_length < keep_thres:
                del_ids.append(id)

            if verbose:
                pbar.update()
        pbar.close()
        
        tracks_del = tracks[~tracks['track'].isin(del_ids)].copy()
        return tracks_del
    
    @staticmethod
    def __find_missing_number(arr: list[int]) -> list[int]:
        full_set = set(range(min(arr), max(arr) + 1))
        arr_set = set(arr)
        missing_numbers = list(full_set - arr_set)
        return missing_numbers
        

    
