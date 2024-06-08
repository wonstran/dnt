from shapely.geometry import Point, Polygon, LineString, box
from shapely import intersects
import geopandas as gpd, pandas as pd
import datetime
from tqdm import tqdm
from cython_bbox import bbox_overlaps
import numpy as np
#import mapply
from ..filter import Filter
from matplotlib import pyplot as plt
import random, os
from ..label.labeler2 import Labeler
from ..engine.bbox_iou import ious

class StopAnalyzer():
    def __init__(self, 
                 stop_zones:list[Polygon], 
                 lane_zones:list[Polygon],
                 event_dicts:list[dict], 

                 stop_iou:float=0.97,
                 iou_mode:str='edge', 
                 frame_buffer:int=5,
                 
                 stop_zone_mode:str='iob',
                 bbox_iob:float=0.05,
                 bbox_offset:tuple=(0, 0, 0, 0),

                 lane_zone_ref:str = 'br',               
                 ref_offset:tuple=(0, 0),
                 lane_adjust:bool=False,

                 leading_axis:str='x',
                 leading_direct:str='+', 
                 leading_buffer:int = 30,
                 verbose:bool=True):
        
        '''
        Parameters:
            stop_zones: A list of Polygons for identifying stopping position, the priority of zones is ascending
            lane_zones: A list of Polygons for identifying lanes
            event_dicts: A list of dicts to define event code, event description, and associated stop zone
            stop_iou: The threshold for stopping, stop if iou > the threshold
            iou_mode: The mode to judge stop, default is 'edge' [the differenc between two ends], 'mean', 'max'
            frame_buffer: the number of consecutive frames for stopping identification
            stop_zone_mode: default is 'iob' - intersection over bbox, 'intersect' - if bbox and stop zone intersected
            bbox_iob: the threshold for iob, default is 0.05
            bbox_offset: the offset (x1, y1, x2, y2) for bbox, default is (0, 0, 0, 0)
            lane_zone_ref: the reference point for lane recognition
                        br (buttom-right, default), bl (bottom-left), bc (bottom-center)
                        tl (top-left), tr (top-right), tc (top-center)
                        cc (center-center), cl (center-left), cr (center-right)
            ref_offset: the offset (x, y) for reference point, default is (0, 0)
            lane_adjust: If adjust lane zone for the missed frames (lane_zone=-1) based on the most frequent lane zone, default is True
            leading_axis: the axis of stop zone indexes to identify leading vehicles (vehicle travel direction)
                          'x' (default), 'y', 'xy'
            leading_direct: leading direction - '+' (default) increased, '-' descreased, '+-', increased in x and decreased in y              
            verbsoe: Display processing bars, default is True
        '''
        self.hzones = stop_zones
        self.vzones = lane_zones
        self.event_dicts = event_dicts
        self.stop_iou = stop_iou
        self.iou_mode = iou_mode
        self.frame_buffer = frame_buffer
        self.ref_point = lane_zone_ref
        self.stop_zone_mode = stop_zone_mode
        self.stop_iob = bbox_iob
        self.hzone_offset = bbox_offset
        self.vzone_offset = ref_offset
        self.vzone_adjust = lane_adjust
        self.leading_axis = leading_axis
        self.leading_direct = leading_direct
        self.leading_buffer = leading_buffer
        self.verbose = verbose
    
    def analysis_first_stop(self, tracks:pd.DataFrame, video_index:int=None, video_tot:int=None):
        '''
        Inputs:
            tracks: vehicle tracks
            video_index: index of video
            video_tot: total number of videos
        Outputs:
            tracks: vehicle tracks with stop analysis results
            events: event count
        '''
        tracks = StopAnalyzer.add_field_names(tracks)
        results = self.scan_stop(tracks)
        results = self.scan_zones(results)
        results = self.scan_leading_at_first_stop(results)
        results = self.scan_first_stop_event(results)
        events = self.count_event(results)

        return results, events
    
    def scan_stop(self, tracks:pd.DataFrame, video_index:int=None, video_tot:int=None)->pd.DataFrame:
        
        tracks = StopAnalyzer.add_field_names(tracks)
        ids = tracks['track'].unique()
        tracks['x2'] = tracks.apply(lambda t: t['x'] + t['w'], axis=1)
        tracks['y2'] = tracks.apply(lambda t: t['y'] + t['h'], axis=1)       
        tracks = tracks.sort_values(by='frame') # sort by frame in ascending
        
        pbar = tqdm(total=len(ids), unit=' tracks')
        if video_index and video_tot:
            pbar.set_description_str("Scanning stops {} of {}".format(video_index, video_tot))
        else:
            pbar.set_description_str("Scanning stops")

        for id in ids:
            track = tracks[tracks['track'] == id]
            
            # exclude the tracks length < frame_buffer
            if [len(track)>=self.frame_buffer]:
                bb = track[['x', 'y', 'x2', 'y2']].to_numpy(dtype=np.int32)

                if self.iou_mode == 'edge':
                    bb_buf = np.vstack((np.array([-2 , -2, -1, -1]*self.frame_buffer).reshape(self.frame_buffer, 4), bb)) # add buff frame
                    #iou_scores = StopAnalyzer.ious(bb, bb_buf)
                    iou_scores = ious(bb, bb_buf)                            
                    tracks.loc[tracks['track']==id, 'iou'] = np.diagonal(iou_scores)
                
                elif self.iou_mode == 'max':
                    bb_buf = np.vstack((np.array([-2 , -2, -1, -1]*self.frame_buffer).reshape(self.frame_buffer, 4), bb)) # add buff frame
                    #iou_scores = StopAnalyzer.ious(bb, bb_buf)
                    iou_scores = ious(bb, bb_buf)
                    nrows, ncols =np.shape(bb)
                    iou_arr = np.empty([nrows, self.frame_buffer])
                    for i in range(nrows):
                        iou_arr[i, :] = iou_scores[i, i:i+self.frame_buffer]
                    tracks.loc[tracks['track']==id, 'iou'] = np.amax(iou_arr, axis=1)
                
                elif self.iou_mode == 'mean':
                    #iou_scores = StopAnalyzer.ious(bb, bb)
                    iou_scores = ious(bb, bb)
                    nrows, ncols =np.shape(bb)
                    iou_arr = np.zeros([nrows])
                    for i in range(1, nrows):
                        beg = max(i-self.frame_buffer, 0)
                        end = max(i-1, 0) + 1
                        iou_arr[i] = np.mean(iou_scores[i, beg:end])

                    tracks.loc[tracks['track']==id, 'iou'] = iou_arr

            if self.verbose:
                pbar.update()
        
        pbar.close()

        if video_index and video_tot:
            tqdm.pandas(desc='Updating stops {} of {}'.format(video_index, video_tot), unit='frames')
        else:
            tqdm.pandas(desc='Updating stops '.format(video_index, video_tot), unit='frames')
        tracks['stop'] = tracks['iou'].progress_apply(lambda iou: 1 if iou>=self.stop_iou else -1)
        tracks = tracks.drop(columns=['x2', 'y2'])

        return tracks
    
    def scan_zones(self, tracks:pd.DataFrame, video_index:int=None, video_tot:int=None):
        
        tracks = StopAnalyzer.add_field_names(tracks)
        tracks['hzone'] = -1

        if self.hzones:

            if video_index and video_tot:
                tqdm.pandas(desc='Generating bbox {} of {}'.format(video_index, video_tot), unit='frames')
            else:
                tqdm.pandas(desc='Generating bbox '.format(video_index, video_tot), unit='frames')           
            g = tracks.progress_apply(lambda track: box(track['x'] + self.hzone_offset[0], 
                                                        track['y'] + self.hzone_offset[1], 
                                                        track['x'] + self.hzone_offset[2] + track['w'], 
                                                        track['y'] + self.hzone_offset[3] + track['h']), 
                                                        axis=1)
            geo_tracks = gpd.GeoDataFrame(tracks, geometry=g)

            pbar = tqdm(total=len(self.hzones), unit= 'zone')
            if video_index and video_tot:
                pbar.set_description_str("Scanning stop-zones {} of {}".format(video_index, video_tot))
            else:
                pbar.set_description_str("Scanning stop-zones")
            
            zones = range(len(self.hzones))

            for i in zones:
                if self.stop_zone_mode == 'intersect':
                    geo_tracks.loc[((geo_tracks.geometry.intersects(self.hzones[i])) & (geo_tracks['hzone']<i)), 'hzone'] = i
                else:
                    geo_tracks['iob'] = geo_tracks.geometry.apply(lambda bbox: 
                                                                  (bbox.intersection(self.hzones[i]).area)/(bbox.area) if bbox.area>0 else -1)
                    geo_tracks.loc[((geo_tracks['iob']>=self.stop_iob) & (geo_tracks['hzone']<i)), 'hzone'] = i

                pbar.update()
            
            pbar.close()

            geo_tracks=geo_tracks[geo_tracks['hzone']>-1]
            tracks = pd.DataFrame(geo_tracks.drop(columns='geometry'))
        
        if self.vzones:
            if video_index and video_tot:
                tqdm.pandas(desc='Generating reference point {} of {}'.format(video_index, video_tot), unit='frames')
            else:
                tqdm.pandas(desc='Generating reference point '.format(video_index, video_tot), unit='frames') 
            if self.ref_point == 'cc':
                g = tracks.progress_apply(lambda track: Point([track['x'] + track['w']/2 + self.vzone_offset[0], 
                                                              track['y'] + track['h']/2 + self.vzone_offset[1]]), axis=1)
            elif self.ref_point == 'tc':
                g = tracks.progress_apply(lambda track: Point([track['x'] + track['w']/2 + self.vzone_offset[0], 
                                                              track['y'] + self.vzone_offset[1]]), axis=1)
            elif self.ref_point == 'bc':
                g = tracks.progress_apply(lambda track: Point([[track['x'] + track['w']/2 + self.vzone_offset[0], 
                                                              track['y'] + track['h'] + self.vzone_offset[1]]]), axis=1)
            elif self.ref_point == 'cl':
                g = tracks.progress_apply(lambda track: Point([[track['x'] + self.vzone_offset[0], 
                                                              track['y'] + track['h']/2 + self.vzone_offset[1]]]), axis=1)
            elif self.ref_point == 'cr':
                g = tracks.progress_apply(lambda track: Point([track['x'] + track['w'] + self.vzone_offset[0], 
                                                              track['y'] + track['h']/2 + self.vzone_offset[1]]), axis=1)
            elif self.ref_point == 'tl':
                g = tracks.progress_apply(lambda track: Point([track['x'] + self.vzone_offset[0], 
                                                              track['y'] + self.vzone_offset[1]]), axis=1)
            elif self.ref_point == 'tr':
                g = tracks.progress_apply(lambda track: Point([track['x'] + track['w'] + self.vzone_offset[0], 
                                                              track['y'] + self.vzone_offset[1]]), axis=1)
            elif self.ref_point == 'bl':
                g = tracks.progress_apply(lambda track: Point([track['x'] + self.vzone_offset[0], 
                                                              track['y'] + tracks['h'] + self.vzone_offset[1]]), axis=1)
            elif self.ref_point == 'br':
                g = tracks.progress_apply(lambda track: Point([track['x'] + track['w'] + self.vzone_offset[0], 
                                                              track['y'] + track['h'] + self.vzone_offset[1]]), axis=1)    
            else:
                g = tracks.progress_apply(lambda track: Point([track['x'] + track['w']/2 + self.vzone_offset[0], 
                                                              track['y'] + track['h'] + self.vzone_offset[1]]), axis=1) 

            geo_tracks = gpd.GeoDataFrame(tracks, geometry=g)

            pbar = tqdm(total=len(self.vzones), unit=' zone')
            if video_index and video_tot: 
                pbar.set_description_str("Scanning lane-zones {} of {}".format(video_index, video_tot))
            else:
                pbar.set_description_str("Scanning lane-zones ")
            for i in range(len(self.vzones)):
                geo_tracks.loc[(geo_tracks.geometry.within(self.vzones[i])), 'vzone'] = i
                pbar.update()
            pbar.close()

            if self.vzone_adjust:
                ids = geo_tracks['track'].unique()
                pbar = tqdm(total=len(ids), unit=' tracks')
                if video_index and video_tot: 
                    pbar.set_description_str("Adjusting lane-zones {} of {}".format(video_index, video_tot))
                else:
                    pbar.set_description_str("Adjusting lane-zones ")
                for id in ids:
                    located_frames = geo_tracks[(geo_tracks['track']==id) & (geo_tracks['vzone']>-1)]
                    if len(located_frames)>0:
                        geo_tracks.loc[geo_tracks['track']==id, 'vzone'] = located_frames['vzone'].mode()[0]

                    pbar.update()
                pbar.close()

            tracks = pd.DataFrame(geo_tracks.drop(columns='geometry'))

        return tracks        

    def scan_zones_v2(self, tracks:pd.DataFrame, video_index:int=None, video_tot:int=None):
        
        tracks = StopAnalyzer.add_field_names(tracks)

        if self.hzones:

            if video_index and video_tot:
                tqdm.pandas(desc='Generating bbox {} of {}'.format(video_index, video_tot), unit='frames')
            else:
                tqdm.pandas(desc='Generating bbox '.format(video_index, video_tot), unit='frames')           
            g = tracks.progress_apply(lambda track: box(track['x'] + self.hzone_offset[0] + track['w']*(1-self.hzone_factor)/2, 
                                                        track['y'] + self.hzone_offset[1] + track['h']*(1-self.hzone_factor)/2, 
                                                        track['x'] + self.hzone_offset[2] + track['w'] - track['w']*(1-self.hzone_factor)/2, 
                                                        track['y'] + self.hzone_offset[3] + track['h'] - track['h']*(1-self.hzone_factor)/2), 
                                                        axis=1)
            geo_tracks = gpd.GeoDataFrame(tracks, geometry=g)

            pbar = tqdm(total=len(self.hzones), unit= 'zone')
            if video_index and video_tot:
                pbar.set_description_str("Scanning stop-zones {} of {}".format(video_index, video_tot))
            else:
                pbar.set_description_str("Scanning stop-zones")
            
            zones = range(len(self.hzones))

            for i in zones:
                # geo_tracks.loc[((geo_tracks.geometry.intersects(self.hzones[i])) & (geo_tracks['hzone']<i)), 'hzone'] = i
                geo_tracks['iob'] = geo_tracks.geometry.apply(lambda bbox: (bbox.intersection(self.hzones[i]).area)/(bbox.area) if bbox.area>0 else -1)
                geo_tracks.loc[((geo_tracks['iob']>=self.stop_iob) & (geo_tracks['hzone']<i)), 'hzone'] = i
                pbar.update()
            
            pbar.close()

            geo_tracks=geo_tracks[geo_tracks['hzone']>-1]
            tracks = pd.DataFrame(geo_tracks.drop(columns='geometry'))

        if self.vzones:
            if video_index and video_tot:
                tqdm.pandas(desc='Generating reference point {} of {}'.format(video_index, video_tot), unit='frames')
            else:
                tqdm.pandas(desc='Generating reference point '.format(video_index, video_tot), unit='frames') 
            if self.ref_point == 'cc':
                g = tracks.progress_apply(lambda track: Point([track['x'] + track['w']/2 + self.vzone_offset[0], 
                                                              track['y'] + track['h']/2 + self.vzone_offset[1]]), axis=1)
            elif self.ref_point == 'tc':
                g = tracks.progress_apply(lambda track: Point([track['x'] + track['w']/2 + self.vzone_offset[0], 
                                                              track['y'] + self.vzone_offset[1]]), axis=1)
            elif self.ref_point == 'bc':
                g = tracks.progress_apply(lambda track: Point([[track['x'] + track['w']/2 + self.vzone_offset[0], 
                                                              track['y'] + track['h'] + self.vzone_offset[1]]]), axis=1)
            elif self.ref_point == 'cl':
                g = tracks.progress_apply(lambda track: Point([[track['x'] + self.vzone_offset[0], 
                                                              track['y'] + track['h']/2 + self.vzone_offset[1]]]), axis=1)
            elif self.ref_point == 'cr':
                g = tracks.progress_apply(lambda track: Point([track['x'] + track['w'] + self.vzone_offset[0], 
                                                              track['y'] + track['h']/2 + self.vzone_offset[1]]), axis=1)
            elif self.ref_point == 'tl':
                g = tracks.progress_apply(lambda track: Point([track['x'] + self.vzone_offset[0], 
                                                              track['y'] + self.vzone_offset[1]]), axis=1)
            elif self.ref_point == 'tr':
                g = tracks.progress_apply(lambda track: Point([track['x'] + track['w'] + self.vzone_offset[0], 
                                                              track['y'] + self.vzone_offset[1]]), axis=1)
            elif self.ref_point == 'bl':
                g = tracks.progress_apply(lambda track: Point([track['x'] + self.vzone_offset[0], 
                                                              track['y'] + tracks['h'] + self.vzone_offset[1]]), axis=1)
            elif self.ref_point == 'br':
                g = tracks.progress_apply(lambda track: Point([track['x'] + track['w'] + self.vzone_offset[0], 
                                                              track['y'] + track['h'] + self.vzone_offset[1]]), axis=1)    
            else:
                g = tracks.progress_apply(lambda track: Point([track['x'] + track['w']/2 + self.vzone_offset[0], 
                                                              track['y'] + track['h'] + self.vzone_offset[1]]), axis=1) 

            geo_tracks = gpd.GeoDataFrame(tracks, geometry=g)

            pbar = tqdm(total=len(self.vzones), unit=' zone')
            if video_index and video_tot: 
                pbar.set_description_str("Scanning lane-zones {} of {}".format(video_index, video_tot))
            else:
                pbar.set_description_str("Scanning lane-zones ")
            for i in range(len(self.vzones)):
                geo_tracks.loc[(geo_tracks.geometry.within(self.vzones[i])), 'vzone'] = i
                pbar.update()
            pbar.close()

            if self.vzone_adjust:
                ids = geo_tracks['track'].unique()
                pbar = tqdm(total=len(ids), unit=' tracks')
                if video_index and video_tot: 
                    pbar.set_description_str("Adjusting lane-zones {} of {}".format(video_index, video_tot))
                else:
                    pbar.set_description_str("Adjusting lane-zones ")
                for id in ids:
                    located_frames = geo_tracks[(geo_tracks['track']==id) & (geo_tracks['vzone']>-1)]
                    if len(located_frames)>0:
                        geo_tracks.loc[geo_tracks['track']==id, 'vzone'] = located_frames['vzone'].mode()[0]

                    pbar.update()
                pbar.close()

            tracks = pd.DataFrame(geo_tracks.drop(columns='geometry'))

        return tracks        
    
    def scan_leading_at_first_stop(self, tracks:pd.DataFrame, video_index:int=None, video_tot:int=None) -> pd.DataFrame:

        tracks = StopAnalyzer.add_field_names(tracks)
        tracks['xc'] = tracks.apply(lambda track: track['x'] + track['w']/2, axis=1)
        tracks['yc'] = tracks.apply(lambda track: track['y'] + track['h']/2, axis=1)
        
        stop_tracks = tracks[tracks['stop']==1]
        stop_ids = stop_tracks['track'].unique()

        pbar = tqdm(total=len(stop_ids), unit=' tracks')
        if video_index and video_tot:
            pbar.set_description_str("Scanning leading vehicles {} of {}".format(video_index, video_tot))
        else:
            pbar.set_description_str("Scanning leading vehicles")
        for stop_id in stop_ids:
            track = stop_tracks[stop_tracks['track']==stop_id].sort_values(by='frame').iloc[0]
            vzone = track['vzone']
            hzone = track['hzone']
            min_frame = track['frame'] - self.leading_buffer
            max_frame = track['frame'] + self.leading_buffer
            xc = track['xc']
            yc = track['yc']

            if (self.leading_axis=='x') and (self.leading_direct=='+'): 
                veh_front = tracks[(tracks['frame']>=min_frame) & (tracks['frame']<=max_frame) & (tracks['track']!=stop_id) 
                                   & (tracks['vzone']==vzone) & (tracks['xc']>=xc) & (tracks['stop']==1)]
            elif (self.leading_axis=='x') and (self.leading_direct=='-'):
                veh_front = tracks[(tracks['frame']>=min_frame) & (tracks['frame']<=max_frame) & (tracks['track']!=stop_id)  
                                   & (tracks['vzone']==vzone) & (tracks['xc']<=xc) & (tracks['stop']==1)]
            elif (self.leading_axis=='y') and (self.leading_direct=='+'):
                veh_front = tracks[(tracks['frame']>=min_frame) & (tracks['frame']<=max_frame) & (tracks['track']!=stop_id) 
                                   & (tracks['vzone']==vzone) & (tracks['yc']>=yc) & (tracks['stop']==1)]
            elif (self.leading_axis=='y') and (self.leading_direct=='-'):
                veh_front = tracks[(tracks['frame']>=min_frame) & (tracks['frame']<=max_frame) & (tracks['track']!=stop_id)  
                                   & (tracks['vzone']==vzone) & (tracks['yc']<=yc) & (tracks['stop']==1)]
            elif (self.leading_axis=='xy') and (self.leading_direct=='+'):
                veh_front = tracks[(tracks['frame']>=min_frame) & (tracks['frame']<=max_frame) & (tracks['track']!=stop_id) 
                                   & (tracks['vzone']==vzone) & (tracks['xc']>=xc) & (tracks['yc']>=yc) & (tracks['stop']==1)]
            elif (self.leading_axis=='xy') and (self.leading_direct=='-'):
                veh_front = tracks[(tracks['frame']>=min_frame) & (tracks['frame']<=max_frame) & (tracks['track']!=stop_id) 
                                   & (tracks['vzone']==vzone) & (tracks['xc']<=xc) & (tracks['yc']<=yc) & (tracks['stop']==1)]
            else:
                print('Leading setting is invalid and may cause incorrect results!')
                veh_front = []

            if len(veh_front) == 0:
                tracks.loc[(tracks['track']==stop_id), 'leading'] = 1
            else:
                tracks.loc[(tracks['track']==stop_id), 'leading'] = -1
            
            pbar.update()
        pbar.close()

        tracks = tracks.drop(columns=['xc', 'yc'])
        return tracks
        
    def scan_first_stop_event(self, tracks:pd.DataFrame, video_index:int=None, video_tot:int=None):

        tracks = StopAnalyzer.add_field_names(tracks)

        if self.vzones:
            leading_tracks = tracks[(tracks['leading']==1) & (tracks['hzone']>-1) & (tracks['vzone']>-1) & (tracks['stop']==1)]
        else:
            leading_tracks = tracks[(tracks['leading']==1) & (tracks['hzone']>-1) & (tracks['stop']==1)]
        leading_ids = leading_tracks['track'].unique()
        pbar = tqdm(total=len(leading_ids), unit=' tracks')
        
        if video_index and video_tot:
            pbar.set_description_str("Scanning first stop events {} of {}".format(video_index, video_tot))
        else:
            pbar.set_description_str("Scanning first stop events")

        for leading_id in leading_ids:
            frame = leading_tracks[leading_tracks['track']==leading_id].sort_values(by='frame').iloc[0]
            hzone = int(frame['hzone'])
            event_code = next(item for item in self.event_dicts if item["zone"] == hzone)['code']
            tracks.loc[tracks['track']==leading_id, 'event'] = event_code

            pbar.update()
        
        pbar.close()

        return tracks

    def count_event(self, tracks:pd.DataFrame, video_index:int=None, video_tot:int=None)->pd.DataFrame:

        tracks = StopAnalyzer.add_field_names(tracks)

        pbar = tqdm(unit='events')
        results = []
        for event in self.event_dicts:
            
            vehicles = tracks[tracks['event']==event['code']]['track'].unique()
            if len(vehicles)>0:
                if video_index and video_tot:
                    pbar.set_description_str("Counting event {} for {} of {}".format(event['desc'], video_index, video_tot))
                else:
                    pbar.set_description_str("Counting event {}".format(event['desc']))
            
                pbar.total = len(vehicles)
                for vehicle in vehicles:
                    track = tracks[(tracks['track'] == vehicle)]
                    start_frame = int(track['frame'].min())
                    end_frame = int(track['frame'].max())
                    vzones = track['vzone'].values.tolist()
                    vzones = [z for z in vzones if z>-1]
                    if len(vzones)>0 :
                        vzone = max(set(vzones), key = vzones.count)
                    else:
                        vzone = -1
                    results.append([event['code'], event['desc'], vehicle, vzone, start_frame, end_frame])
                    
                    if self.verbose:
                        pbar.update()

        results = pd.DataFrame(results, columns=['event', 'desc', 'track', 'vzone', 'start_frame', 'end_frame'])
        pbar.close()

        return results
    
    def generate_labels(self, tracks:pd.DataFrame, events:pd.DataFrame, 
                        method:str='all', random_number:int=10, event_codes:list=None, track_ids:list=None,
                        size:int=1, thick:int=1, show_track:bool=False, show_desc:bool=False, show_code:bool=False,
                        video_index:int=None, video_tot:int=None)->pd.DataFrame:
        '''
        Inputs:
            tracks: tracks with stop analysis results
            events: stop events file
            method: 'all' (default), 'random', 'event', 'track'
            random_number: if method=='random', the number of tracks for labelling, default is 10
            event_codes: if method=='event', the list of event codes for labelling, default is None for all event codes
            track_ids: if method=='tracks', the list of track ids for labelling, default is not for all track ids
        Return:
            a dataframe of labelling events: ['frame','type','coords','color','size','thick','desc']
        '''
        
        event_tracks = tracks.loc[tracks['event']>-1].copy()
        if method == 'random':
            track_ids = event_tracks['track'].unique().tolist()
            if random_number<=0:
                random_number = 10
            random_ids = random.sample(track_ids, random_number)
            event_tracks = event_tracks[event_tracks['track'].isin(random_ids)].copy()
        elif method == 'event':
            if (event_codes is None) or (len(event_codes)==0):
                print('No event codes are provided!')
                return pd.DataFrame()
            event_tracks = event_tracks[event_tracks['event'].isin(event_codes)].copy()
        elif method == 'track':
            if (track_ids is None) or (len(track_ids)==0):
                print('No tracks are provided!')
                return pd.DataFrame()
            event_tracks = event_tracks[event_tracks['track'].isin(track_ids)].copy()

        event_tracks['desc'] = ''
        event_tracks['color'] = None
        for index, event in events.iterrows():
            track_id = event['track']
            start_frame = event['start_frame']
            end_frame = event['end_frame']
            event_id = event['event']
            event_desc = event['desc']
            event_dict = next(item for item in self.event_dicts if item["code"] == event_id)

            event_tracks.loc[((event_tracks['track']==track_id) & (event_tracks['frame']>=start_frame) & 
                             (event_tracks['frame']<=end_frame)), ['event', 'desc']] = [event_id, event_desc]

        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

        pbar = tqdm(total= len(event_tracks))        
        if video_index and video_tot:
            pbar.set_description_str("Generating labeles for {} of {}".format(video_index, video_tot))
        else:
            pbar.set_description_str("Generating labels")

        results = []
        for index, track in event_tracks.iterrows():
            event_dict = next(item for item in self.event_dicts if item["code"] == track['event'])
            if 'color' in event_dict:
                event_color = event_dict['color']
            else:
                color = colors[int(track['event']) % len(colors)]
                event_color = [int(i * 255) for i in color]

            label_text = ''
            if show_track:
                label_text+='| T:'+str(int(track['track']))
                
            if show_code:
                label_text+='| C:'+str(int(track['event']))

            if show_desc:
                label_text+='| '+str(track['desc'])

            if label_text: 
                results.append([track['frame'], 'bbox', [(track['x'], track['y']), (track['x']+track['w'], track['y']+track['h'])], 
                            event_color, size, thick, label_text])
            else:
                results.append([track['frame'], 'box', [(track['x'], track['y']), (track['x']+track['w'], track['y']+track['h'])], 
                            event_color, size, thick, label_text])
            
            if self.verbose:
                pbar.update()

        df = pd.DataFrame(results, columns=['frame','type','coords','color','size','thick','desc'])
        df.sort_values(by='frame')
        
        return df
    
    @staticmethod
    def generate_label_clips(events:pd.DataFrame, tracks:pd.DataFrame, out_path:str,
                        method:str='all', random_number:int=10, event_codes:list=None, track_ids:list=None,
                        event_dicts:list=None, input_video:str=None, size:int=1, thick:int=2, padding:int=0, 
                        show_track:bool=False, show_code:bool=False, show_desc:bool=False, verbose:bool=True,
                        video_index:int=None, video_tot:int=None)->list:
        '''
        Parameters:
        - events: the dataframe of stop events
        - tracks: the dataframe of tracks
        - out_path: the folder for outputing track clips
        - method: 'all' (default) - all tracks, 'random' - random select tracks, 'specify' - specify track ids, 'event' - by event type
        - random_number: the number of track ids if method == 'random'
        - event_codes: the list event code if method == 'event'
        - track_ids: the list of track ids if method == 'specify'
        - event_dicts: the list of event dictionaries
        - input_video: the raw video file, if None, generate the label files only
        - size: font size, default is 1
        - thick: line thinckness, defualt is 2
        - padding: add addtional frames at beggining and ending, default is 0
        - show_track: show track id, default is False
        - show_code: show event code, default is False
        - show_desc: show event desc, default is Falase
        - verbose: if show progressing bar, default is True
        - video_idex: the index of the video in processing
        - video_tot: the total number of videos

        Return:
        - A list of dataframes for labels, ['frame','type','coords','color','size','thick','desc']
        '''

        event_tracks = tracks.loc[tracks['event']>-1].copy()
        if method == 'random':
            track_ids = event_tracks['track'].unique().tolist()
            if random_number<=0:
                random_number = 10
            random_ids = random.sample(track_ids, random_number)
            event_tracks = event_tracks[event_tracks['track'].isin(random_ids)].copy()
        elif method == 'event':
            if (event_codes is None) or (len(event_codes)==0):
                print('No event codes are provided!')
                return pd.DataFrame()
            event_tracks = event_tracks[event_tracks['event'].isin(event_codes)].copy()
        elif method == 'track':
            if (track_ids is None) or (len(track_ids)==0):
                print('No tracks are provided!')
                return pd.DataFrame()
            event_tracks = event_tracks[event_tracks['track'].isin(track_ids)].copy()
        
        event_tracks['desc'] = ''
        event_tracks['color'] = None
        for index, event in events.iterrows():
            track_id = event['track']
            start_frame = event['start_frame']
            end_frame = event['end_frame']
            event_id = event['event']
            event_desc = event['desc']
            event_dict = next(item for item in event_dicts if item["code"] == event_id)
            event_tracks.loc[((event_tracks['track']==track_id) & (event_tracks['frame']>=start_frame) & 
                             (event_tracks['frame']<=end_frame)), ['event', 'desc']] = [event_id, event_desc]
            
        ids = event_tracks['track'].unique().tolist()
        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

        pbar = tqdm(total= len(ids) , unit='tracks')        
        if video_index and video_tot:
            pbar.set_description_str("Generating labeles for {} of {}".format(video_index, video_tot))
        else:
            pbar.set_description_str("Generating labels")

        labeler = Labeler()

        results = []
        for id in ids:
            result = []
            selected_tracks = event_tracks[event_tracks['track']==id]
            for index, track in selected_tracks.iterrows():
                event_dict = next(item for item in event_dicts if item["code"] == track['event'])
                if 'color' in event_dict:
                    event_color = event_dict['color']
                else:
                    color = colors[int(track['event']) % len(colors)]
                    event_color = [int(i * 255) for i in color]

                label_text = ''
                if show_track:
                    label_text+='| T:'+str(int(track['track']))
                
                if show_code:
                    label_text+='| C:'+str(int(track['event']))

                if show_desc:
                    label_text+='| '+str(track['desc'])

                result.append([track['frame'], 'bbox', [(track['x'], track['y']), (track['x']+track['w'], track['y']+track['h'])], 
                            event_color, size, thick, label_text])
            
            df = pd.DataFrame(result, columns=['frame','type','coords','color','size','thick','desc'])

            if out_path:
                file_name = os.path.join(out_path, str(id)+'_label.csv')
                df.to_csv(file_name, index=False)

                if input_video:

                    file_name = os.path.join(out_path, str(id)+'_label.mp4')
                    min_frame = df['frame'].min() - padding
                    max_frame = df['frame'].max() + padding
                    labeler.draw(input_video=input_video, output_video=file_name, draws=df, 
                                start_frame=min_frame, end_frame=max_frame, verbose=False)
                
            results.append(df)

            if verbose:
                pbar.update()    

        return results

    @staticmethod
    def add_field_names(tracks: pd.DataFrame)->pd.DataFrame:
        if len(tracks.columns)==10:            
            tracks.columns = ['frame', 'track', 'x', 'y', 'w', 'h', 'score', 'cls', 'r3','r4']
            tracks['iou'] = -1.0
            tracks['stop'] = -1
            tracks['vzone'] = -1
            tracks['iob'] = -1.0
            tracks['hzone'] = -1
            tracks['leading'] = -1
            tracks['event'] = -1  
        elif len(tracks.columns) == 17:
            tracks.columns = ['frame', 'track', 'x', 'y', 'w', 'h', 'score', 'cls', 'r3','r4', 
                              'iou', 'stop', 'vzone', 'iob', 'hzone', 'leading', 'event']
        else:
            raise Exception('The number of fields is invalid.')
                
        return tracks

    '''copy from bot sort'''
    @staticmethod
    def ious(atlbrs, btlbrs):
        """
        Compute cost based on IoU
        :type atlbrs: list[tlbr] | np.ndarray
        :type atlbrs: list[tlbr] | np.ndarray

        :rtype ious np.ndarray
        """
        ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float64)
        if ious.size == 0:
            return ious

        ious = bbox_overlaps(
            np.ascontiguousarray(atlbrs, dtype=np.float64),
            np.ascontiguousarray(btlbrs, dtype=np.float64)
        )

        return ious
    
    @staticmethod
    def gen_zones(coords:list)->list[Polygon]:
        """
            Generate a list of shapely polygons
            Inputs:
                line_coords: a list of line coords ([[(x11, y11),(x12, y12)], [(x21, y21),(x22, y22)], ...])
            Returns:
                A list of PloyGons  
        """

        zones = []
        for coord in coords:
            zones.append(Polygon(coord))        

        return zones
    
