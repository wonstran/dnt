from shapely.geometry import Point, Polygon, LineString, box
from shapely import intersection, distance, intersects
from shapelysmooth import taubin_smooth, chaikin_smooth, catmull_rom_smooth
import geopandas as gpd, pandas as pd
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt
from dnt.label.labeler2 import Labeler
import os
from ..track import Tracker

class YieldAnalyzer:
    def __init__(self, waiting_dist_p:int, waiting_dist_y:int, 
                 leading_p:bool=False, leading_axis_p:str='y',
                 leading_y:bool=True, leading_axis_y:str='x',
                 yield_gap:int=10, fps:int=30, ref_point='bc', 
                 ref_offset:tuple=(0,0), filter_buffer:int=0,
                 p_zone:Polygon=None, y_zone:Polygon=None) -> None:
        '''
        Parameters:
            threshold: the hyperparameter to determine if a yield event (frame difference <=yield_gap*fps), default is 3 seconds
            fps: frames per sencond, default is 30
            ref_point: the reference point for lane recognition
                        br (buttom-right, default), bl (bottom-left), bc (bottom-center)
                        tl (top-left), tr (top-right), tc (top-center)
                        cc (center-center), cl (center-left), cr (center-right)
            ref_offset: the offset (x, y) for reference point, default is (0, 0)
            filter_buffer: the buffer between two track frames, default is 0 means tracks will be paired only if two tracks are overlapped in time
        '''
        self.waiting_dist_p = waiting_dist_p
        self.waiting_dist_y = waiting_dist_y
        self.leading_p = leading_p
        self.leading_axis_p = leading_axis_p
        self.leading_y = leading_y
        self.leading_axis_y = leading_axis_y
        self.yield_gap = yield_gap
        self.fps = fps
        self.ref_point = ref_point
        self.ref_offset = ref_offset
        self.filter_buffer = filter_buffer
        self.p_zone = p_zone
        self.y_zone = y_zone

    def analyze(self, tracks_p:pd.DataFrame, tracks_y:pd.DataFrame, name_p:str='', name_y:str=''):
        '''
        Parameters:
            tracks_p: tracks with priority
            tracks_y: tracks should yield to tracks_p
        '''
        tracks_p = YieldAnalyzer.add_field_names(tracks_p)
        tracks_y = YieldAnalyzer.add_field_names(tracks_y)

        tracks_p = self.gen_ref(tracks_p, info=name_p)
        tracks_y = self.gen_ref(tracks_y, info=name_y)

        trajectories_p = self.gen_trajectory(tracks_p, info=name_p)
        trajectories_y = self.gen_trajectory(tracks_y, info=name_y)

        intersect_pairs = self.scan_intersections(trajectories_p, trajectories_y)
        if (self.p_zone is not None) and (self.y_zone is not None):
            intersect_pairs = self.scan_yield_events_byzone(intersect_pairs, tracks_p, tracks_y)
        else:
            intersect_pairs = self.scan_yield_events(intersect_pairs, tracks_p, tracks_y)
        
        return intersect_pairs

    def gen_ref(self, tracks:pd.DataFrame, info:str='', video_index:int=None, video_tot:int=None)->pd.DataFrame:
        if video_index and video_tot:
            tqdm.pandas(desc='Generating ref {}, {} of {}'.format(info, video_index, video_tot), unit='frames')
        else:
            tqdm.pandas(desc='Generating ref {}'.format(info), unit='frames') 
        
        if self.ref_point == 'cc':
            tracks[['ref_x', 'ref_y']] = tracks.progress_apply(lambda track: pd.Series([track['x'] + track['w']//2 + self.ref_offset[0], 
                                                              track['y'] + track['h']//2 + self.ref_offset[1]]), axis=1)
        elif self.ref_point == 'tc':
            tracks[['ref_x', 'ref_y']] = tracks.progress_apply(lambda track: pd.Series([track['x'] + track['w']//2 + self.ref_offset[0], 
                                                              track['y'] + self.ref_offset[1]]), axis=1)
        elif self.ref_point == 'bc':
            tracks[['ref_x', 'ref_y']] = tracks.progress_apply(lambda track: pd.Series([track['x'] + track['w']//2 + self.ref_offset[0], 
                                                              track['y'] + track['h'] + self.ref_offset[1]]), axis=1)
        elif self.ref_point == 'cl':
            tracks[['ref_x', 'ref_y']] = tracks.progress_apply(lambda track: pd.Series([track['x'] + self.ref_offset[0], 
                                                              track['y'] + track['h']//2 + self.ref_offset[1]]), axis=1)
        elif self.ref_point == 'cr':
            tracks[['ref_x', 'ref_y']] = tracks.progress_apply(lambda track: pd.Series([track['x'] + track['w'] + self.ref_offset[0], 
                                                              track['y'] + track['h']//2 + self.ref_offset[1]]), axis=1)
        elif self.ref_point == 'tl':
            tracks[['ref_x', 'ref_y']] = tracks.progress_apply(lambda track: pd.Series([track['x'] + self.ref_offset[0], 
                                                              track['y'] + self.ref_offset[1]]), axis=1)
        elif self.ref_point == 'tr':
            tracks[['ref_x', 'ref_y']] = tracks.progress_apply(lambda track: pd.Series([track['x'] + track['w'] + self.ref_offset[0], 
                                                              track['y'] + self.ref_offset[1]]), axis=1)
        elif self.ref_point == 'bl':
            tracks[['ref_x', 'ref_y']] = tracks.progress_apply(lambda track: pd.Series([track['x'] + self.ref_offset[0], 
                                                              track['y'] + tracks['h'] + self.ref_offset[1]]), axis=1)
        elif self.ref_point == 'br':
            tracks[['ref_x', 'ref_y']] = tracks.progress_apply(lambda track: pd.Series([track['x'] + track['w'] + self.ref_offset[0], 
                                                              track['y'] + track['h'] + self.ref_offset[1]]), axis=1)    
        else:
            tracks[['ref_x', 'ref_y']] = tracks.progress_apply(lambda track: pd.Series([track['x'] + track['w']//2 + self.ref_offset[0], 
                                                              track['y'] + track['h'] + self.ref_offset[1]]), axis=1)
        
        return tracks
        
    def gen_trajectory(self, tracks:pd.DataFrame, info:str='', video_index:int=None, video_tot:int=None)->gpd.GeoDataFrame:
        
        ids = tracks['track'].unique()
        results = []
        lines = []
        pbar = tqdm(total=len(ids), unit='track')
        if video_index and video_tot:
            pbar.desc = 'Generating trajectory {}, {} of {}'.format(info, video_index, video_tot)
        else:
            pbar.desc = 'Generating trajectory {}'.format(info)

        for id in ids:
            frames = tracks[tracks['track']==id].sort_values(by=['frame'])
            line = LineString(list(zip(frames['ref_x'].values.tolist(), frames['ref_y'].values.tolist())))
            results.append([id, frames['frame'].values, frames['ref_x'].values, frames['ref_y'].values])
            lines.append(line)
            pbar.update()
        pbar.close()

        df = pd.DataFrame(results, columns=['track', 'frames', 'ref_x', 'ref_y'])
        return gpd.GeoDataFrame(df, geometry=lines)
        
    def scan_intersections(self, trajectories_p:gpd.GeoDataFrame, trajectories_y:gpd.GeoDataFrame, 
                           video_index:int=None, video_tot:int=None):

        results = []
        points = []
        buffer = self.filter_buffer*self.fps

        pbar = tqdm(total=len(trajectories_p)*len(trajectories_y), unit=' pair')
        if video_index and video_tot:
            pbar.desc = 'Scan intesections {} of {}'.format(video_index, video_tot)
        else:
            pbar.desc = 'Scan intesections'
        for index1, trj_p in trajectories_p.iterrows():
            for index2, trj_y in trajectories_y.iterrows():
                '''
                if (trj_p['track'] == 511) and (trj_y['track'] == 132590):
                    print(trj_p['frames'].min(), trj_p['frames'].max()) 
                    print(trj_y['frames'].min(), trj_y['frames'].max()) 
                    input('...')
                '''                
                
                if ((trj_p['frames'].min() > (trj_y['frames'].max() + buffer)) or 
                    (trj_y['frames'].min() > (trj_p['frames'].max() + buffer))): 
                    pass    
                else:
                    geo_intersect = intersection(trj_p.geometry, trj_y.geometry)
                    if not geo_intersect.is_empty:
                        point = self.__recursion(geo_intersect)
                        point_xy = np.array([point.x, point.y])
                        '''
                        if (trj_p['track'] == 2414) & (trj_y['track'] == 386456):
                            plt.gca().invert_yaxis()
                            plt.plot(*trj_p.geometry.xy, color='red')
                            plt.plot(*trj_y.geometry.xy, color='green')
                            plt.scatter(*point.xy)
                            plt.savefig('/mnt/d/a.jpg')

                            import cv2
                            img = cv2.imread('/mnt/d/b.png')
                            for x, y in list(zip(trj_p['ref_x'], trj_p['ref_y'])):
                                img = cv2.circle(img, (x, y), 1, (0, 0, 255))
                            for x, y in list(zip(trj_y['ref_x'], trj_y['ref_y'])):
                                img = cv2.circle(img, (x, y), 1, (0, 255, 0))
                            img = cv2.circle(img, (int(point.x), int(point.y)), 5, (255, 255, 255), thickness=-1)
                            print(int(point.x), int(point.y))               
                            cv2.imwrite('/mnt/d/c.jpg', img)
                            input('...')
                        '''
                        ref_pxy = np.array(list(zip(trj_p['ref_x'], trj_p['ref_y'])))
                        index_p = self.__get_closest_index(ref_pxy, point_xy)
                        frame_p = trj_p['frames'][index_p]
                        
                        ref_yxy = np.array(list(zip(trj_y['ref_x'], trj_y['ref_y'])))
                        index_y = self.__get_closest_index(ref_yxy, point_xy)
                        frame_y = trj_y['frames'][index_y]

                        results.append([trj_p['track'], frame_p, trj_y['track'], frame_y, frame_y-frame_p])
                        points.append(point)
                pbar.update()
        
        pbar.close()

        df = pd.DataFrame(results, columns=['track_p', 'frame_p', 'track_y', 'frame_y', 'frame_gap'])
        return gpd.GeoDataFrame(df, geometry=points)

    def scan_yield_events(self, pairs:gpd.GeoDataFrame, tracks_p:gpd.GeoDataFrame, tracks_y:gpd.GeoDataFrame, 
                          video_index:int=None, video_tot:int=None)->pd.DataFrame:

        pbar = tqdm(total=len(pairs), unit=' pair')
        if video_index and video_tot:
            pbar.desc = 'Scan events {} of {}'.format(video_index, video_tot)
        else:
            pbar.desc = 'Scan events'
        for index, pair in pairs.iterrows():
            
            if pair['frame_p'] < pair['frame_y']:

                selected = tracks_y[(tracks_y['track']==pair['track_y']) & (tracks_y['frame']<=pair['frame_y'])].copy() # find the frames for track_y before reaching conflict point
                selected['dif'] = selected['frame'].apply(lambda f: abs(f - pair['frame_p'])) # calculate the frame difference between each before frame and the conflict frame
                selected = selected[selected['dif']<=self.yield_gap]        # the frame difference should <= self.yield_gap
                closest = selected[selected['dif']==selected['dif'].min()]  # find the closest frame
                if len(closest)>0:
                    closest_point = Point(closest['x'].iloc[0]+closest['w'].iloc[0]/2, 
                                          closest['y'].iloc[0]+closest['h'].iloc[0]) # find the cloest point
                    dist = distance(closest_point, pair.geometry)   # calculate the distance
                    pairs.at[index, 'dist_y'] = dist
                    inwaiting = 1 if dist<=self.waiting_dist_y else 0

                    if self.leading_y:
                        isleading = 1 if self.__is_leading(pair['track_y'], closest['frame'].iloc[0], closest['x'].iloc[0], 
                                      closest['y'].iloc[0], pair.geometry.x, pair.geometry.y, tracks_y, self.leading_axis_y) else 0
                    else:
                        isleading = 1
                
                    if (inwaiting == 1) and (isleading == 1):
                        pairs.at[index, 'event'] = 'yield'

                    pairs.at[index, 'closest_frame'] = closest['frame'].iloc[0]

            elif pair['frame_p'] >= pair['frame_y']:
                selected = tracks_p[(tracks_p['track']==pair['track_p']) & (tracks_p['frame']<=pair['frame_p'])].copy()
                selected['dif'] = selected['frame'].apply(lambda f: abs(f - pair['frame_y']))
                selected = selected[selected['dif']<=self.yield_gap] 
                closest = selected[selected['dif']==selected['dif'].min()]
                if len(closest)>0:
                    closest_point = Point(closest['x'].iloc[0]+closest['w'].iloc[0]/2, 
                                          closest['y'].iloc[0]+closest['h'].iloc[0])
                    dist = distance(closest_point, pair.geometry)
                    inwaiting = 1 if dist<=self.waiting_dist_p else 0
                    pairs.at[index, 'dist_p'] = dist
                    if self.leading_p:
                        isleading = 1 if self.__is_leading(pair['track_p'], closest['frame'].iloc[0], closest['x'].iloc[0], 
                                      closest['y'].iloc[0], pair.geometry.x, pair.geometry.y, tracks_p, self.leading_axis_p) else 0
                    else:
                        isleading = 1
                
                    if (inwaiting == 1) and (isleading == 1):
                        pairs.at[index, 'event'] = 'not_yield'

                    pairs.at[index, 'closest_frame'] = closest['frame'].iloc[0]

            pbar.update()
        
        pbar.close()

        return self.__export(pairs)
    
    def scan_yield_events_byzone(self, pairs:gpd.GeoDataFrame, tracks_p:gpd.GeoDataFrame, tracks_y:gpd.GeoDataFrame, 
                                 video_index:int=None, video_tot:int=None)->pd.DataFrame:

        pbar = tqdm(total=len(pairs), unit=' pair')
        if video_index and video_tot:
            pbar.desc = 'Scan events {} of {}'.format(video_index, video_tot)
        else:
            pbar.desc = 'Scan events'
        for index, pair in pairs.iterrows():
            
            if pair['frame_p'] < pair['frame_y']:

                selected = tracks_y[(tracks_y['track']==pair['track_y']) & (tracks_y['frame']<=pair['frame_y'])].copy() # find the frames for track_y before reaching conflict point
                selected['dif'] = selected['frame'].apply(lambda f: abs(f - pair['frame_p'])) # calculate the frame difference between each before frame and the conflict frame
                selected = selected[selected['dif']<=self.yield_gap]        # the frame difference should <= self.yield_gap
                closest = selected[selected['dif']==selected['dif'].min()]  # find the closest frame
                if len(closest)>0:
                    x, y, w, h = closest['x'].iloc[0], closest['y'].iloc[0], closest['w'].iloc[0], closest['h'].iloc[0]
                    closest_bbox = Polygon([(x, y), (x + w, y), (x + w, y + h), (x, y + h)])
                    
                    inwaiting = 1 if intersects(closest_bbox, self.y_zone) else 0
                    pairs.at[index, 'dist_y'] = -1
                    if self.leading_y:
                        isleading = 1 if self.__is_leading(pair['track_y'], closest['frame'].iloc[0], closest['x'].iloc[0], 
                                      closest['y'].iloc[0], pair.geometry.x, pair.geometry.y, tracks_y, self.leading_axis_y) else 0
                    else:
                        isleading = 1
                
                    if (inwaiting == 1) and (isleading == 1):
                        pairs.at[index, 'event'] = 'yield'

                    pairs.at[index, 'closest_frame'] = closest['frame'].iloc[0]

            elif pair['frame_p'] >= pair['frame_y']:
                selected = tracks_p[(tracks_p['track']==pair['track_p']) & (tracks_p['frame']<=pair['frame_p'])].copy()
                selected['dif'] = selected['frame'].apply(lambda f: abs(f - pair['frame_y']))
                selected = selected[selected['dif']<=self.yield_gap] 
                closest = selected[selected['dif']==selected['dif'].min()]
                if len(closest)>0:
                    x, y, w, h = closest['x'].iloc[0], closest['y'].iloc[0], closest['w'].iloc[0], closest['h'].iloc[0]
                    closest_bbox = Polygon([(x, y), (x + w, y), (x + w, y + h), (x, y + h)])
                            
                    inwaiting = 1 if intersects(closest_bbox, self.p_zone) else 0
                    pairs.at[index, 'dist_p'] = -1
                    if self.leading_p:
                        isleading = 1 if self.__is_leading(pair['track_p'], closest['frame'].iloc[0], closest['x'].iloc[0], 
                                      closest['y'].iloc[0], pair.geometry.x, pair.geometry.y, tracks_p, self.leading_axis_p) else 0
                    else:
                        isleading = 1
                
                    if (inwaiting == 1) and (isleading == 1):
                        pairs.at[index, 'event'] = 'not_yield'

                    pairs.at[index, 'closest_frame'] = closest['frame'].iloc[0]

            pbar.update()
        
        pbar.close()

        return self.__export(pairs)
    
    def __export(self, pairs:gpd.GeoDataFrame):
        pairs['int_x'] = pairs.geometry.x
        pairs['int_y'] = pairs.geometry.y
        return pd.DataFrame(pairs.drop(columns=['geometry']))

    def __recursion(self, intersect):
        if intersect.geom_type == 'Point':
            return Point(intersect.coords[0])

        elif intersect.geom_type == 'MultiPoint':
            return Point(intersect.geoms[0].x, intersect.geoms[0].y)

        elif intersect.geom_type == 'LineString':
            return Point(intersect.coords[0])

        elif intersect.geom_type == 'MultiLineString':
            return Point(intersect.geoms[0].coords[0])

        else:
            for geom in intersect.geoms:
                return self.__recursion(geom)

    def __get_closest_index(self, ref:np.array, point:np.array) -> int:
        return np.argmin(np.sqrt(np.sum(np.square(ref - point), axis=1)))

    def __is_leading(self, track_id:int, frame_close:int, close_x:int, close_y:int, int_x:int, int_y:int, 
                          tracks:gpd.GeoDataFrame, leading_axis:str):
        
        min_x = min(close_x, int_x)
        max_x = max(close_x, int_x)
        min_y = min(close_y, int_y)
        max_y = max(close_y, int_y)
        if leading_axis == 'x':
            leadings = tracks[(tracks['track']!=track_id) & (tracks['frame']==frame_close) & 
                              (tracks['ref_x']>=min_x) & (tracks['ref_x']<=max_x)]
        elif leading_axis == 'y':
            leadings = tracks[(tracks['track']!=track_id) & (tracks['frame']==frame_close) & 
                              (tracks['ref_y']>=min_y) & (tracks['ref_y']<=max_y)]
        elif leading_axis == 'xy':
            leadings = tracks[(tracks['track']!=track_id) & (tracks['frame']==frame_close) & 
                              (tracks['ref_x']>=min_x) & (tracks['ref_x']<=max_x) &
                              (tracks['ref_y']>=min_y) & (tracks['ref_y']<=max_y)]
        
        if len(leadings) > 0:
            return False
        else:
            return True 

    def __get_waiting_trj_y(self, track_id:int, frame_id_p:int, frame_id_y:int, intersect_point:Point, trajectories:gpd.GeoDataFrame) -> any:
        
        result = -1
        if frame_id_p < frame_id_y:
            selected = trajectories[trajectories['track']==track_id]
            df = pd.DataFrame(np.vstack((selected['frames'].tolist(), selected['ref_x'].tolist(), selected['ref_y'].tolist())).T, 
                              columns=['frame', 'x', 'y'])
            df = df.loc[df['frame']<=frame_id_y]
            df['dif'] = df['frame'].apply(lambda f: abs((f - frame_id_p)))
            closest = df[df['dif']==df['dif'].min()]
            closest_point = Point(closest['x'].ioc[0], closest['y'].ioc[0])
            dist = distance(closest, intersect_point)

        return result

    @staticmethod
    def add_field_names(tracks: pd.DataFrame)->pd.DataFrame:
        if len(tracks.columns)==10:            
            tracks.columns = ['frame', 'track', 'x', 'y', 'w', 'h', 'score', 'cls', 'r3','r4']
            tracks['ref_x'] = -1
            tracks['ref_y'] = -1
        elif len(tracks.columns) == 12:
            tracks.columns = ['frame', 'track', 'x', 'y', 'w', 'h', 'score', 'cls', 'r3','r4', 
                              'ref_x', 'ref_y']
        else:
            raise Exception('The number of fields is invalid.')        
        return tracks
    
    @staticmethod
    def draw_event_clips(yields:pd.DataFrame, tracks_p:pd.DataFrame, tracks_y:pd.DataFrame, input_video:str, out_path:str,
                         method:str='all', random_number:int=10, event_list:list=None, size:int=1, thick:int=2, padding:int=0, 
                         show_track:bool=False, show_desc:bool=False, verbose:bool=True,
                         video_index:int=None, video_tot:int=None):
        '''
        Parameters:
        - yields: the dataframe of stop events
        - tracks_p: the dataframe of tracks with priority
        - tracks_y: the dataframe of tracks should yield
        - input_video: the raw video file, if None, generate the label files only
        - out_path: the folder for outputing track clips
        - method: 'all' (default) - all tracks,  'specify' - specify events, 
                    'yield' - yield events, 'not_yield' - non-yield events, 'none' - non interaction events
        - random_number: the number of track ids if method == 'random'
        - event_list: the list of events [(track_p_id, track_y_id)] if method == 'specify'
        - size: font size, default is 1
        - thick: line thinckness, defualt is 2
        - padding: add addtional frames at beggining and ending, default is 0
        - show_track: show track id, default is False
        - show_desc: show event desc, default is Falase
        - verbose: if show progressing bar, default is True
        - video_idex: the index of the video in processing
        - video_tot: the total number of videos

        Return:
        - A list of dataframes for labels, ['frame','type','coords','color','size','thick','desc']
        '''
        
        tracks_p = YieldAnalyzer.add_field_names(tracks_p)
        tracks_y = YieldAnalyzer.add_field_names(tracks_y)

        tracks_p_grouped = tracks_p.groupby('track')
        tracks_y_grouped = tracks_y.groupby('track')


        if method == 'all':
            events = yields
        elif method == 'specify':
            events = []
            for p, y in event_list:
                events.append(yields[(yields['track_p']==p) & (yields['track_y']==y)])
            events = pd.concat(events)
        elif method == 'yield':
            events = yields[yields['event']=='yield'].copy()
        elif method == 'not_yield':
            events = yields[yields['event']=='not_yield'].copy()
        elif method == 'event':
            events = yields[yields['event'].isin(['yield', 'not_yield'])].copy()
        elif method == 'none':
            events = yields[~yields['event'].isin(['yield', 'not_yield'])].copy()        

        pbar = tqdm(total= len(events) , unit='event')        
        if video_index and video_tot:
            pbar.set_description_str("Generating labeles for yielding events {} of {}".format(video_index, video_tot))
        else:
            pbar.set_description_str("Generating labels for yielding events")

        labeler = Labeler()
        for index, event in events.iterrows():
            result = []
            track_p_id = event['track_p']
            track_y_id = event['track_y']
            int_x = int(event['int_x'])
            int_y = int(event['int_y'])

            if event['event'] == 'yield':
                desc = 'yield'
                color = (0, 255, 0)
            elif event['event'] == 'not_yield':
                desc = 'not yield'
                color = (0, 0, 255)
            else:
                desc = 'no interaction'
                color = (255, 0, 0)
            
            frames_p = tracks_p_grouped.get_group(track_p_id)            
            frames_y = tracks_y_grouped.get_group(track_y_id)
            frames = pd.concat([frames_p, frames_y])
            for index2, frame in frames.iterrows():
                label_text = ''
                if show_track:
                    label_text+='| T:'+str(int(frame['track']))
                
                if show_desc:
                    label_text+= ' | '+desc

                result.append([frame['frame'], 'bbox', [(frame['x'], frame['y']), (frame['x']+frame['w'], frame['y']+frame['h'])], 
                            color, size, thick, label_text])
                result.append([frame['frame'], 'box', [(int_x - 5, int_y - 5), (int_x + 5, int_y + 5)], 
                            (255, 255, 0), size, thick, ''])

            df = pd.DataFrame(result, columns=['frame','type','coords','color','size','thick','desc'])
                
            if out_path:
                file_name = os.path.join(out_path, str(track_p_id)+'-'+str(track_y_id)+'_label.csv')
                df.to_csv(file_name, index=False)

            if input_video:
                file_name = os.path.join(out_path, str(track_p_id)+'-'+str(track_y_id)+'_label.mp4')
                min_frame = df['frame'].min() - padding
                max_frame = df['frame'].max() + padding
                labeler.draw(input_video=input_video, output_video=file_name, draws=df, 
                                start_frame=min_frame, end_frame=max_frame, verbose=False)

            if verbose:
                pbar.update()      
        
        pbar.close()

        