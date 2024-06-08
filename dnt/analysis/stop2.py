from shapely.geometry import Point, Polygon, LineString, box
import geopandas as gpd, pandas as pd
import datetime
from tqdm import tqdm

class StopAnalyzer():
    def __init__(self, 
                 hzones: list[Polygon] = [], 
                 vzones: list[Polygon] = [],
                 event_dict:dict[int, list[int]] = {}, 
                 stop_iou:float=0.97, 
                 frame_buffer:int=5,
                 verbose:bool=True):
        
        self.hzones = hzones
        self.vzones = vzones
        self.event_dict = event_dict
        self.stop_iou = stop_iou
        self.frame_buffer = frame_buffer
        self.verbose = verbose

    def analyze(self, tracks:pd.DataFrame=None, track_file:str=None,video_index:int=None, video_tot:int=None)->list[pd.DataFrame, pd.DataFrame]:

        if tracks is None:
            tracks = pd.read_csv(track_file, header=None)
        tracks = StopAnalyzer.add_field_names(tracks)

        tracks = self.scan_stop(tracks, video_index, video_tot)
        tracks = self.identify_event(tracks, video_index, video_tot)
        events = self.count_event(tracks, video_index, video_tot)

        return tracks, events
    
    def scan_stop(self, tracks:pd.DataFrame, video_index:int=None, video_tot:int=None)->pd.DataFrame:
        
        ids = tracks['track'].unique()
        tracks['hzone'] = -1
        
        pbar = tqdm(total=len(ids), unit=' tracks')
        if video_index and video_tot:
            pbar.set_description_str("Scanning stops {} of {}".format(video_index, video_tot))
        else:
            pbar.set_description_str("Scanning stops")

        for id in ids:
            track = tracks[tracks['track'] == id].sort_values(by='frame') # sort by frame in ascending

            for i in range(self.frame_buffer, len(track)):
                index = track.iloc[i].name
                bb0 = [track.iloc[i-self.frame_buffer]['x'], track.iloc[i-self.frame_buffer]['y'], 
                       track.iloc[i-self.frame_buffer]['w'], track.iloc[i-self.frame_buffer]['h']]
                bb1 = [track.iloc[i]['x'], track.iloc[i]['y'], track.iloc[i]['w'], track.iloc[i]['h']]
                iou_score = StopAnalyzer.iou(bb0, bb1)

                tracks.at[index, 'iou'] = iou_score 
                if iou_score >= self.stop_iou:
                    tracks.at[index, 'stop'] = 1

                if len(self.vzones)>0:
                    center = Point(track.iloc[i]['x']+track.iloc[i]['w'], track.iloc[i]['y']+track.iloc[i]['h'])
                    for j in range(len(self.vzones)):
                        if center.within(self.vzones[j]):
                            tracks.loc[tracks['track']==id, 'vzone']=j
                            break

                if len(self.hzones)>0:
                    bb = box(track.iloc[i]['x'], track.iloc[i]['y'], track.iloc[i]['x'] 
                             + track.iloc[i]['w'], track.iloc[i]['y'] + track.iloc[i]['h'])
                    for j in range(len(self.hzones)):
                        if bb.intersects(self.hzones[j]):
                            if j > tracks.at[index, 'hzone']:
                                tracks.at[index, 'hzone'] = j
                                                 
            if self.verbose:
                pbar.update()

        pbar.close()
        return tracks  

    def identify_event(self, tracks:pd.DataFrame, video_index:int=None, video_tot:int=None)->pd.DataFrame:

        pbar = tqdm(total=len(tracks), unit=' frames')
        if video_index and video_tot:
            pbar.set_description_str("Identifying events {} of {}".format(video_index, video_tot))
        else:
            pbar.set_description_str("Identifying events")

        for i in range(len(tracks)):

            if tracks.iloc[i]['stop'] == 1:
                vzone = tracks.iloc[i]['vzone']
                hzone = tracks.iloc[i]['hzone']
                frame = tracks.iloc[i]['frame']
                id = tracks.iloc[i]['track']
                vehs_inlane = tracks.loc[(tracks['frame']==frame) & (tracks['track']!=id) & (tracks['vzone']==vzone) & (tracks['hzone']>hzone)]

                if len(vehs_inlane)==0:
                    pre_key = -1
                    for key in self.event_dict:
                        if key >= pre_key:
                            pre_key = key
                            if tracks.iloc[i]['hzone'] in self.event_dict[key]:
                                tracks.at[i, 'event'] = key

            if self.verbose:
                pbar.update()
        
        pbar.close()

        return tracks

    def count_event(self, tracks:pd.DataFrame, video_index:int=None, video_tot:int=None)->pd.DataFrame:

        pbar = tqdm(unit='events')
        results = []
        for key in self.event_dict:

            vehicles = tracks.loc[tracks['event']==key]['track'].unique()

            if video_index and video_tot:
                pbar.set_description_str("Counting event {} for {} of {}".format(key, video_index, video_tot))
            else:
                pbar.set_description_str("Counting event {}".format(key))

            pbar.total = len(vehicles)
            for vehicle in vehicles:

                track = tracks[(tracks['track'] == vehicle) & (tracks['event'] == key)]

                start_frame = int(track['frame'].min())
                end_frame = int(track['frame'].max())
                vzone = track['vzone'].mode()[0]
                results.append([key, vehicle, vzone, start_frame, end_frame])
                
                if self.verbose:
                    pbar.update()

        results = pd.DataFrame(results, columns=['event', 'track', 'vzone', 'start_frame', 'end_frame'])
        pbar.close()

        return results
    
    def generate_labels(self, tracks:pd.DataFrame, events:pd.DataFrame, video_index:int=None, video_tot:int=None)->pd.DataFrame:
        
        event_tracks = tracks.loc[tracks['event']>-1].copy()
        for index, event in events.iterrows():
            track_id = event['track']
            start_frame = event['start_frame']
            end_frame = event['end_frame']
            event_id = event['event']

            event_tracks.loc[(event_tracks['track']==track_id) & (event_tracks['frame']>=start_frame) 
                             & (event_tracks['frame']<=end_frame), 'event'] = event_id

        pbar = tqdm(total= len(event_tracks))        
        if video_index and video_tot:
            pbar.set_description_str("Generating labeles for {} of {}".format(video_index, video_tot))
        else:
            pbar.set_description_str("Generating labels")

        results = []
        for index, track in event_tracks.iterrows():
            desc = 'Stop'
            if track['event'] == 1:
                desc = 'Pass'
            elif track['event'] == 2:
                desc = 'Invading'

            results.append([track['frame'], 'bbox', [(track['x'], track['y']), (track['x']+track['w'], track['y']+track['h'])], 
                            (0, 255,0), 1, 1, desc])
            
            if self.verbose:
                pbar.update()

        df = pd.DataFrame(results, columns=['frame','type','coords','color','size','thick','desc'])
        df.sort_values(by='frame')
        
        return df

    
    @staticmethod
    def add_field_names(tracks: pd.DataFrame)->pd.DataFrame:
        if len(tracks.columns)!=10:
            raise Exception('The number of fields is not nine.')
        tracks.columns = ['frame', 'track', 'x', 'y', 'w', 'h', 'score', 'cls', 'r3','r4']
        tracks['iou'] = -1.0
        tracks['stop'] = -1
        tracks['vzone'] = -1
        tracks['hzone'] = -1
        tracks['event'] = -1   
        return tracks
    
    @staticmethod
    def iou(bb1:tuple[int, int, int, int], bb2:tuple[int, int, int, int]):
        """
        Calculate the Intersection over Union (IoU) of two bounding boxes.
        Parameters
        ----------
        bb1 : [x1, y1, w, h]
            x1, y1 - top left corner
            w, h - width and height
        bb2 : [x1, y1, w, h]
            x1, y1 - top left corner
            w, h - width and height
        Returns
        -------
        iou: float [0, 1]
        """

        assert bb1[0] < bb1[0] + bb1[2]
        assert bb1[1] < bb1[1] + bb1[3]
        assert bb2[0] < bb2[0] + bb2[2]
        assert bb2[1] < bb2[1] + bb2[3]

        # determine the coordinates of the intersection rectangle
        x_left = max(bb1[0], bb2[0])
        y_top = max(bb1[1], bb2[1])
        x_right = min(bb1[0]+bb1[2], bb2[0]+bb2[2])
        y_bottom = min(bb1[1]+bb1[3], bb2[1]+bb2[3])

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        # The intersection of two axis-aligned bounding boxes is always an
        # axis-aligned bounding box
        intersection_area = (x_right - x_left) * (y_bottom - y_top)

        # compute the area of both AABBs
        bb1_area = bb1[2] * bb1[3]
        bb2_area = bb2[2] * bb2[3]

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
        assert iou >= 0.0
        assert iou <= 1.0
        return iou

    @staticmethod
    def gen_zones(line_coords:list[list[tuple[int, int], tuple[int, int]]])->list[Polygon]:
        """
            Generate a list of shapely polygons
            Inputs:
                line_coords: a list of line coords ([[(x11, y11),(x12, y12)], [(x21, y21),(x22, y22)], ...])
            Returns:
                A list of PloyGons  
        """

        zones = []
        for i in range(1, len(line_coords)):
            shell = line_coords[i-1] + [line_coords[i][1], line_coords[i][0]]
            zones.append(Polygon(shell))        

        return zones
      
if __name__=='__main__':
    track_file = '/mnt/d/videos/ped2stage/tracks/gh021291_track_veh.txt'
    tracks = pd.read_csv(track_file, header=None)
    tracks['r1'] = -1
    tracks['r2'] = -1
    tracks['r3'] = -1
    tracks = StopAnalyzer.add_field_names(tracks)

    v_coords = []
    h_coords = [[(194, 769),(480, 484)], 
          [(550, 812),(767, 487)],
          [(677, 835),(824, 484)],
          [(1087, 870),(958, 483)]]
    hzones = StopAnalyzer.gen_zones(h_coords)

    event_dict = {
        0: [0],    # Stop before stop bar
        1: [1],    # Pass stop bar
        2: [2]     # Invading crosswalk
        }
    analyzer = StopAnalyzer(hzones=hzones, event_dict=event_dict)

    tracks_stop = analyzer.scan_stop(tracks)
    tracks_event = analyzer.identify_event(tracks_stop)
    tracks_event.to_csv('/mnt/d/videos/ped2stage/stop/gh021291_track_veh_stop.txt', index=False)

    event_count = analyzer.count_event(tracks_event)
    event_count.to_csv('/mnt/d/videos/ped2stage/stop/gh021291_track_stops_count.txt', index=False)

    labels = analyzer.generate_labels(tracks_event, event_count)
    labels.to_csv('/mnt/d/videos/ped2stage/stop/gh021291_track_stops_label.txt', index=False)

    
