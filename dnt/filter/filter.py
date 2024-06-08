import pandas as pd
from shapely import geometry, LineString, Polygon, Point
import geopandas as gpd
from tqdm import tqdm

class Filter:
    def __init__(self) -> None:
        pass
    
    @staticmethod
    def filter_iou(detections: pd.DataFrame, zones: geometry.multipolygon = None, class_list: list[int] = None, score_threshold: float = 0):

        detections = detections.loc[detections[6]>=score_threshold].copy()

        # filter classess        
        if class_list:
            detections = detections.loc[detections[7].isin(class_list)].copy()

        if zones:
            # filter locations
            g = [geometry.Point(xy) for xy in zip((detections[2] + detections[4]/2), (detections[3] + detections[5]/2))]
            geo_detections = gpd.GeoDataFrame(detections, geometry=g)

            frames = geo_detections.loc[geo_detections.geometry.within(zones)].drop(columns='geometry')

            if frames:
                results = pd.concat(frames)
                results = results[~results.index.duplicated()].reset_index(drop=True)
            else:
                results = pd.DataFrame()
                
        else:
            results = detections

        return results

    @staticmethod
    def filter_tracks(tracks:pd.DataFrame, 
                    include_zones: geometry.MultiPolygon = None, 
                    exclude_zones: geometry.MultiPolygon = None, 
                    video_index:int = None, video_tot:int = None):

        g = [geometry.Point(xy) for xy in zip((tracks[2] + tracks[4]/2), (tracks[3] + tracks[5]/2))]
        geo_tracks = gpd.GeoDataFrame(tracks, geometry=g)
           
        track_ids = tracks[1].unique()
        include_ids = []
        exclude_ids = []

        pbar = tqdm(total=len(track_ids), unit=' tracks')
        if video_index and video_tot: 
            pbar.set_description_str("Filtering zones {} of {}".format(video_index, video_tot))
        else:
            pbar.set_description_str("Filtering zones ")
        
        for track_id in track_ids:
            if include_zones:
                selected_tracks = geo_tracks.loc[(geo_tracks[1]==track_id) & (geo_tracks.geometry.within(include_zones))]
                if len(selected_tracks)>0:
                    include_ids.append(track_id)
            
            if exclude_zones:
                selected_tracks = geo_tracks.loc[(geo_tracks[1]==track_id) & (geo_tracks.geometry.within(exclude_zones))]
                if len(selected_tracks)>0:
                    exclude_ids.append(track_id)
            
            pbar.update()

        pbar.close()

        if len(include_ids)>0:
            results = tracks.loc[tracks[1].isin(include_ids)].copy()
        else:
            results = tracks.copy()

        if len(exclude_ids)>0:
            results = results.loc[~results[1].isin(exclude_ids)].copy()

        return results
    
    @staticmethod
    def filter_tracks_by_zones_agg(tracks:pd.DataFrame, 
                    zones: geometry.MultiPolygon = None, 
                    method: str = 'include',
                    ref_point: str = 'bc',
                    offset: tuple = (0, 0),
                    col_names = ['frame', 'track', 'x', 'y', 'w', 'h', 'score', 'cls', 'r3', 'r4'], 
                    video_index:int = None, video_tot:int = None)->pd.DataFrame:
        '''
        Filter tracks by zones.
        Inputs:
            tracks: tracks 
            zones: a list of polygons
            method - 'include' (default), 'exclude'
            ref_point - the reference point of a track bbox, 
                defalt is br, others - bl, bc, tl, tc, tr, cl, cc, cr
            offset - the offset to ref_point, default is (0, 0)
            video_index - video index
            video_tot - total videos
        Return:
            Filtered tracks
        '''

        try:
            tracks.columns = col_names
        except:
            print('Tracks is invalid!')

        if ref_point == 'cc':
            g = [Point(xy) for xy in zip((tracks['x'] + tracks['w']/2 + offset[0]), (tracks['y'] + tracks['h']/2 + offset[1]))]
        elif ref_point == 'tc':
            g = [Point(xy) for xy in zip((tracks['x'] + tracks['w']/2 + offset[0]), (tracks['y'] + offset[1]))]
        elif ref_point == 'bc':
            g = [Point(xy) for xy in zip((tracks['x'] + tracks['w']/2 + offset[0]), (tracks['y'] + tracks['h'] + offset[1]))]
        elif ref_point == 'cl':
            g = [Point(xy) for xy in zip((tracks['x'] + offset[0]), (tracks['y'] + tracks['h']/2 + offset[1]))]
        elif ref_point == 'cr':
            g = [Point(xy) for xy in zip((tracks['x'] + tracks['w'] + offset[0]), (tracks['y'] + tracks['h']/2 + offset[1]))]
        elif ref_point == 'tl':
            g = [Point(xy) for xy in zip((tracks['x'] + offset[0]), (tracks['y'] + offset[1]))]
        elif ref_point == 'tr':
            g = [Point(xy) for xy in zip((tracks['x'] + tracks['w'] + offset[0]), (tracks['y'] + offset[1]))]
        elif ref_point == 'bl':
            g = [Point(xy) for xy in zip((tracks['x'] + offset[0]), (tracks['y'] + tracks['h'] + offset[1]))]
        elif ref_point == 'br':
            g = [Point(xy) for xy in zip((tracks['x'] + tracks['w'] + offset[0]), (tracks['y'] + tracks['h'] + offset[1]))]    
        else: 
            g = [Point(xy) for xy in zip((tracks['x'] + tracks['w']/2 + offset[0]), (tracks['y'] + tracks['h'] + offset[1]))]

        geo_tracks = gpd.GeoDataFrame(tracks, geometry=g)

        matched_ids = []
        pbar = tqdm(total=len(zones), unit=' zones')
        if video_index and video_tot: 
            pbar.set_description_str("Filtering zones {} of {}".format(video_index, video_tot))
        else:
            pbar.set_description_str("Filtering zones ")
        
        for zone in zones:
            matched  = geo_tracks[geo_tracks.geometry.within(zone)]
            if len(matched)>0:
                matched_ids.extend(matched['track'].unique().tolist())
            pbar.update()
        
        pbar.close()
        
        if len(matched_ids)>0:
            if method == 'include':
                results = tracks.loc[tracks['track'].isin(matched_ids)].copy()                    
            else:
                results = tracks.loc[~tracks['track'].isin(matched_ids)].copy()
        else:
            results = tracks.copy()

        return results
    
    @staticmethod
    def filter_frames_by_zones_agg(tracks:pd.DataFrame, 
                    zones: geometry.MultiPolygon = None, 
                    method: str = 'include',
                    ref_point:str = 'bc',
                    offset: tuple = (0, 0),
                    col_names = ['frame', 'track', 'x', 'y', 'w', 'h', 'score', 'cls', 'r3', 'r4'], 
                    video_index:int = None, video_tot:int = None)->pd.DataFrame:
        '''
        Filter tracks by zones.
        Inputs:
            tracks - tracks 
            zones - zones (polygon)
            method - 'include' (default) include the tracks if they are within the zones; 'exclude' exclude the tracks if they are within the zones
            ref_point - the reference point of a track bbox, defalt is bottom_point, center_point, left_up, right_up, left_buttom, right_buttom
            offset - the offset to ref_point, default is (0, 0)
            video_index - video index
            video_tot - total videos
        Return:
            Filtered tracks
        '''

        try:
            tracks.columns = col_names
        except:
            print('Tracks is invalid!')

        if ref_point == 'cc':
            g = [Point(xy) for xy in zip((tracks['x'] + tracks['w']/2 + offset[0]), (tracks['y'] + tracks['h']/2 + offset[1]))]
        elif ref_point == 'tc':
            g = [Point(xy) for xy in zip((tracks['x'] + tracks['w']/2 + offset[0]), (tracks['y'] + offset[1]))]
        elif ref_point == 'bc':
            g = [Point(xy) for xy in zip((tracks['x'] + tracks['w']/2 + offset[0]), (tracks['y'] + tracks['h'] + offset[1]))]
        elif ref_point == 'cl':
            g = [Point(xy) for xy in zip((tracks['x'] + offset[0]), (tracks['y'] + tracks['h']/2 + offset[1]))]
        elif ref_point == 'cr':
            g = [Point(xy) for xy in zip((tracks['x'] + tracks['w'] + offset[0]), (tracks['y'] + tracks['h']/2 + offset[1]))]
        elif ref_point == 'tl':
            g = [Point(xy) for xy in zip((tracks['x'] + offset[0]), (tracks['y'] + offset[1]))]
        elif ref_point == 'tr':
            g = [Point(xy) for xy in zip((tracks['x'] + tracks['w'] + offset[0]), (tracks['y'] + offset[1]))]
        elif ref_point == 'bl':
            g = [Point(xy) for xy in zip((tracks['x'] + offset[0]), (tracks['y'] + tracks['h'] + offset[1]))]
        elif ref_point == 'br':
            g = [Point(xy) for xy in zip((tracks['x'] + tracks['w'] + offset[0]), (tracks['y'] + tracks['h'] + offset[1]))]    
        else: 
            g = [Point(xy) for xy in zip((tracks['x'] + tracks['w']/2 + offset[0]), (tracks['y'] + tracks['h'] + offset[1]))]


        geo_tracks = gpd.GeoDataFrame(tracks, geometry=g)

        matched_frames = []
        pbar = tqdm(total=len(zones), unit=' zones')
        if video_index and video_tot: 
            pbar.set_description_str("Filtering zones {} of {}".format(video_index, video_tot))
        else:
            pbar.set_description_str("Filtering zones ")
        
        for zone in zones:
            matched  = geo_tracks[geo_tracks.geometry.within(zone)]
            if len(matched)>0:
                matched_frames.extend(matched.index.values.tolist())
            pbar.update()
        
        pbar.close()
        
        if len(matched_frames)>0:
            if method == 'include':
                results = tracks.iloc[matched_frames].copy()                    
            else:
                results = tracks.drop(matched_frames, axis=0).copy()
        else:
            results = tracks.copy()

        return results
    
    @staticmethod
    def filter_tracks_by_zones(tracks:pd.DataFrame, 
                    zones: geometry.MultiPolygon = None,
                    method: str = 'list', 
                    ref_point: str = 'bc',
                    offset: tuple = (0, 0),
                    col_names = ['frame', 'track', 'x', 'y', 'w', 'h', 'score', 'cls', 'r3', 'r4'],
                    zone_name: str = 'zone',
                    video_index:int = None, video_tot:int = None)->pd.DataFrame:
        '''
        Filter tracks by zones.
        Inputs:
            tracks - tracks 
            zones - zones (polygon)
            method - 'list' (default), 'filter', 'label'
            ref_point - the reference point of a track bbox, 
                        br - buttom_right, center_point, 
                        left_up, right_up, left_buttom, right_buttom
            offset - the offset to ref_point, default is (0, 0)
            aggregate - combine outputs to one dataframe, add zone column
            zone_name - if aggregate, the field name of zone variable, default is 'zone' 
            video_index - video index
            video_tot - total videos
        Return:
            Filtered tracks
        '''

        try:
            tracks.columns = col_names
        except:
            print('Tracks is invalid!')

        if ref_point == 'cc':
            g = [Point(xy) for xy in zip((tracks['x'] + tracks['w']/2 + offset[0]), (tracks['y'] + tracks['h']/2 + offset[1]))]
        elif ref_point == 'tc':
            g = [Point(xy) for xy in zip((tracks['x'] + tracks['w']/2 + offset[0]), (tracks['y'] + offset[1]))]
        elif ref_point == 'bc':
            g = [Point(xy) for xy in zip((tracks['x'] + tracks['w']/2 + offset[0]), (tracks['y'] + tracks['h'] + offset[1]))]
        elif ref_point == 'cl':
            g = [Point(xy) for xy in zip((tracks['x'] + offset[0]), (tracks['y'] + tracks['h']/2 + offset[1]))]
        elif ref_point == 'cr':
            g = [Point(xy) for xy in zip((tracks['x'] + tracks['w'] + offset[0]), (tracks['y'] + tracks['h']/2 + offset[1]))]
        elif ref_point == 'tl':
            g = [Point(xy) for xy in zip((tracks['x'] + offset[0]), (tracks['y'] + offset[1]))]
        elif ref_point == 'tr':
            g = [Point(xy) for xy in zip((tracks['x'] + tracks['w'] + offset[0]), (tracks['y'] + offset[1]))]
        elif ref_point == 'bl':
            g = [Point(xy) for xy in zip((tracks['x'] + offset[0]), (tracks['y'] + tracks['h'] + offset[1]))]
        elif ref_point == 'br':
            g = [Point(xy) for xy in zip((tracks['x'] + tracks['w'] + offset[0]), (tracks['y'] + tracks['h'] + offset[1]))]    
        else: 
            g = [Point(xy) for xy in zip((tracks['x'] + tracks['w']/2 + offset[0]), (tracks['y'] + tracks['h'] + offset[1]))]

        geo_tracks = gpd.GeoDataFrame(tracks, geometry=g)

        matched_ids = []
        pbar = tqdm(total=len(zones), unit=' zones')
        if video_index and video_tot: 
            pbar.set_description_str("Filtering zones {} of {}".format(video_index, video_tot))
        else:
            pbar.set_description_str("Filtering zones ")
        
        for zone in zones:
            matched  = geo_tracks[geo_tracks.geometry.within(zone)]
            if len(matched)>0:
                matched_ids.append(matched['track'].unique().tolist())
            pbar.update()
        
        pbar.close()
        
        if (method == 'filter') or (method == 'label'):
            tracks[zone_name] = -1
            for i in range(len(matched_ids)):
                tracks.loc[tracks['track'].isin(matched_ids[i]), zone_name] = i
            if method == 'filter':
                results = tracks[tracks['zone']!=-1].copy()
            else:
                results = tracks
        else:
            results = []
            if len(matched_ids)>0:
                for i in range(len(matched_ids)):
                    result = tracks.loc[tracks['track'].isin(matched_ids[i])].copy()
                    results.append(result)

        return results
    
    @staticmethod
    def filter_tracks_by_lines(tracks:pd.DataFrame, 
                    lines: list[LineString]= None, 
                    method: str = 'include',
                    video_index:int = None, video_tot:int = None) -> pd.DataFrame:
        '''
        Filter tracks by lines
        Inputs:
            tracks - a DataFrame of tracks, [FRAME, TRACK_ID, TOPX, TOPY, WIDTH, LENGTH, RESERVED, RESERVED, RESERVED]
            lines - a list of LineString
            method - filtering method, include (default) - including tracks crossing the lines, exclude - exclude tracks crossing the lines
            video_index - the index of video for processing
            video_tot - the total number of videos
        Return:
            a DataFrame of [FRAME, TRACK_ID, TOPX, TOPY, WIDTH, LENGTH, RESERVED, RESERVED, RESERVED]
        '''
        
        track_ids = tracks[1].unique()
        ids = []

        pbar = tqdm(total=len(track_ids), unit=' tracks')
        if video_index and video_tot: 
            pbar.set_description_str("Filtering tracks {} of {}".format(video_index, video_tot))
        else:
            pbar.set_description_str("Filtering tracks ")
            
        for track_id in track_ids:
            selected = tracks.loc[(tracks[1]==track_id)].copy()
            if len(selected)>0:
                g = selected.apply(lambda track: Polygon([(track[2], track[3]), (track[2] + track[4], track[3]), 
                                (track[2] + track[4], track[3] + track[5]), (track[2], track[3] + track[5])]), axis =1)
                intersected = True
                for line in lines:
                    intersected = intersected and any(line.intersects(g).values.tolist())    

                if intersected:
                    ids.append(track_id)
                    
            pbar.update()

        pbar.close()

        results = []
        if method=='include':    
            results = tracks.loc[tracks[1].isin(ids)].copy()
        elif method=='exclude':
            results = tracks.loc[~tracks[1].isin(ids)].copy()

        results.sort_values(by=[0, 1], inplace=True)
        return results

    @staticmethod
    def filter_tracks_by_lines_v2(tracks:pd.DataFrame, 
                    lines: list[LineString]= None,
                    method: str = 'include',
                    tolerance: int = 0,
                    bbox_size: int = 0,
                    force_line_indexes: list[int] = None,
                    video_index:int = None, video_tot:int = None) -> pd.DataFrame:
        '''
        Filter tracks by lines
        Inputs:
            tracks - a DataFrame of tracks, [FRAME, TRACK_ID, TOPX, TOPY, WIDTH, LENGTH, RESERVED, RESERVED, RESERVED]
            lines - a list of LineString
            method - filtering method, include (default) - including tracks crossing the lines, exclude - exclude tracks crossing the lines
            tolerance - if a bbox intesect the reference lines of (number of lanes - tolerance), it is hit. default is 0.
            force_line_indexes: the line indexes that a bbox must intersect for matching 
            bbox_size - the size of detection bbox, default is 0 - the orginal bbox  
            video_index - the index of video for processing
            video_tot - the total number of videos
        Return:
            a DataFrame of [FRAME, TRACK_ID, TOPX, TOPY, WIDTH, LENGTH, RESERVED, RESERVED, RESERVED]
        '''
        tracks.columns = ['frame', 'track', 'x', 'y', 'w', 'h', 'score', 'cls', 'r3', 'r4']
        track_ids = tracks['track'].unique()
        ids = []

        # set hit criterion
        hit_criterion = len(lines) - tolerance
        if (hit_criterion < 1) or (hit_criterion > len(lines)):
            hit_criterion = len(lines)

        pbar = tqdm(total=len(track_ids), unit=' tracks')
        if video_index and video_tot: 
            pbar.set_description_str("Filtering tracks {} of {}".format(video_index, video_tot))
        else:
            pbar.set_description_str("Filtering tracks ")
            
        for track_id in track_ids:
            selected = tracks.loc[(tracks['track']==track_id)].copy()
            hit_cnt = 0
            hit_force = True
            if len(selected)>0:
                if bbox_size == 0:
                    g = selected.apply(lambda track: Polygon([(track['x'], track['y']), (track['x'] + track['w'], track['y']), 
                                (track['x'] + track['w'], track['y'] + track['h']), (track['x'], track['y'] + track['h'])]), axis =1)
                else:
                    g = selected.apply(lambda track: Polygon([
                        (track['x'] + track['w']/2 - bbox_size, track['y'] + track['h'] - bbox_size),
                        (track['x'] + track['w']/2 + bbox_size, track['y'] + track['h'] - bbox_size), 
                        (track['x'] + track['w']/2 + bbox_size, track['y'] + track['h']),
                        (track['x'] + track['w']/2 - bbox_size, track['y'] + track['h'])
                        ]), axis =1)

                for line in lines:
                    if any(line.intersects(g).values.tolist()):
                        hit_cnt+=1
                            
                if force_line_indexes is not None:
                    force_lines = [lines[i] for i in force_line_indexes]
                    for line in force_lines:
                        if any(line.intersects(g).values.tolist()):
                            hit_force = True
                        else:
                            hit_force = False

                if (hit_cnt >= hit_criterion) and (hit_force == True):
                    ids.append(track_id)

            pbar.update()

        pbar.close()

        results = []
        if method=='include':    
            results = tracks.loc[tracks['track'].isin(ids)].copy()
        elif method=='exclude':
            results = tracks.loc[~tracks['track'].isin(ids)].copy()

        results.sort_values(by=['frame', 'track'], inplace=True)
        return results
    
if __name__=='__main__':
    pass

    