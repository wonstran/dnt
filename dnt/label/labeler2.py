import os, sys
sys.path.append(os.path.dirname(__file__))

import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import itertools
from ..shared.util import load_classes
import random

class Labeler:
    def __init__(self, method:str='opencv', compress_message:bool=False, nodraw_empty:bool=True):
        
        self.method = method
        self.compress_message=compress_message
        self.nodraw_empty = nodraw_empty
    
    def draw(self, input_video:str, output_video:str, 
            draws:pd.DataFrame = None, draw_file:str = None,
            start_frame:int=None, end_frame:int=None, 
            video_index:int=None, video_tot:int=None, verbose:bool=True):
        '''
        General labeling function
        Inputs:
                draws: a DataFrame contains labeling information, if None, read label_file
                label_file: a txt file with a header ['frame','type','coords','color','size','thick','desc']
                input_video: raw video
                output_video: labeled video
                start_frame:
                end_frame:
                video_index: display video index in batch processing
                video_tot: display total video number in batch processing
        '''
        if draws is not None:
            data = draws
        else:
            data = pd.read_csv(draw_file, dtype={'frame':int, 'type':str, 'size':float, 'desc':str, 'thick':int}, 
                    converters={'coords': lambda x:list(eval(x)), 'color': lambda x:eval(x)})

        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

        cap = cv2.VideoCapture(input_video)
        if not cap.isOpened():
            raise IOError("Couldn't open webcam or video")

        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))-1

        tot_frames = end_frame - start_frame + 1       
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

        if verbose:
            pbar = tqdm(total=tot_frames, unit=" frames")
            if self.compress_message:
                pbar.set_description_str("Labeling")
            else:
                if video_index and video_tot:
                    pbar.set_description_str("Labeling {} of {}".format(video_index, video_tot))
                else:
                    pbar.set_description_str("Labeling {} ".format(input_video))

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        while cap.isOpened():
            pos_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            ret, frame = cap.read()
            if (not ret) or (pos_frame>end_frame):
                break
            
            elements = data.loc[data['frame']==pos_frame]

            for index, element in elements.iterrows():
                if element['type'] == 'txt':
                    label_txt = element['desc']
                    color = element['color']
                    size = element['size']
                    thick = element['thick']
                    cv2.putText(frame, label_txt, tuple(map(int, element['coords'][0])), 0, size, color, thick)
                
                elif element['type'] == 'line':
                    coords = element['coords']
                    color = element['color']
                    thick = element['thick']
                    cv2.line(frame, tuple(map(int, coords[0])), tuple(map(int, coords[1])), color, thick)

                elif element['type'] == 'box':
                    coords = element['coords']
                    color = element['color']
                    thick = element['thick']
                    cv2.rectangle(frame, tuple(map(int, coords[0])), tuple(map(int, coords[1])), color, thick)
                
                elif element['type'] == 'bbox':
                    coords = element['coords']
                    color = element['color']
                    thick = element['thick']
                    label_txt = element['desc']
                    size = element['size'] 

                    cv2.rectangle(frame, tuple(map(int, coords[0])), tuple(map(int, coords[1])), color, thick)
                    cv2.putText(frame, str(label_txt), (int(coords[0][0]), int(coords[0][1]-int(10*size))), 
                            cv2.FONT_HERSHEY_SIMPLEX, size, color, thick)
                    
                elif element['type'] == 'circle':
                    coords = element['coords']
                    color = element['color']
                    thick = element['thick']
                    label_txt = element['desc']
                    radius = int(element['size']) 
                    
                    cv2.circle(frame, tuple(map(int, coords[0])), radius=radius, color=color,  thickness=thick)
                
                elif element['type'] == 'polygon':
                    coords = element['coords']
                    color = element['color']
                    thick = element['thick']
                    cv2.polylines(frame, [np.array(coords)], isClosed=True, color=color, thickness=thick)

                elif element['type'] == 'polylines':
                    coords = element['coords']
                    color = element['color']
                    thick = element['thick']
                    cv2.polylines(frame, [np.array(coords)], isClosed=False, color=color, thickness=thick)

            writer.write(frame)
            #key = cv2.waitKey(1) & 0xFF
            #if key == ord("q"):
            #    break
            
            if verbose:
                pbar.update()

        if verbose:
            pbar.close()
        #cv2.destroyAllWindows()
        cap.release()
        writer.release()      

    def draw_track_clips(self, input_video:str, output_path:str,
                    tracks:pd.DataFrame = None, track_file:str = None,
                    method:str='all', random_number:int=10, track_ids:list=None,
                    tail:int=0, prefix:bool=False,
                    size:int=1, thick:int=1,
                    verbose:bool=True):
        '''
        Parameters:
            input_video: the raw video file
            outputh_path: the folder for outputing track clips
            tracks: the dataframe of tracks
            track_file: the track file if tracks are none
            method: 'all' (default) - all tracks, 'random' - random select tracks, 'specify' - specify track ids
            random_number: the number of track ids if method == 'random'
            track_ids: the list of track ids if method == 'specify'
            tail: the length of tail, default is 0
            prefix: if add the video file name as the prefix in output file names, default is False
            size: font size, default is 1
            thick: line thinckness, defualt is 1
            verbose: if show progressing bar, default is True
        '''

        if tracks is None:
            tracks = pd.read_csv(track_file, header=None, dtype={0:int, 1:int, 2:int, 3:int, 4:int, 5:int, 6:float, 7:int, 8:int, 9:int})
        tracks.columns = ['frame', 'track', 'x', 'y', 'w', 'h', 'score', 'cls', 'r3', 'r4']

        if method == 'random':
            track_ids = tracks['track'].unique().tolist()
            if random_number<=0:
                random_number = 10
            track_ids = random.sample(track_ids, random_number)
        elif method == 'specify':
            if (track_ids is None) or (len(track_ids)==0):
                print('No tracks are provided!')
                return pd.DataFrame()        
        else:
            track_ids = tracks['track'].unique().tolist()

        pbar = tqdm(total=len(track_ids), desc='Labeling tracks ', unit='videos')
        for id in track_ids:
            selected_tracks = tracks[tracks['track']==id].copy()
            start_frame = selected_tracks['frame'].min()
            end_frame = selected_tracks['frame'].max()
            if prefix:
                out_video = os.path.join(output_path, os.path.splitext(os.path.basename(input_video))[0]+"_"+str(id)+'.mp4')
            else:
                out_video = os.path.join(output_path, str(id)+'.mp4')

            self.draw_tracks(input_video=input_video, output_video=out_video, tracks=selected_tracks, 
                             start_frame=start_frame, end_frame=end_frame, verbose=False, tail=tail, thick=thick, size=size)
            pbar.update()
        pbar.close()

    def draw_tracks(self, input_video:str, output_video:str,
                    tracks:pd.DataFrame = None, track_file:str = None, label_file:str=None, 
                    color = None, tail:int=0, thick:int=2, size:int=1,
                    class_name = False, 
                    start_frame:int=None, end_frame:int=None, 
                    video_index:int=None, video_tot:int=None, verbose:bool=True):

        if tracks is None:
            tracks = pd.read_csv(track_file, header=None, dtype={0:int, 1:int, 2:int, 3:int, 4:int, 5:int, 6:float, 7:int, 8:int, 9:int})
        tracks.columns = ['frame', 'track', 'x', 'y', 'w', 'h', 'score', 'cls', 'r3', 'r4']

        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

        cap = cv2.VideoCapture(input_video)
        if not cap.isOpened():
            raise IOError("Couldn't open webcam or video")

        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))-1

        tot_frames = end_frame - start_frame + 1       

        if verbose:
            pbar = tqdm(total=tot_frames, unit=" frames")
            if self.compress_message:
                    pbar.set_description_str("Generating labels")
            else:
                if video_index and video_tot:
                    pbar.set_description_str("Generating labels {} of {}".format(video_index, video_tot))
                else:
                    pbar.set_description_str("Generating labels {} ".format(input_video))

        selected_tracks = tracks.loc[(tracks['frame']>=start_frame) & (tracks['frame']<=end_frame)].copy()

        results = []
        for index, track in selected_tracks.iterrows():

            if color is None:
                final_color = colors[int(track['track']) % len(colors)]
                final_color = [i * 255 for i in final_color]
            else:
                final_color = color

            if class_name == True:
                label_str = str(int(track['track'])) + ' ' + str(int(track['cls']))
            else:
                label_str = str(int(track['track']))
            results.append([track['frame'], 'bbox', [(track['x'], track['y']), (track['x'] + track['w'], track['y'] + track['h'])], 
                            final_color, size, thick, label_str])
            if tail>0:
                frames = [*range(int(track['frame'])-tail, int(track['frame']))]
                pre_boxes = tracks.loc[(tracks['frame'].isin(frames)) & (tracks['track']==track['track'])].values.tolist()
                
                if len(pre_boxes)>0:
                    for pre_box in pre_boxes:
                        xc = int(pre_box[2]) + int(pre_box[4]/2)
                        yc = int(pre_box[3]) + int(pre_box[5]/2)
                        results.append([track['frame'], 'circle', [(xc, yc)], 
                            final_color, 0, -1, ''])

            if verbose:
                pbar.update()
        
        if verbose:
            pbar.close()

        results.sort()
        results = list(results for results,_ in itertools.groupby(results))
        df = pd.DataFrame(results, columns=['frame','type','coords','color','size','thick','desc'])
        df.sort_values(by='frame', inplace=True)

        if output_video:
            self.draw(input_video = input_video, output_video = output_video, 
                        draws = df, start_frame = start_frame, end_frame = end_frame, 
                        video_index = video_index, video_tot = video_tot, verbose=verbose)

        if label_file:
            df.to_csv(label_file, index=False)

        return df
    
    def draw_dets(self, input_video:str, output_video:str,
                    dets:pd.DataFrame = None, det_file:str = None, label_file:str=None, 
                    color = None, class_name = False,
                    start_frame:int=None, end_frame:int=None, 
                    video_index:int=None, video_tot:int=None):

        if dets is None:
            dets = pd.read_csv(det_file, header=None)

        names = load_classes()

        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

        cap = cv2.VideoCapture(input_video)
        if not cap.isOpened():
            raise IOError("Couldn't open webcam or video")

        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))-1

        tot_frames = end_frame - start_frame + 1       

        pbar = tqdm(total=len(dets), unit=" dets")
        if self.compress_message:
                pbar.set_description_str("Generating labels")
        else:
            if video_index and video_tot:
                pbar.set_description_str("Generating labels {} of {}".format(video_index, video_tot))
            else:
                pbar.set_description_str("Generating labels {} ".format(input_video))

        selected_dets = dets.loc[(dets[0]>=start_frame) & (dets[0]<=end_frame)].copy()

        results = []
        for index, det in selected_dets.iterrows():

            if color is None:
                final_color = colors[int(det[7]) % len(colors)]
                final_color = [i * 255 for i in final_color]
            else:
                final_color = color

            if class_name == True:
                desc = names[int(det[7])]
            else:
                desc = str(int(det[7]))

            results.append([det[0], 'bbox', [(det[2], det[3]), (det[2]+det[4], det[3]+det[5])], 
                            final_color, 0.8, 1, desc])
            pbar.update()
        
        results.sort()
        results = list(results for results,_ in itertools.groupby(results))
        df = pd.DataFrame(results, columns=['frame','type','coords','color','size','thick','desc'])
        df.sort_values(by='frame', inplace=True)

        if output_video:
            self.draw(input_video = input_video, output_video = output_video, 
                        draws = df, start_frame = start_frame, end_frame = end_frame, 
                        video_index = video_index, video_tot = video_tot)

        if label_file:
            df.to_csv(label_file, index=False)

        return df

    def clip(self, input_video:str, output_video:str, start_frame:int=None, end_frame:int=None):
        cap = cv2.VideoCapture(input_video)
        if not cap.isOpened():
            raise IOError("Couldn't open webcam or video")

        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))-1

        tot_frames = end_frame - start_frame + 1       
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

        pbar = tqdm(total=tot_frames, unit=" frames")
        if self.compress_message:
            pbar.set_description_str("Cutting")
        else:     
            pbar.set_description_str("Cutting {} ".format(input_video))

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        while cap.isOpened():
            pos_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            ret, frame = cap.read()
            if (not ret) or (pos_frame>end_frame):
                break

            writer.write(frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            
            pbar.update()

        cv2.destroyAllWindows()
        cap.release()
        writer.release()

    @staticmethod
    def export_frames(input_video:str, frames:list[int], output_path:str, prefix:str=None):
    
        cap = cv2.VideoCapture(input_video)
        if not cap.isOpened():
            raise IOError("Couldn't open webcam or video")

        pbar = tqdm(total=len(frames), unit=" frames")
        pbar.set_description_str("Extracting frame")

        for frame in frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
            ret, frame_read = cap.read()

            if prefix is None:
                frame_file = os.path.join(output_path, str(frame)+'.jpg')
            else:
                frame_file = os.path.join(output_path, prefix+'-'+str(frame)+'.jpg')

            if ret:
                cv2.imwrite(frame_file, frame_read)            
            else:
                break
        
            pbar.update()
    
        pbar.close()
        cap.release()
    
        print("Writing frames to {}".format(output_path))

    @staticmethod
    def export_track_frames(input_video:str, tracks:pd.DataFrame, output_path:str, bbox = True, prefix:str=None, thick:int=2):
        
        if (tracks is None) or (len(tracks.columns)<10):
            raise Exception("Invalid tracks!")
        tracks.columns = ['frame', 'track', 'x', 'y', 'w', 'h', 'score', 'cls', 'r3', 'r4']
        ids = tracks['track'].unique()

        cap = cv2.VideoCapture(input_video)
        if not cap.isOpened():
            raise IOError("Couldn't open webcam or video")

        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

        pbar = tqdm(total=len(ids), unit=' frame')
        for id in ids:
            pbar.desc = "Extracting track: "+str(id)
            selected = tracks[tracks['track']==id]
            if len(selected) > 0:
                for index, track in selected.iterrows():
                    frame = track['frame']
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
                    ret, img = cap.read()

                    if ret:
                        if bbox == True:
                            x1 = track['x']
                            y1 = track['y']
                            x2 = track['x'] + track['w']
                            y2 = track['y'] + track['h']
                            final_color = colors[int(id) % len(colors)]
                            final_color = [i * 255 for i in final_color]
                            cv2.rectangle(img, (int(x1), int(y1) ), (int(x2), int(y2)), final_color, thick)
                            
                            if prefix is None:
                                frame_file = os.path.join(output_path, str(id)+'_'+str(frame)+'.jpg')
                            else:
                                frame_file = os.path.join(output_path, prefix+'-'+str(id)+'_'+str(frame)+'.jpg')

                            cv2.imwrite(frame_file, img)            
                    else:
                        break

                pbar.update()
        
            pbar.close()
        cap.release()
    
        print("Writing frames to {}".format(output_path))

    @staticmethod
    def time2frame(input_video:str, time:float):
        cap = cv2.VideoCapture(input_video)
        if not cap.isOpened():
            raise IOError("Couldn't open webcam or video")

        video_fps = int(cap.get(cv2.CAP_PROP_FPS))                    #original fps
        frame = int(video_fps * time)
        return frame

class LabelGenerator():
    def __init__(self) -> None:
        self.draws = []

    
if __name__=='__main__':
    video_file = "/mnt/d/videos/hfst/Standard_SCU7WH_2022-09-16_0630.02.001.mp4"    
    iou_file = "/mnt/d/videos/hfst/Standard_SCU7WH_2022-09-16_0630.02.001_iou.txt"
    track_file = "/mnt/d/videos/hfst/tracks/Standard_SCU7WH_2022-09-16_0630.02.001_track.txt"
    label_video = "/mnt/d/videos/hfst/labels/Standard_SCU7WH_2022-09-16_0630.02.001_track.mp4"
    label_file = "/mnt/d/videos/hfst/tracks/Standard_SCU7WH_2022-09-16_0630.02.001_label.txt"

    labeler = Labeler(video_file, zoom_factor=0.5, nodraw_empty=True, label_fields=[6])
    labeler.draw(label_file, video_file, label_video)
