import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import os

class Labeler:
    def __init__(self, method:str='opencv', frame_field:int=0,
             label_fields:list[int]=[1], color_field:int=1, zoom_factor:int=1, line_thickness:int=1,
             color_bgr = (0, 255, 0), compress_message:bool=False, nodraw_empty:bool=True):
        
        self.method = method
        self.frame_field = frame_field
        self.label_fields=label_fields
        self.color_field=color_field
        self.zoom_factor=zoom_factor 
        self.line_thickness=line_thickness
        self.color_bgr = color_bgr
        self.compress_message=compress_message
        self.nodraw_empty = nodraw_empty
    
    def draw2(self, input_video:str, output_video:str, draws:pd.DataFrame = None, draw_file:str = None,
            start_frame:int=None, end_frame:int=None, 
            video_index:int=None, video_tot:int=None):
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
                    
            writer.write(frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            
            pbar.update()

        cv2.destroyAllWindows()
        cap.release()
        writer.release()

    def draw(self, label_file:str, input_video:str, output_video:str, start_frame:int=None, end_frame:int=None, 
             video_index:int=None, video_tot:int=None):
        tracks = pd.read_csv(label_file, header=None)
        
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
            
            #boxes = tracks.query('@tracks[0]==2').values.tolist()
            boxes = tracks[tracks.iloc[:,self.frame_field]==pos_frame].values.tolist()
            #boxes = tracks.loc[tracks.columns[0]==pos_frame].values.tolist()
            
            for box in boxes:
                x1 = int(box[2])
                y1 = int(box[3])
                x2 = x1 + int(box[4])
                y2 = y1 + int(box[5])

                color = colors[int(box[self.color_field]) % len(colors)]
                color = [i * 255 for i in color]
                
                label_txt = ''
                for field in self.label_fields:
                    if (str(box[field]).strip() == '-1'):
                        if self.nodraw_empty:
                            label_txt += ''
                        else:
                            label_txt += str(box[field]) + ' '
                    else:
                        label_txt += str(box[field]) + ' '
                        
                label_txt = label_txt.strip()

                if label_txt:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, self.line_thickness)
                    cv2.rectangle(frame, (x1, int(y1-30*self.zoom_factor)), (x1+len(label_txt)*int(17*self.zoom_factor), y1), 
                              color, -1)
                    cv2.putText(frame,str(label_txt),(int(x1), int(y1-int(10*self.zoom_factor))), 
                            0, 0.75*self.zoom_factor, (255,255,255), 1)

            writer.write(frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            
            pbar.update()

        cv2.destroyAllWindows()
        cap.release()
        writer.release()

    
    def draw_lines(self, lines:list[list[int]], input_video:str, output_video:str, start_frame:int=None, 
                   end_frame:int=None, video_index:int=None, video_tot:int=None):
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

            for line in lines:
                cv2.line(frame, (int(line[0]), int(line[1])), (int(line[2]), int(line[3])), self.color_bgr, self.line_thickness)

            writer.write(frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            
            pbar.update()

        cv2.destroyAllWindows()
        cap.release()
        writer.release()

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
            pbar.set_description_str("Labeling")
        else:     
            pbar.set_description_str("Labeling {} ".format(input_video))

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

    def draw_batch(self, label_files:list[str], input_videos:list[str], output_path:str, 
                   suffix:str='_track', is_overwrite:bool=False)->list[str]:
        results = []
        total_videos = len(label_files)
        count=0
        for label_file in label_files:
            count+=1

            base_filename = os.path.splitext(os.path.basename(label_file))[0].replace("_track","")
            raw_video = input_videos[count-1] #os.path.join(input_path, base_filename+".mp4")           

            if output_path:
                if not os.path.exists(output_path):
                    os.mkdir(output_path)

                if suffix:
                    output_video = os.path.join(output_path, base_filename+suffix+".mp4") 
                else:
                    output_video = os.path.join(output_path, base_filename+".mp4")
                

            if not is_overwrite:
                if os.path.exists(output_video):
                    continue 

            self.draw(label_file=label_file, input_video=raw_video, output_video=output_video, video_index=count, video_tot=total_videos)

            results.append(output_video)

        return results
    
    def draw_lines_batch(self, label_files:list[str], input_videos:list[str], output_path:str, suffix:str, 
                         is_overwrite:bool=False, is_report:bool=True)->list[str]:
        results = []
        total_videos = len(label_files)
        count=0
        for label_file in label_files:
            count+=1

            base_filename = os.path.splitext(os.path.basename(label_file))[0]
            raw_video = input_videos[count-1]         

            if output_path:
                if not os.path.exists(output_path):
                    os.mkdir(output_path)

                if suffix:
                    output_video = os.path.join(output_path, base_filename+suffix+".mp4")
                else:
                    output_video = os.path.join(output_path, base_filename+".mp4")

            if not is_overwrite:
                if os.path.exists(output_video):
                    if is_report:
                        results.append(output_video)

                    continue 

            self.draw_lines(label_file=label_file, input_video=raw_video, output_video=output_video, video_index=count, video_tot=total_videos)

            results.append(output_video)

        return results
    
    def draw_shapes(self, input_video:str, output_video:str, 
                    points:list[list[int]]=None, 
                    lines:list[list[int]]=None, 
                    polygons:list[np.array]=None, 
                    point_color:tuple=None, line_color:tuple=None, polygon_color:tuple=None, 
                    start_frame:int=None, end_frame:int=None, video_index:int=None, video_tot:int=None):
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
            
            if points is not None:
                if point_color is None:
                    point_color = self.color_bgr
                for point in points:
                    cv2.circle(frame, (point[0], point[1]), radius=self.line_thickness, color=point_color, thickness=-1)

            if lines is not None:
                if line_color is None:
                    line_color = self.color_bgr
                for line in lines:
                    cv2.line(frame, (line[0], line[1]), (line[2], line[3]), line_color, self.line_thickness)
            
            if polygons is not None:
                if polygon_color is None:
                    polygon_color = self.color_bgr
                for polygon in polygons:
                    pts = np.array(polygon, np.int32)
                    pts = pts.reshape((-1, 1, 2))
                    cv2.polylines(frame, [pts], isClosed=True, color=polygon_color, thickness=self.line_thickness)

            writer.write(frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            
            pbar.update()

        cv2.destroyAllWindows()
        cap.release()
        writer.release()
    
    def draw_tracks(self, track_file:str, 
                    input_video:str, 
                    output_video:str, 
                    tail:int=0, 
                    start_frame:int=None, end_frame:int=None, 
                    video_index:int=None, video_tot:int=None):
    
        tracks = pd.read_csv(track_file, header=None)
        
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
                
            boxes = tracks[tracks.iloc[:,self.frame_field]==pos_frame].values.tolist()  
            for box in boxes:
                track_id = box[1]

                x1 = int(box[2])
                y1 = int(box[3])
                x2 = x1 + int(box[4])
                y2 = y1 + int(box[5])

                color = colors[int(box[self.color_field]) % len(colors)]
                color = [i * 255 for i in color]
                    
                label_txt = ''
                for field in self.label_fields:
                    if (str(box[field]).strip() == '-1'):
                        if self.nodraw_empty:
                            label_txt += ''
                        else:
                            label_txt += str(box[field]) + ' '
                    else:
                        label_txt += str(box[field]) + ' '
                            
                    label_txt = label_txt.strip()

                if tail>0:
                    frames = [*range(pos_frame-tail, pos_frame)]
                    pre_boxes = tracks.loc[(tracks[self.frame_field].isin(frames)) & (tracks[1]==track_id)].values.tolist()   
                    if len(pre_boxes)>0:
                        for pre_box in pre_boxes:
                            xc = int(pre_box[2]) + int(pre_box[4]/2)
                            yc = int(pre_box[3]) + int(pre_box[5]/2)
                            cv2.circle(frame, (xc, yc), radius=0, color=color,  thickness=self.line_thickness)

                if label_txt:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, self.line_thickness)
                    cv2.rectangle(frame, (x1, int(y1-30*self.zoom_factor)), (x1+len(label_txt)*int(17*self.zoom_factor), y1), 
                                color, -1)
                    cv2.putText(frame,str(label_txt),(int(x1), int(y1-int(10*self.zoom_factor))), 
                                0, 0.75*self.zoom_factor, (255,255,255), 1)

            writer.write(frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
                
            pbar.update()

        cv2.destroyAllWindows()
        cap.release()
        writer.release()
    
    @staticmethod
    def export_frames(input_video:str, frames:list[int], output_path:str):
    
        cap = cv2.VideoCapture(input_video)
        if not cap.isOpened():
            raise IOError("Couldn't open webcam or video")

        pbar = tqdm(total=len(frames), unit=" frames")
        pbar.set_description_str("Extracting frame")

        for frame in frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
            ret, frame_read = cap.read()

            frame_file = os.path.join(output_path, str(frame)+'.jpg')
            if ret:
                cv2.imwrite(frame_file, frame_read)            
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
    
if __name__=='__main__':
    video_file = "/mnt/d/videos/hfst/Standard_SCU7WH_2022-09-16_0630.02.001.mp4"    
    iou_file = "/mnt/d/videos/hfst/Standard_SCU7WH_2022-09-16_0630.02.001_iou.txt"
    track_file = "/mnt/d/videos/hfst/tracks/Standard_SCU7WH_2022-09-16_0630.02.001_track.txt"
    label_video = "/mnt/d/videos/hfst/labels/Standard_SCU7WH_2022-09-16_0630.02.001_track.mp4"
    label_file = "/mnt/d/videos/hfst/tracks/Standard_SCU7WH_2022-09-16_0630.02.001_label.txt"

    labeler = Labeler(video_file, zoom_factor=0.5, nodraw_empty=True, label_fields=[6])
    labeler.draw(label_file, video_file, label_video)

