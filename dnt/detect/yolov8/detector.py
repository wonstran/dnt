from ultralytics import YOLO, RTDETR, NAS, YOLOWorld
import cv2
import pandas as pd
from tqdm import tqdm
import random, os
from pathlib import Path

class Detector:
    def __init__(self, model:str='yolo', weights:str='x', conf:float=0.25, nms:float=0.7, max_det:int=300):
        '''
        model: 'yolo' (default), 'redetr'
        weights: 'x' - extra-large (default), 'l' - large, 'm' - mdium, 's' - small, 'n' - nano, '.pt' - custom model weights
        '''
        # Load YOLOV8 model
        cwd = Path(__file__).parent.absolute()
        if model == 'yolo':
            if weights in ['x', 'l', 'm', 's', 'n']:
                model_path = os.path.join(cwd, 'models/yolov8'+weights+'.pt')
            elif ".pt" in weights:
                model_path = os.path.join(cwd, 'models/'+weights)
        elif model == 'rtdetr':
            if weights in ['x', 'l']:
                model_path = os.path.join(cwd, 'models/rtdetr-'+weights+'.pt')
            elif ".pt" in weights:
                model_path = os.path.join(cwd, 'models/'+weights)
            else:
                model_path = os.path.join(cwd, 'models/rtdetr-x.pt')
        #elif model == 'nas':
        #    if weights in ['l', 'm', 's']:
        #        model_path = os.path.join(cwd, 'models/yolo_nas_'+weights+'.pt')
        #    elif ".pt" in weights:
        #        model_path = os.path.join(cwd, 'models/'+weights)
        #    else:
        #        model_path = os.path.join(cwd, 'models/yolo_nas_l.pt')
        else:
            raise Exception('Invalid detection model type!')
        
        if model == 'yolo':
            self.model = YOLO(model_path)
        elif model == 'rtdetr':
            self.model = RTDETR(model_path)
        #elif model == 'nas':
        #    self.model = NAS(model_path)
        else:
            raise Exception('Invalid model weights!')  
        
        self.conf = conf
        self.nms = nms
        self.max_det = max_det

    def detect(self, input_video:str, iou_file:str=None,
               video_index:int=None, video_tot:int=None,
               start_time:int=None, end_time:int=None, verbose:bool=False) -> pd.DataFrame:
     
        # open input video
        cap = cv2.VideoCapture(input_video)
        if not cap.isOpened():
            print('Failed to open the video!')

        random.seed(3)  # deterministic bbox colors
        results = []

        video_fps = int(cap.get(cv2.CAP_PROP_FPS))                    #original fps
        if start_time:
            start_frame = int(video_fps * start_time)
            if start_frame > int(cap.get(cv2.CAP_PROP_FRAME_COUNT))-1:
                start_frame = 0    
        else:
            start_frame = 0

        if end_time:
            end_frame = int(video_fps * end_time)
            if end_frame > int(cap.get(cv2.CAP_PROP_FRAME_COUNT))-1:
                end_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))-1    
        else:
            end_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))-1

        if start_frame>=end_frame:
            raise Exception('The given start time exceeds the end time!')
        
        frame_total = end_frame - start_frame      
        pbar = tqdm(total=frame_total, unit=" frames")
        if video_index and video_tot: 
            pbar.set_description_str("Detecting {} of {}".format(video_index, video_tot))
        else:
            pbar.set_description_str("Detecting ")

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        while cap.isOpened():
            pos_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            ret, frame = cap.read()
            if (not ret) or (pos_frame>end_frame):
                break
            
            detects = self.model.predict(frame, verbose=False, conf=self.conf, iou=self.nms, max_det=self.max_det)   
            if len(detects)>0:
                detect = detects[0]    

                xyxy = pd.DataFrame(detect.boxes.xyxy.tolist(), columns=['x', 'y', 'x2', 'y2'])
                cls = pd.DataFrame(detect.boxes.cls.tolist(), columns=['class'])
                conf = pd.DataFrame(detect.boxes.conf.tolist(), columns = ['conf'])
                result = pd.concat([xyxy, conf, cls], axis=1)
                result.insert(0, 'frame', pos_frame)
                result.insert(1, 'res', -1)
                results.append(result)

            pbar.update()

        pbar.close()
        cap.release()
        #cv2.destroyAllWindows()

        df = pd.concat(results)
        df['x'] = round(df['x'], 1)
        df['y'] = round(df['y'], 1)
        df['w'] = round(df['x2']-df['x'], 0)
        df['h'] = round(df['y2']-df['y'], 0)
        df['conf'] = round(df['conf'], 2)
        df = df[['frame', 'res', 'x', 'y', 'w', 'h', 'conf', 'class']].reindex()

        if iou_file:
            df.to_csv(iou_file, index=False, header=False)
            if verbose:
                print("Wrote to {}".format(iou_file))
        
        return df

    def detect_v8(self, input_video:str, iou_file:str=None, save:bool=False, verbose:bool=False, show:bool=False):
        detects = self.model.predict(input_video, 
                        verbose=verbose, stream=True, save=save, show=show)
        
        if iou_file:
            results = []
            frame = 0
            for detect in detects:
    
                xywh = pd.DataFrame(detect.boxes.xywh.tolist(), columns=['x', 'y', 'w', 'h'])
                cls = pd.DataFrame(detect.boxes.cls.tolist(), columns=['class'])
                conf = pd.DataFrame(detect.boxes.conf.tolist(), columns = ['conf'])

                result = pd.concat([xywh, conf, cls], axis=1)
                result.insert(0, 'frame', frame) 
                result.insert(1, 'revserve', -1) 

                result['x'] = result['x']-result['w']/2
                result['y'] = result['y']+result['h']/2

                frame += 1

                results.append(result)

            df = pd.concat(results)
            df.to_csv(iou_file, index=False, header=False)

    def detect_batch(self, input_videos:list[str], output_path:str=None, is_overwrite:bool=False, 
                     is_report:bool=True, verbose:bool=False) -> list[str]:
    
        results = []
        total_videos = len(input_videos)
        video_count=0
        for input_video in input_videos:
            video_count+=1

            base_filename = os.path.splitext(os.path.basename(input_video))[0]
            raw_video = input_video #os.path.join(input_path, base_filename+".mp4")

            if verbose:
                print("Processing {} of {} - {}           ".format(video_count, total_videos, raw_video))
            

            if output_path:
                if not os.path.exists(output_path):
                    os.mkdir(output_path)

                iou_file = os.path.join(output_path, base_filename+"_iou.txt")

            if not is_overwrite:
                if os.path.exists(iou_file):
                    if is_report:
                        results.append(iou_file)
                    continue 

            self.detect(input_video=raw_video, iou_file=iou_file, 
                        video_index=video_count, video_tot=total_videos,
                        verbose=verbose)

            results.append(iou_file)

        return results

    @staticmethod
    def get_fps(video:str)->float:
        cap = cv2.VideoCapture(video)
        if not cap.isOpened():
            print('Failed to open the video!')
        fps = cap.get(cv2.CAP_PROP_FPS)
        return fps
    
    @staticmethod
    def get_frames(video:str)->int:
        cap = cv2.VideoCapture(video)
        if not cap.isOpened():
            print('Failed to open the video!')
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return frames

if __name__=='__main__':

    detector = Detector()
    detector.detect('/mnt/d/videos/samples/traffic_short.mp4', '/mnt/d/videos/samples/traffic_short_iou.txt')
