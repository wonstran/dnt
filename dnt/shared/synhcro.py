import pandas as pd
from datetime import datetime
from pytz import timezone
from dnt.detect import Detector
from tqdm import tqdm
import os

class Synchronizer():
    def __init__(self, videos:list[str], ref_frame:int, ref_time:int, ref_timezone:str='US/Eastern', offsets:list[int]=None) -> None:
        self.videos = videos
        self.ref_frame = ref_frame
        self.ref_time = ref_time
        self.ref_timezone = timezone(ref_timezone)
        self.offsets = offsets 

    def process(self, output_path:str=None, local:bool=False, message:bool=False)->pd.DataFrame:
        '''
        Add unix time stamp to each frame
        Inputs:
            output_path - Output folder for output files, if None, will not produce files
            local - if generate local time string
        Returns:
            a Dataframe contains frame and timestamps ['frame', 'unix_time', 'video', 'local_time'(if local is True)]
        '''
        ref_frame = self.ref_frame
        ref_time = self.ref_time
        offsets = [0] * len(videos)
        if self.offsets:
           offsets = self.offsets

        results = []
        cnt = 0
        for video in self.videos:
            
            if fps <= 0:
                raise Exception("fps is invalid.")
            self.milliseconds_per_frame = 1/self.fps * 1000 

            df = Synchronizer.add_unix_time(video, ref_frame, ref_time, video_index=cnt+1, video_tot=len(self.videos), message=message)
            
            if local:
                df['local_time'] = df['unix_time'].apply(lambda x: Synchronizer.convert_unix_local(x))
            results.append(df)

            if output_path:
                basename = os.path.basename(video).split('.')[0]
                time_file = os.path.join(output_path, basename+'_time.csv')
                df.to_csv(time_file, index=False)

            cnt+=1
            if cnt<len(videos):
                fps = Detector.get_fps(video)
                if fps <= 0:
                    raise Exception('fps is invalid for {}'.format(video))
                milliseconds_per_frame = 1/fps * 1000
                ref_frame = 0
                ref_time = df['unix_time'].max() + offsets[cnt] * milliseconds_per_frame
                       
        return pd.concat(results)

    @staticmethod
    def add_unix_time_to_frame(frame:int, ref_frame:int, ref_time:int, ref_timezone:str='US/Eastern', 
                                fps:int=30, verbos=True, local=False) -> int:
        if fps <= 0:
            raise Exception("fps is invalid.")
        milliseconds_per_frame = 1/fps * 1000 

        dif_frame = frame - ref_frame
        return round(ref_time + dif_frame * milliseconds_per_frame)

    @staticmethod
    def add_unix_time_to_frames(frames:pd.DataFrame, ref_frame:int, ref_time:int, ref_timezone:str='US/Eastern', 
                                fps:int=30, verbos=True, local=False) -> None:

        if fps <= 0:
            raise Exception("fps is invalid.")
        milliseconds_per_frame = 1/fps * 1000  
        
        pbar = tqdm(total=frames, desc='Adding timestamp')
        for index, frame in frames.iterrows():
            dif_frame = frame['frame'] - ref_frame
            frame_time = round(ref_time + dif_frame * milliseconds_per_frame)
            frames.at[index, 'unix_time'] = frame_time
            if local:
                frames.at[index, 'local_time'] = Synchronizer.convert_unix_local(frame_time)

            if verbos:
                pbar.update()

        return frames

    @staticmethod
    def add_unix_time(video, ref_frame, ref_time, video_index:int=None, video_tot=None, message:bool=False)->pd.DataFrame:
        '''
        Add unit timestamp to each frame
        Inputs:
            video - video file name
            ref_frame - the reference frame
            ref_time - the unix time for the reference frame
            video_index - the index of video
            video_tot - the total number of videos
        '''
        fps = Detector.get_fps(video)
        if fps <= 0:
            raise Exception("fps is invalid for {}".format(video))
        milliseconds_per_frame = 1/fps * 1000
        frames = Detector.get_frames(video)       

        if message:
            if video_index and video_tot:
                pbar = tqdm(total=frames, desc='Adding unix time: {} of {}'.format(video_index, video_tot))
            else:
                pbar = tqdm(total=frames, desc='Adding unix time: {}'.format(video))

        results = []
        for frame in range(frames):
            dif_frame = frame - ref_frame
            frame_time = round(ref_time + dif_frame * milliseconds_per_frame)
            results.append([frame, frame_time, video])

            if message:
                pbar.update()

        df = pd.DataFrame(results, columns=['frame', 'unix_time', 'video'])
        return df

    @staticmethod
    def convert_unix_local(unix_time:int, ref_timezone:str='US/Eastern')->str:
        tz = timezone(ref_timezone)
        return str(datetime.fromtimestamp(unix_time/1000, tz).__str__())
    
if __name__ == '__main__':
    videos = ['/mnt/d/videos/ped2stage/1026_2/videos/gh011293.mp4', 
              '/mnt/d/videos/ped2stage/1026_2/videos/gh021293.mp4',
              '/mnt/d/videos/ped2stage/1026_2/videos/gh031293.mp4',
              '/mnt/d/videos/ped2stage/1026_2/videos/gh041293.mp4']
    
    ref_frame = 2
    ref_time = 1698331984441

    synchronizer = Synchronizer(videos, ref_frame, ref_time)
    df = synchronizer.process('/mnt/d/videos/ped2stage/time_synchro', local=True, message=True)
    df.to_csv('/mnt/d/videos/ped2stage/time_synchro/text2.csv', index=False)

    '''
    results = Synchronizer.add_unix_time(input_video, ref_frame, ref_time)
    results['local_time'] = results['unix_time'].apply(lambda x: Synchronizer.convert_unix_local(x))

    results.to_csv('/mnt/d/videos/ped2stage/time_synchro/text.csv', index=False)
    '''
