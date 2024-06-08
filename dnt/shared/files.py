from email import header
import pandas as pd

def read_iou(iou_file):
    results = pd.read_csv(iou_file, header=None, dtype={0:int, 1:int, 2:float, 3:float, 4:float, 5:float, 6:float, 7:int})
    return results

def write_iou(ious, iou_file):
    ious.to_csv(iou_file, header=False, index=False)

def read_track(track_file):
    results = pd.read_csv(track_file, header=None, dtype={0:int, 1:int, 2:float, 3:float, 4:float, 5:float, 6:float, 7:int, 8:int, 9:int})
    return results

def read_spd(spd_file):
    '''
        columns: vid, reference, frame
    '''
    results = pd.read_csv(spd_file, header=None, dtype={0:int, 1:int, 2:int})
