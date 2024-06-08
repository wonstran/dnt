import pandas as pd

def frame_reflines(tracks, ref_lines, tolerance):
    '''
        tracks - a dataframe contains vehicle tracks without header  
                frame, veh_id, leftup_x, leftup_y, w, h
        ref_lineas - a list of ref_lines
                [(pt1_x, pt1_y, pt2_x, pt2_y), (....), ...]
        tolerance - allowed error to intersect (reserved) 

        *************************************************************
        return - a dataframe contains frames passing reference lines
                veh_id, ref_line_index, frame
    '''

