import pandas as pd
from shapely import geometry
import geopandas as gpd

def filter_iou(detections, zones = [], class_list = [], score_threshold = 0):

    detections = detections.loc[detections[6]>=score_threshold].copy()

    # filter classess        
    if class_list:
        detections = detections.loc[detections[7].isin(class_list)].copy()

    if zones:
        # filter locations
        g = [geometry.Point(xy) for xy in zip((detections[2] + detections[4]/2), (detections[3] + detections[5]/2))]
        geo_detections = gpd.GeoDataFrame(detections, geometry=g)

        frames = []
        for zone in zones:    
            geo_zones = geometry.Polygon(zone)
            frames.append(geo_detections.loc[geo_detections.geometry.within(geo_zones)].drop(columns='geometry'))

        if frames:
            results = pd.concat(frames)
            results = results[~results.index.duplicated()].reset_index(drop=True)
        else:
            results = pd.DataFrame()
    else:
        results = detections

    return results
    

def filter_class_iou(iou_file, out_file, class_list=[], score_threshold=0):
    detections = pd.read_csv(iou_file, header=None)
    if class_list:
        detections = detections.loc[(detections[7].isin(class_list)) & (detections[6]>=score_threshold)].copy().reindex()
    detections.to_csv(out_file, header=False, index=False)