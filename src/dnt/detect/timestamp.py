"""Timestamp extraction module using OCR.

This module provides functionality to extract timestamps from images and videos
using optical character recognition (OCR). It includes the TimestampExtractor
class which can extract timestamps from:
- Static images
- Specific frames in video files
- The first readable frame in a video file
"""

from datetime import datetime

import cv2
import dateparser
import easyocr
import numpy as np


class TimestampExtractor:
    """A class for extracting timestamps from images and videos using OCR.

    Attributes
    ----------
    zone : np.ndarray, optional
        The coordinates of the timestamp area in the image.
    reader : easyocr.Reader
        The EasyOCR reader instance for text recognition.
    allowlist : str
        Characters to allow in the OCR process.

    Methods
    -------
    extract_timestamp(img: np.ndarray, gray: bool = False) -> datetime
        Extract timestamp from an image using OCR.
    extract_timestamp_video(video_path: str, frame: int = 0, gray: bool = False) -> datetime
        Extract timestamp from a specific frame in a video file.
    extract_timestamp_video_auto(video_path: str, gray: bool = False) -> tuple
        Extract timestamp from the first readable frame in a video file.

    """

    def __init__(self, zone: np.ndarray = None, allowlist="0123456789-:/"):
        """Initialize the TimestampExtractor.

        Args:
            zone (np.ndarray): The coordinates of the timestamp area in the image.
            allowlist (str): Characters to allow in the OCR process.

        """
        self.zone = zone
        self.reader = easyocr.Reader(["en"])
        self.allowlist = allowlist

    def extract_timestamp(
        self,
        img: np.ndarray,
        gray: bool = False,
    ) -> datetime:
        """Extract timestamp from the image using OCR.

        Args:
            img (np.ndarray): The input image from which to extract the timestamp.
            gray (bool): Whether to convert the image to grayscale before OCR.
        returns:
            datetime: The extracted timestamp.

        """
        # Ensure the zone is a numpy array
        if self.zone is None:
            crop_img = img
        else:
            x1, y1, x2, y2 = self.zone[0][0], self.zone[0][1], self.zone[2][0], self.zone[2][1]
            crop_img = img[y1:y2, x1:x2]

        crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY) if gray else crop_img

        result = self.reader.readtext(crop_img, detail=0, allowlist="0123456789-:/")
        dt = dateparser.parse(" ".join(result))
        return dt

    def extract_timestamp_video(
        self,
        video_path: str,
        frame: int = 0,
        gray: bool = False,
    ) -> datetime:
        """Extract timestamp from a video file using OCR.

        Args:
            video_path (str): The path to the video file.
            frame (int): The frame index to extract the timestamp from.
            gray (bool): Whether to convert the image to grayscale before OCR.
        returns:
            datetime: The extracted timestamp.

        """
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ret, img = cap.read()
        cap.release()

        if not ret:
            raise ValueError("Could not read video file")

        return self.extract_timestamp(img, gray=gray)

    def extract_timestamp_video_auto(
        self,
        video_path: str,
        gray: bool = False,
    ) -> tuple:
        """Extract the initial timestamp from the first readable frame in a video file using OCR.

        Args:
            video_path (str): The path to the video file.
            gray (bool): Whether to convert the image to grayscale before OCR.
        returns:
            dt, frame: The extracted timestamp and the frame index.

        """
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        tot = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        for frame in range(0, tot - 1):
            ret, img = cap.read()
            if not ret:
                break
            dt = self.extract_timestamp(img, gray=gray)
            if dt:
                cap.release()
                return dt, frame

        cap.release()
        return None, None


if __name__ == "__main__":
    img = cv2.imread("/mnt/d/videos/sample/frames/007_day.png")
    zone = np.array([[62, 1003], [543, 1000], [547, 1059], [67, 1061]])
    extractor = TimestampExtractor(zone=zone)
    timestamp = extractor.extract_timestamp(img)
    print("Extracted Timestamp:")
    print(timestamp)
