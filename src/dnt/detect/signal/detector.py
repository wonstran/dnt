import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from shared.download import download_file
from torchvision.models.resnet import ResNet18_Weights
from tqdm import tqdm


class Model(nn.Module):
    """ResNet18-based model for traffic signal detection.

    A neural network model that uses ResNet18 as the backbone for classifying
    traffic signal states.

    Attributes
    ----------
    resnet18 : torchvision.models.ResNet
        The ResNet18 backbone model with a custom fully connected layer.

    Methods
    -------
    forward(x)
        Pass input through the model and return predictions.

    """

    def __init__(self, num_class=2):
        """Initialize the Model with ResNet18 backbone.

        Parameters
        ----------
        num_class : int, optional
            The number of output classes, by default 2

        """
        super().__init__()

        self.resnet18 = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.resnet18.fc = nn.Linear(512, num_class)

    def forward(self, x):
        return self.resnet18(x)


class SignalDetector:
    """Detector for traffic signal status in videos.

    A class that uses a pre-trained ResNet18 model to detect pedestrian signal
    states (walking or not walking) in video frames.

    Attributes
    ----------
    det_zones : list
        List of detection zones as tuples (x, y, w, h).
    model : Model
        The neural network model for signal detection.
    device : str
        The device to run the model on ('cuda', 'mps', or 'cpu').
    batchsz : int
        The batch size for predictions.
    threshold : float
        The confidence threshold for positive detections.

    Methods
    -------
    detect(input_video, det_file, video_index, video_tot)
        Detect signal states in a video file.
    gen_ped_interval(dets, input_video, walk_interval, countdown_interval, out_file, factor, video_index, video_tot)
        Generate pedestrian signal intervals from detections.
    crop_zone(frame)
        Crop detection zones from a video frame.
    predict(batch)
        Make predictions on a batch of cropped images.
    generate_labels(signals, input_video, label_file, size_factor, thick, video_index, video_tot)
        Generate visualization labels for signal states.

    """

    def __init__(
        self,
        det_zones: list,
        model: str = "ped",
        weights: str = None,
        batchsz: int = 64,
        num_class: int = 2,
        threshold: float = 0.98,
        device="auto",
    ):
        """Detect traffic signal status.

        Parameters
        ----------
        det_zones : list
            Cropped zones for detection list[(x, y, w, h)]
        model : str, optional
            Detection model, default is 'ped', 'custom'
        weights : str, optional
            Path of weights, default is None
        batchsz : int, optional
            The batch size for prediction, default is 64
        num_class : int, optional
            The number of classes, default is 2
        threshold : float, optional
            The threshold for detection, default is 0.98
        device : str, optional
            The device to run the model on ('cuda', 'mps', 'cpu', or 'auto'), default is 'auto'

        """
        self.det_zones = det_zones

        cwd = Path(__file__).parent.absolute()
        if not weights and model == "ped":
            weights = os.path.join(cwd, "weights", "ped_signal.pt")

        if not os.path.exists(weights):
            url = "https://its.cutr.usf.edu/alms/downloads/ped_signal.pt"
            download_file(url, weights)

        self.model = Model(num_class)
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        elif device == "cuda" and torch.cuda.is_available():
            self.device = "cuda"
        elif device == "mps" and torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        self.model.load_state_dict(torch.load(weights))
        self.model.to(self.device)

        self.batchsz = batchsz
        self.threshold = threshold

    def detect(
        self,
        input_video: str,
        det_file: str | None = None,
        video_index: int | None = None,
        video_tot: int | None = None,
    ) -> pd.DataFrame:
        """Detect signal states in a video file.

        Parameters
        ----------
        input_video : str
            Path to the input video file.
        det_file : str, optional
            Path to save detection results CSV. If None, results are not saved.
        video_index : int, optional
            Video index for batch processing display.
        video_tot : int, optional
            Total video count for batch processing display.

        Returns
        -------
        pd.DataFrame
            Detection results with columns: ['frame', 'signal', 'detection'].

        Raises
        ------
        OSError
            If the input video cannot be opened.

        """
        # initialize the result array
        results = np.array([])  # np.full((0, len(self.det_zones)), -1)
        frames = np.array([])
        zones = np.array([])
        batch = []
        temp_frames = []

        # open input video
        cap = cv2.VideoCapture(input_video)
        if not cap.isOpened():
            raise OSError(f"Failed to open video: {input_video}")

        tot_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        pbar = tqdm(total=tot_frames, unit=" frame")
        if video_index and video_tot:
            pbar.set_description_str(f"Detecting signals {video_index} of {video_tot}")
        else:
            pbar.set_description_str("Detecting signals ")

        while cap.isOpened():
            pos_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            ret, frame = cap.read()
            if not ret:
                break

            crop_img = self.crop_zone(frame)
            batch.append(crop_img)
            temp_frames.append(pos_frame)

            if ((pos_frame + 1) % self.batchsz == 0) or (pos_frame >= tot_frames - 1):
                # batch_pred = self.predict(batch).reshape(-1, len(self.det_zones))
                # results = np.append(results, batch_pred, axis=0)

                batch_pred = self.predict(batch).flatten()
                results = np.append(results, batch_pred, axis=0)
                zones = np.append(zones, np.tile(np.array(list(range(len(self.det_zones)))), self.batchsz), axis=0)
                frames = np.append(frames, np.repeat(np.array(temp_frames), len(self.det_zones)), axis=0)

                batch = []
                temp_frames = []

            pbar.update()

        pbar.close()
        cap.release()

        df = pd.DataFrame(list(zip(frames, zones, results, strict=True)), columns=["frame", "signal", "detection"])

        if det_file:
            df.to_csv(det_file, index=False)

        return df

    def gen_ped_interval(
        self,
        dets: pd.DataFrame,
        input_video: str,
        walk_interval: int,
        countdown_interval: int,
        out_file: str | None,
        factor: float = 0.75,
        video_index: int | None = None,
        video_tot: int | None = None,
    ) -> pd.DataFrame:
        """Generate pedestrian signal intervals from detections.

        Parameters
        ----------
        dets : pd.DataFrame
            Signal detection results from `detect()` method.
        input_video : str
            Path to the input video file.
        walk_interval : int
            Walking signal interval in seconds (typically 4-7 seconds).
        countdown_interval : int
            Countdown interval in seconds. Typically calculated as:
            crossing length (ft) / 4 (ft/s).
        out_file : str, optional
            Path to save interval results CSV. If None, results are not saved.
        factor : float, optional
            Detection threshold (0-1). If *factor* portion of frames in a
            sliding window show walking signal, it's classified as walking.
            Default is 0.75 (75%).
        video_index : int, optional
            Video index for batch processing display.
        video_tot : int, optional
            Total video count for batch processing display.

        Returns
        -------
        pd.DataFrame
            Interval data with columns: ['signal', 'status', 'beg_frame', 'end_frame'].

        Raises
        ------
        OSError
            If the input video cannot be opened.

        """
        # open input video
        cap = cv2.VideoCapture(input_video)
        if not cap.isOpened():
            raise OSError(f"Failed to open video: {input_video}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        results = []
        for i in range(len(self.det_zones)):
            dets_i = dets[dets["signal"] == i]
            results.append(
                self.scan_walk_interval(dets_i, int(fps * walk_interval), factor, i, fps, countdown_interval)
            )

        df = pd.concat(results, axis=0)
        if out_file:
            df.to_csv(out_file, index=False)

        return df

    def scan_walk_interval(
        self,
        dets: pd.DataFrame,
        window: int,
        factor: float,
        zone: int,
        fps: int = 30,
        countdown_interval: int = 10,
        video_index: int | None = None,
        video_tot: int | None = None,
    ) -> pd.DataFrame:
        """Scan detection sequence to identify walking signal intervals.

        Parameters
        ----------
        dets : pd.DataFrame
            Detection data for a specific signal zone.
        window : int
            Sliding window size in frames.
        factor : float
            Threshold ratio (0-1) for positive detection within window.
        zone : int
            Signal zone identifier.
        fps : int, optional
            Video frame rate. Default is 30.
        countdown_interval : int, optional
            Duration in seconds for countdown phase after walk signal.
            Default is 10.
        video_index : int, optional
            Video index for batch processing display (currently unused).
        video_tot : int, optional
            Total video count for batch processing display (currently unused).

        Returns
        -------
        pd.DataFrame
            Interval data with columns: ['signal', 'status', 'beg_frame', 'end_frame'].
            Status: 1 = walking, 2 = countdown.

        """
        sequence = dets["detection"].to_numpy()

        if len(sequence) == 0:
            return pd.DataFrame(columns=["signal", "status", "beg_frame", "end_frame"])

        frame_intervals = []
        pre_walk = False
        tmp_cnt = 0

        pbar = tqdm(total=len(sequence) - window, unit=" frame")
        if video_index and video_tot:
            pbar.set_description_str(f"Scanning intervals for signal {zone}, {video_index} of {video_tot}")
        else:
            pbar.set_description_str(f"Scanning intervals for signal {zone}")

        for i in range(len(sequence) - window):
            count = sum(sequence[i : i + window])

            # check if the current frame can be a start of green light
            is_walking = count >= factor * window

            # if the current is green
            # 1) if prev status is green, update the latest interval
            # 2) if prev status is not green, append a new interval
            if is_walking:
                if not pre_walk:
                    frame_intervals.append([i, i + window])
                    tmp_cnt = 0
                else:
                    if count > tmp_cnt:
                        tmp_cnt = count
                        frame_intervals[-1] = [i, i + window]

            pre_walk = is_walking

            pbar.update()
        pbar.close()

        results = []
        for start, end in frame_intervals:
            results.append([zone, 1, int(dets["frame"].iloc[start]), int(dets["frame"].iloc[end])])
            results.append([
                zone,
                2,
                int(dets["frame"].iloc[end]) + 1,
                int(dets["frame"].iloc[end] + int(countdown_interval * fps)),
            ])

        df = pd.DataFrame(results, columns=["signal", "status", "beg_frame", "end_frame"])
        return df

    def crop_zone(self, frame: np.ndarray) -> list[Image.Image]:
        """Crop detection zones from a video frame.

        Parameters
        ----------
        frame : np.ndarray
            Input video frame (BGR image).

        Returns
        -------
        list[Image.Image]
            List of cropped PIL Images, one per detection zone.

        """
        crop_regions = []
        for _, region in enumerate(self.det_zones):
            x, y, w, h = region
            cropped = frame[y : y + h, x : x + w]
            cropped = Image.fromarray(cropped)
            crop_regions.append(cropped)
        return crop_regions

    def predict(self, batch: list[list[Image.Image]]) -> np.ndarray:
        """Make predictions on a batch of cropped images.

        Parameters
        ----------
        batch : list[list[Image.Image]]
            Batch of cropped images. Outer list represents frames, inner list
            represents zones per frame.

        Returns
        -------
        np.ndarray
            Binary predictions (0 or 1) with shape (batch_size, num_zones).
            1 indicates detected walking signal, 0 otherwise.

        """
        self.model.eval()
        sf = nn.Softmax(dim=1)
        batchsz = len(batch)
        num_group = len(batch[0])

        # define the image transform
        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Transform all images and stack to different groups(corresponding light)
        transformed_crops = [[transform(image) for image in crop] for crop in batch]
        transformed_batch = [torch.stack([transformed_crops[j][i] for j in range(batchsz)]) for i in range(num_group)]

        # send to model and make predictions zone by zone
        batch_pred = []
        for group in transformed_batch:
            # compute the probability of the detections
            outputs = self.model(group.to(self.device))
            sf_output = sf(outputs)

            # predict green only if the probability > threshold
            y_pred = (sf_output[:, 1] > self.threshold).int()
            batch_pred.append(y_pred.data.cpu().numpy())

        pred_array = np.array(batch_pred).T
        # prob_array = np.array(batch_prob).T

        return pred_array

    def generate_labels(
        self,
        signals: pd.DataFrame,
        input_video: str,
        label_file: str | None,
        size_factor: float = 1.5,
        thick: int = 1,
        video_index: int | None = None,
        video_tot: int | None = None,
    ) -> pd.DataFrame:
        """Generate visualization labels for signal states.

        Parameters
        ----------
        signals : pd.DataFrame
            Signal interval data from `gen_ped_interval()` method.
        input_video : str
            Path to the input video file.
        label_file : str, optional
            Path to save label CSV. If None, results are not saved.
        size_factor : float, optional
            Multiplier for circle radius relative to zone size. Default is 1.5.
        thick : int, optional
            Circle outline thickness. Use -1 for filled circles. Default is 1.
        video_index : int, optional
            Video index for batch processing display.
        video_tot : int, optional
            Total video count for batch processing display.

        Returns
        -------
        pd.DataFrame
            Label data with columns: ['frame', 'type', 'coords', 'color', 'size', 'thick', 'desc'].
            Colors: red (0,0,255) = no walking, green (0,255,0) = walking,
            yellow (0,255,255) = countdown.

        Raises
        ------
        OSError
            If the input video cannot be opened.

        """
        # open input video
        cap = cv2.VideoCapture(input_video)
        if not cap.isOpened():
            raise OSError(f"Failed to open video: {input_video}")
        tot_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        pbar = tqdm(total=tot_frames)
        if video_index and video_tot:
            pbar.set_description_str(f"Generating signal labeles for {video_index} of {video_tot}")
        else:
            pbar.set_description_str("Generating signal labels")

        results = []
        for i in range(tot_frames):
            for j in range(len(self.det_zones)):
                status = 0  # no walking
                selected = signals[(signals["beg_frame"] <= i) & (signals["end_frame"] >= i) & (signals["signal"] == j)]
                if len(selected) > 0:
                    status = selected["status"].iloc[0]

                x, y, w, h = self.det_zones[j]
                cx = int(x + w / 2)
                cy = int(y + h / 2)
                r = int(max(w, h) * size_factor)

                if status == 0:
                    results.append([i, "circle", [(cx, cy)], (0, 0, 255), r, thick, ""])
                elif status == 1:
                    results.append([i, "circle", [(cx, cy)], (0, 255, 0), r, thick, ""])
                elif status == 2:
                    results.append([i, "circle", [(cx, cy)], (0, 255, 255), r, thick, ""])

                pbar.update()

        df = pd.DataFrame(results, columns=["frame", "type", "coords", "color", "size", "thick", "desc"])
        df.sort_values(by="frame")

        if label_file:
            df.to_csv(label_file, index=False)

        return df
