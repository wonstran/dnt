"""Detection wrapper around Ultralytics YOLO/RT-DETR for video frames.

Revised by wonstran
    01/28/2026
    11/11/2025.
"""

import os
from enum import Enum
from pathlib import Path
from time import time

import cv2
import pandas as pd
import torch
from tqdm import tqdm
from ultralytics import RTDETR, YOLO


class DetectorModel(str, Enum):  # noqa: UP042
    """Enum of available YOLO and RT-DETR model weights.

    Each member represents a different model variant with its corresponding
    weight file name.
    """

    YOLOv8x = "yolov8x.pt"
    YOLOv8l = "yolov8l.pt"
    YOLOv8m = "yolov8m.pt"
    YOLOv8s = "yolov8s.pt"
    YOLOv8n = "yolov8n.pt"
    YOLO11x = "yolo11x.pt"
    YOLO11l = "yolo11l.pt"
    YOLO11m = "yolo11m.pt"
    YOLO11s = "yolo11s.pt"
    YOLO11n = "yolo11n.pt"
    YOLO26x = "yolo26x.pt"
    YOLO26l = "yolo26l.pt"
    YOLO26m = "yolo26m.pt"
    YOLO26s = "yolo26s.pt"
    YOLO26n = "yolo26n.pt"
    RTDETRx = "rtdetr-x.pt"
    RTDETRl = "rtdetr-l.pt"


YOLO_MODELS = {
    DetectorModel.YOLOv8x,
    DetectorModel.YOLOv8l,
    DetectorModel.YOLOv8m,
    DetectorModel.YOLOv8s,
    DetectorModel.YOLOv8n,
    DetectorModel.YOLO11x,
    DetectorModel.YOLO11l,
    DetectorModel.YOLO11m,
    DetectorModel.YOLO11s,
    DetectorModel.YOLO11n,
    DetectorModel.YOLO26x,
    DetectorModel.YOLO26l,
    DetectorModel.YOLO26m,
    DetectorModel.YOLO26s,
    DetectorModel.YOLO26n,
}

RTDETR_MODELS = {
    DetectorModel.RTDETRx,
    DetectorModel.RTDETRl,
}


class Detector:
    """A wrapper around Ultralytics detection models for running object detection on
    videos and selected frames.

    This class loads a YOLO (v8, v11, 26) or RT-DETR model from a local `models/`
    directory (or from a user-supplied .pt file) and provides convenience
    methods to:

    - detect objects frame by frame in a video and return results as a
      pandas DataFrame,
    - run detection only on specified frame indices,
    - process a batch of videos and save per-video detection text files, and
    - query basic video properties (FPS, frame count).

    The detector automatically chooses an inference device (`cuda`, `xpu`,
    `mps`, or `cpu`) when `device="auto"`, and it can optionally enable half-precision
    inference on GPU.

    Parameters
    ----------
    model : DetectorModel, optional
        Built-in model weights to use (for example `DetectorModel.YOLO26x`).
        Default is `DetectorModel.YOLO26x`.
    weights : str, optional
        Optional custom model weights to load. If relative, path is resolved
        under `<module_dir>/models/`.
        Default is None.
    conf : float, optional
        Confidence threshold for detections. Default is `0.25`.
    nms : float, optional
        IoU / non-maximum suppression threshold. Default is `0.7`.
    max_det : int, optional
        Maximum number of detections per frame. Default is `300`.
    device : {"auto", "cuda", "xpu", "cpu", "mps"}, optional
        Inference device to use. If `"auto"`, the detector will pick an
        available accelerator first (`cuda` → `xpu` → `mps`) and fall back to CPU. Default is
        `"auto"`.
    half : bool, optional
        Whether to enable half-precision inference. This is only effective on
        GPU (CUDA). Default is `False`.

    Notes
    -----
    - The class expects model weight files to be located under
      `<module_dir>/models/` when using the built-in weight names.
    - Returned detection tables typically contain the columns:
      `frame, res, x, y, w, h, conf, class`.

    """  # noqa: D205

    from typing import ClassVar

    DET_FIELDS: ClassVar[list[str]] = ["frame", "res", "x", "y", "w", "h", "conf", "class"]

    def __init__(
        self,
        model: DetectorModel = DetectorModel.YOLO26x,
        weights: str | None = None,
        conf: float = 0.25,
        nms: float = 0.7,
        max_det: int = 300,
        device: str = "auto",
        half: bool = False,
    ):
        """Initialize a Detector for Ultralytics YOLO/RT-DETR models.

        Parameters
        ----------
        model : DetectorModel, optional
            Built-in model to use. Default is "yolo26x".
        weights : str, optional
            Customized model weights to load.
            Default is None, which means using the built-in weights in `model` choice.
        conf : float, optional
            Confidence threshold. Default is 0.25.
        nms : float, optional
            IoU/NMS threshold. Default is 0.7.
        max_det : int, optional
            Maximum detections per frame.
            Default is 300. In crowded scenes, you may want to increase this.
        device : {"auto", "cuda", "xpu", "cpu", "mps"}, optional
            Inference device. Default is "auto".
        half : bool, optional
            Whether to use half precision (GPU only). Default is False.

        """
        # Load model
        cwd = Path(__file__).parent.absolute()
        model_dir = cwd / "models"
        if not model_dir.exists():
            os.makedirs(model_dir)

        if weights:
            model_path = Path(weights) if os.path.isabs(weights) else model_dir / weights
        else:
            model_path = model_dir / f"{model.value}"

        # actually load model
        if ("yolo" in str(weights).lower()) or (model in YOLO_MODELS):
            self.model = YOLO(str(model_path))
        elif ("rtdetr" in str(weights).lower()) or (model in RTDETR_MODELS):
            self.model = RTDETR(str(model_path))
        else:
            raise ValueError(
                f"Cannot infer model family from model={model} and weights={weights!r}. "
                "Use a known DetectorModel or provide weights containing 'yolo' or 'rtdetr'."
            )
        self.conf = conf
        self.nms = nms
        self.max_det = max_det

        # device selection
        requested_device = str(device).lower().strip()
        requested_backend = requested_device.split(":", maxsplit=1)[0]
        valid_devices = {"auto", "cuda", "xpu", "mps", "cpu"}
        if requested_backend not in valid_devices:
            raise ValueError(
                f"Invalid device={device!r}. Choose one of {sorted(valid_devices)} or backend:index like 'cuda:0'."
            )

        backend_available = {
            "cuda": torch.cuda.is_available(),
            "xpu": hasattr(torch, "xpu") and hasattr(torch.xpu, "is_available") and torch.xpu.is_available(),
            "mps": hasattr(torch.backends, "mps") and torch.backends.mps.is_available(),
            "cpu": True,
        }

        if requested_backend == "auto":
            auto_priority = ("cuda", "xpu", "mps", "cpu")
            self.device = next(d for d in auto_priority if backend_available[d])
        else:
            self.device = requested_device if backend_available[requested_backend] else "cpu"

        # half precision only makes sense on GPU
        self.half = half and (self.device == "cuda")

    def detect(
        self,
        input_video: str,
        iou_file: str | None = None,
        video_index: int | None = None,
        video_tot: int | None = None,
        start_frame: int | None = None,
        end_frame: int | None = None,
        verbose: bool = True,
        show: bool = False,
        disp_filename: bool = False,
    ) -> pd.DataFrame:
        """Run object detection on a video and return per-frame detections.

        Parameters
        ----------
        input_video : str
            Path to the input video file.
        iou_file : str, optional
            If provided, detection results are written to this file (CSV without header).
        video_index : int, optional
            Index of this video in a batch, used only for progress display.
        video_tot : int, optional
            Total number of videos in the batch, used only for progress display.
        start_frame : int, optional
            Frame index to start detection from. If None or out of range, starts at 0.
        end_frame : int, optional
            Frame index to stop detection at. If None or out of range, uses the last frame.
        verbose : bool, optional
            Whether to show a progress bar. Default is True.
        show : bool, optional
            Whether to display the video frames with detections. Default is False.
        disp_filename: bool, optional
            Whether to show the file name in the progress bar. Default is False.

        Returns
        -------
        pandas.DataFrame
            DataFrame with columns:
            `frame, res, x, y, w, h, conf, class`.
            If the video cannot be opened or no detections are found, an empty DataFrame
            with those columns is returned.

        """
        # validate path
        cap = cv2.VideoCapture(input_video)
        if not cap.isOpened():
            if verbose:
                print(f"Cannot open video: {input_video}")
            return pd.DataFrame(columns=self.DET_FIELDS)

        tot_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # normalize start_frame
        if start_frame is None or start_frame < 0 or start_frame >= tot_frames:
            start_frame = 0
        # normalize end_frame
        if end_frame is None or end_frame < 0 or end_frame >= tot_frames:
            end_frame = tot_frames - 1
        if start_frame > end_frame:
            cap.release()
            raise ValueError("start_frame must be less than or equal to end_frame.")

        frame_total = end_frame - start_frame + 1

        # Some codecs return 0 or -1 for frame count
        if verbose:
            if tot_frames <= 0:
                pbar = tqdm(desc="Detecting", unit="frame")
            else:
                pbar = tqdm(total=frame_total, desc="Detecting", unit="frame")

            if (video_index is not None) and (video_tot is not None):
                desc = f"Detecting {video_index} of {video_tot}"
                if disp_filename:
                    desc += f" - {Path(input_video).name}"
                    pbar.set_description_str(desc)
            else:
                desc = "Detecting"
                if disp_filename:
                    desc += f" {Path(input_video).name}"
                    pbar.set_description_str(desc)

        results: list[dict] = []
        frame_idx = start_frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

        win_name = "Detection (press q/ESC to quit)"
        if show:
            cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

        # optional FPS calc
        t0 = time()
        n_show = 0

        while cap.isOpened():
            pos_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            ret, frame = cap.read()
            if not ret:
                break

            if end_frame is not None and frame_idx > end_frame:
                break

            preds = self.model.predict(
                source=frame,
                conf=self.conf,
                iou=self.nms,
                max_det=self.max_det,
                device=self.device,
                half=self.half,
                verbose=False,
            )

            det = preds[0]
            boxes = det.boxes
            if boxes is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.cpu().numpy()  # (N,4)
                confs = boxes.conf.cpu().numpy()  # (N,)
                clss = boxes.cls.cpu().numpy().astype(int)  # (N,)

                for (x1, y1, x2, y2), cf, c in zip(xyxy, confs, clss, strict=True):
                    results.append({
                        "frame": pos_frame,
                        "res": -1,
                        "x": float(x1),
                        "y": float(y1),
                        "x2": float(x2),
                        "y2": float(y2),
                        "conf": float(cf),
                        "class": int(c),
                    })

            if show:
                # Ultralytics built-in drawing (fast & clean)
                vis = det.plot()  # returns BGR image with boxes/labels

                # add simple overlay: frame index + FPS
                n_show += 1
                dt = time() - t0
                fps = n_show / dt if dt > 0 else 0.0
                cv2.putText(
                    vis,
                    f"frame={pos_frame}/{frame_total}  fps={fps:.1f}",
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                )

                cv2.imshow(win_name, vis)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:  # q or ESC
                    break

            if verbose and pbar is not None:
                pbar.update(1)

            frame_idx += 1

        cap.release()
        if verbose and pbar is not None:
            pbar.close()

        if not results:
            empty_df = pd.DataFrame(columns=self.DET_FIELDS)
            if iou_file:
                empty_df.to_csv(iou_file, index=False, header=False)
            return empty_df

        else:
            results_df = pd.DataFrame(results, columns=["frame", "res", "x", "y", "x2", "y2", "conf", "class"])
            results_df["w"] = (results_df["x2"] - results_df["x"]).astype(int)
            results_df["h"] = (results_df["y2"] - results_df["y"]).astype(int)
            results_df["x"] = results_df["x"].astype(int)
            results_df["y"] = results_df["y"].astype(int)
            results_df["conf"] = results_df["conf"].round(2)
            results_df = results_df[self.DET_FIELDS].reset_index(drop=True)

        if iou_file:
            folder = Path(iou_file).parent
            if not folder.exists():
                Path(folder).mkdir(parents=True, exist_ok=True)

            results_df.to_csv(iou_file, index=False, header=False)
            if verbose:
                print(f"Wrote detections to {iou_file}")

        return results_df

    def detect_frames(
        self,
        input_video: str,
        frames: list[int],
        verbose: bool = True,
    ) -> pd.DataFrame:
        """Run object detection on specific frames of a video.

        This method is useful when you don't need to process the entire video and
        only want detections for selected frame indices.

        Parameters
        ----------
        input_video : str
            Path to the input video file.
        frames : list of int
            List of frame indices to process.
        verbose : bool, optional
            Whether to show a progress bar. Default is True.

        Returns
        -------
        pandas.DataFrame
            DataFrame with columns
            `frame, res, x, y, w, h, conf, class`.
            If the video cannot be opened or no detections are found, an empty
            DataFrame with those columns is returned.

        """
        # validate path
        if not os.path.exists(input_video):
            # return an empty, well-shaped DataFrame instead of None
            if verbose:
                print(f"{input_video} does not exist!")
            return pd.DataFrame(columns=self.DET_FIELDS)

        cap = cv2.VideoCapture(input_video)
        if not cap.isOpened():
            if verbose:
                print(f"Cannot open {input_video}")
            return pd.DataFrame(columns=self.DET_FIELDS)

        results: list[dict] = []

        pbar = tqdm(total=len(frames), unit=" frames") if verbose else None

        for pos_frame in frames:
            # move to target frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos_frame)
            ret, frame = cap.read()
            if not ret:
                # e.g. frame index out of range
                continue

            preds = self.model.predict(
                frame,
                verbose=False,
                conf=self.conf,
                iou=self.nms,
                max_det=self.max_det,
                device=self.device,
                half=self.half,
            )
            det = preds[0]
            boxes = det.boxes
            if boxes is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.cpu().numpy()  # (N,4)
                confs = boxes.conf.cpu().numpy()  # (N,)
                clss = boxes.cls.cpu().numpy().astype(int)  # (N,)

                for (x1, y1, x2, y2), cf, c in zip(xyxy, confs, clss, strict=True):
                    results.append({
                        "frame": pos_frame,
                        "res": -1,
                        "x": float(x1),
                        "y": float(y1),
                        "x2": float(x2),
                        "y2": float(y2),
                        "conf": float(cf),
                        "class": int(c),
                    })

            if pbar is not None:
                pbar.update()

        if pbar is not None:
            pbar.close()
        cap.release()

        # no detections at all

        if not results:
            return pd.DataFrame(columns=self.DET_FIELDS)

        df = pd.DataFrame(results)
        # compute width/height and round
        df["w"] = (df["x2"] - df["x"]).round(0)
        df["h"] = (df["y2"] - df["y"]).round(0)
        df["x"] = df["x"].round(1)
        df["y"] = df["y"].round(1)
        df["conf"] = df["conf"].round(2)
        df["class"] = df["class"].round(0).astype(int)

        df = df[self.DET_FIELDS].reset_index(drop=True)

        return df

    def detect_batch(
        self,
        input_videos: list[str],
        output_path: str | None = None,
        is_overwrite: bool = False,
        is_report: bool = True,
        verbose: bool = True,
        disp_filename: bool = True,
    ) -> list[str]:
        """Run detection on multiple videos and optionally write per-video output files.

        Parameters
        ----------
        input_videos : list of str
            Paths to the input video files to be processed.
        output_path : str, optional
            Directory where per-video detection files will be written. If None,
            detections are not written to disk and the returned list will be empty.
        is_overwrite : bool, optional
            If False (default), existing detection files with the same name will be
            skipped. If True, they will be regenerated.
        is_report : bool, optional
            If True (default), existing detection files (that were skipped) are still
            included in the returned list.
        verbose : bool, optional
            If True, prints progress messages. Default is True.
        disp_filename: bool, optional
            If True, prints file name in progress bar. Default is True.

        Returns
        -------
        list of str
            A list of paths to detection files that were created or already existed.
            If `output_path` is None, this will be an empty list.

        """
        results: list[str] = []
        total_videos = len(input_videos)

        for idx, input_video in enumerate(input_videos, start=1):
            # default: no output file
            iou_file = None

            # build output path / file name if requested
            if output_path is not None:
                Path(output_path).mkdir(parents=True, exist_ok=True)
                base_filename = os.path.splitext(os.path.basename(input_video))[0]
                iou_file = os.path.join(output_path, f"{base_filename}_iou.txt")

            # if we have an output file name, check overwrite logic
            if (iou_file is not None) and (not is_overwrite) and os.path.exists(iou_file):
                if is_report:
                    results.append(iou_file)
                # skip processing this video
                continue

            # run detection (may write to iou_file if not None)
            self.detect(
                input_video=input_video,
                iou_file=iou_file,
                video_index=idx,
                video_tot=total_videos,
                verbose=verbose,
                disp_filename=disp_filename,
            )

            if iou_file is not None:
                results.append(iou_file)

        return results

    @staticmethod
    def get_fps(video: str) -> float:
        """Return the frames-per-second (FPS) value of a video file.

        Parameters
        ----------
        video : str
            Path to the video file.

        Returns
        -------
        float
            FPS of the video. Returns 0.0 if the video cannot be opened.

        """
        if not Path(video).exists():
            print(f"{video} does not exist!")
            return 0.0
        cap = cv2.VideoCapture(video)
        if not cap.isOpened():
            print(f"Failed to open the video: {video}")
            return 0.0

        fps = float(cap.get(cv2.CAP_PROP_FPS))
        cap.release()
        return fps

    @staticmethod
    def get_frames(video: str) -> int:
        """Return the total number of frames in a video file.

        Parameters
        ----------
        video : str
            Path to the video file.

        Returns
        -------
        int
            Total frame count. Returns 0 if the video cannot be opened.

        """
        if not Path(video).exists():
            print(f"{video} does not exist!")
            return 0
        cap = cv2.VideoCapture(video)
        if not cap.isOpened():
            print(f"Failed to open the video: {video}")
            return 0

        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return frames


if __name__ == "__main__":
    detector = Detector(half=True)
    result = detector.detect("/mnt/d/videos/sample/traffic.mp4", verbose=True)
    print(result)
