"""Segmentation wrapper around Ultralytics YOLO for video frames.

Revised by wonstran
    02/14/2026.
"""

import json
import os
from enum import Enum
from pathlib import Path

import cv2
import torch
from tqdm import tqdm
from ultralytics import YOLO


class SegmentorModel(str, Enum):  # noqa: UP042
    """Enum of available segmentation model weights."""

    YOLOv8xSeg = "yolov8x-seg.pt"
    YOLOv8lSeg = "yolov8l-seg.pt"
    YOLOv8mSeg = "yolov8m-seg.pt"
    YOLOv8sSeg = "yolov8s-seg.pt"
    YOLOv8nSeg = "yolov8n-seg.pt"
    YOLO11xSeg = "yolo11x-seg.pt"
    YOLO11lSeg = "yolo11l-seg.pt"
    YOLO11mSeg = "yolo11m-seg.pt"
    YOLO11sSeg = "yolo11s-seg.pt"
    YOLO11nSeg = "yolo11n-seg.pt"
    YOLO26xSeg = "yolo26x-seg.pt"
    YOLO26lSeg = "yolo26l-seg.pt"
    YOLO26mSeg = "yolo26m-seg.pt"
    YOLO26sSeg = "yolo26s-seg.pt"
    YOLO26nSeg = "yolo26n-seg.pt"


class Segmentor:
    """A wrapper around Ultralytics segmentation models.

    Parameters
    ----------
    model : SegmentorModel, optional
        Built-in segmentation model to use. Default is `SegmentorModel.YOLO26xSeg`.
    weights : str, optional
        Optional custom model weights to load. If relative, path is resolved
        under `<module_dir>/models/`.
    conf : float, optional
        Confidence threshold. Default is `0.25`.
    nms : float, optional
        IoU / non-maximum suppression threshold. Default is `0.7`.
    max_det : int, optional
        Maximum number of detections per frame. Default is `300`.
    device : {"auto", "cuda", "xpu", "cpu", "mps"}, optional
        Inference device to use. If `"auto"`, selection priority is
        `cuda` -> `xpu` -> `mps` -> `cpu`.
    enable_half : bool, optional
        Whether to use half precision inference. Effective on CUDA only.

    """

    def __init__(
        self,
        model: SegmentorModel | str = SegmentorModel.YOLO26xSeg,
        weights: str | None = None,
        conf: float = 0.25,
        nms: float = 0.7,
        max_det: int = 300,
        device: str = "auto",
        enable_half: bool = False,
        half: bool | None = None,
    ):
        """Initialize the Segmentor with model weights and inference parameters."""
        cwd = Path(__file__).parent.absolute()
        model_dir = cwd / "models"
        if not model_dir.exists():
            os.makedirs(model_dir)

        if weights:
            model_path = Path(weights) if os.path.isabs(weights) else model_dir / weights
        else:
            model_name = model.value if isinstance(model, SegmentorModel) else str(model)
            model_path = model_dir / model_name

        self.model = YOLO(str(model_path))
        self.conf = conf
        self.nms = nms
        self.max_det = max_det

        if half is not None:
            enable_half = half

        self.device = self._resolve_device(device)
        self.half = enable_half and self.device.startswith("cuda")

    @staticmethod
    def _resolve_device(device: str) -> str:
        """Resolve a device string into an available backend.

        Parameters
        ----------
        device : str
            Requested device, e.g. "auto", "cuda", "cpu", or "cuda:0".

        Returns
        -------
        str
            Resolved device string that is available on the host.

        Raises
        ------
        ValueError
            If the requested backend is not supported.

        """
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
            return next(d for d in auto_priority if backend_available[d])
        return requested_device if backend_available[requested_backend] else "cpu"

    @staticmethod
    def _normalize_frame_range(
        cap: cv2.VideoCapture, start_frame: int | None, end_frame: int | None
    ) -> tuple[int, int]:
        """Clamp and normalize a frame range to valid video bounds.

        Parameters
        ----------
        cap : cv2.VideoCapture
            Open video capture object.
        start_frame : int, optional
            Requested start frame index.
        end_frame : int, optional
            Requested end frame index.

        Returns
        -------
        tuple[int, int]
            Normalized (start_frame, end_frame) bounds.

        """
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            return 0, -1

        if start_frame is None or start_frame < 0 or start_frame >= total_frames:
            start_frame = 0
        if end_frame is None or end_frame < 0 or end_frame >= total_frames:
            end_frame = total_frames - 1
        if end_frame < start_frame:
            end_frame = start_frame

        return start_frame, end_frame

    def segment(
        self,
        input_video: str,
        mask_file: str | None = None,
        video_index: int | None = None,
        video_tot: int | None = None,
        start_frame: int | None = None,
        end_frame: int | None = None,
        verbose: bool = False,
    ) -> list[dict]:
        """Run segmentation on a video and return per-frame masks."""
        cap = cv2.VideoCapture(input_video)
        if not cap.isOpened():
            if verbose:
                print(f"Cannot open video: {input_video}")
            return []

        start_frame, end_frame = self._normalize_frame_range(cap, start_frame, end_frame)
        if end_frame < start_frame:
            cap.release()
            return []

        results: list[dict] = []
        frame_total = end_frame - start_frame + 1

        pbar = tqdm(total=frame_total, unit=" frames", disable=not verbose)
        if video_index is not None and video_tot is not None and verbose:
            pbar.set_description_str(f"Segmenting {video_index} of {video_tot}")
        elif verbose:
            pbar.set_description_str("Segmenting")

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        while cap.isOpened():
            pos_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            ret, frame = cap.read()
            if (not ret) or (pos_frame > end_frame):
                break

            detects = self.model.predict(
                frame,
                verbose=False,
                conf=self.conf,
                iou=self.nms,
                max_det=self.max_det,
                device=self.device,
                half=self.half,
            )

            if len(detects) > 0 and detects[0].masks is not None and detects[0].boxes is not None:
                d = {
                    "frame": pos_frame,
                    "res": -1,
                    "class": detects[0].boxes.cls.tolist(),
                    "conf": detects[0].boxes.conf.tolist(),
                    "mask": [x.tolist() for x in detects[0].masks.xy],
                }
                results.append(d)

            pbar.update(1)

        pbar.close()
        cap.release()
        cv2.destroyAllWindows()

        if mask_file:
            with open(mask_file, "w", encoding="utf-8") as out_file:
                for r in results:
                    out_file.write(json.dumps(r))
                    out_file.write("\n")
            if verbose:
                print(f"Wrote to {mask_file}")

        return results

    def segment_single(self, input_video: str, frame: int | None = None) -> list[dict]:
        """Run segmentation on a single frame of a video and return masks.

        Parameters
        ----------
        input_video : str
            Path to the input video file.
        frame : int, optional
            Frame index to segment. If None, segments the first frame.

        Returns
        -------
        list[dict]
            List of segmentation results for the specified frame.

        """
        return self.segment(
            input_video=input_video,
            mask_file=None,
            video_index=None,
            video_tot=None,
            start_frame=frame,
            end_frame=frame,
            verbose=False,
        )

    def segment_crop(self, input_video: str, frame: int, zone: list[int]) -> list[list[list[float]]]:
        """Segment objects in a specific bounding zone for one frame.

        Parameters
        ----------
        input_video : str
            Path to the input video file.
        frame : int
            Frame index to segment.
        zone : list[int]
            Crop region as [x1, y1, x2, y2] in pixel coordinates.

        Returns
        -------
        list[list[list[float]]]
            List of mask polygons in the cropped coordinate space.

        Raises
        ------
        ValueError
            If the video cannot be opened or the frame is out of range.

        """
        cap = cv2.VideoCapture(input_video)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {input_video}")

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame < 0 or frame >= frame_count:
            cap.release()
            raise ValueError("The given frame is out of range.")

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ret, frame_img = cap.read()
        if not ret:
            cap.release()
            return []

        x1, y1, x2, y2 = zone
        cropped_frame = frame_img[y1:y2, x1:x2]
        detects = self.model.predict(
            cropped_frame,
            verbose=False,
            conf=self.conf,
            iou=self.nms,
            max_det=self.max_det,
            device=self.device,
            half=self.half,
        )
        cap.release()

        if len(detects) == 0 or detects[0].masks is None:
            return []
        return [x.tolist() for x in detects[0].masks.xy]
