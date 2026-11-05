"""Segmentation wrapper around Ultralytics YOLO for video frames.

Revised by wonstran
    02/14/2026.
"""

import json
import os
from enum import Enum
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm
from ultralytics import YOLO

from ..._device import half_allowed, predict_precision_kwargs, resolve_device
from .detector import Detector


class SegmentorModel(str, Enum):  # ruff: ignore[replace-str-enum]
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


def _coerce_model(model: "SegmentorModel | str") -> "SegmentorModel | str":
    """Accept a SegmentorModel, its value ('yolo26x-seg.pt'), stem ('yolo26x-seg') or name ('YOLO26xSeg').

    Anything else is returned unchanged so it can still be used as a weight file name.
    """
    if isinstance(model, SegmentorModel):
        return model
    text = str(model).strip().lower()
    for candidate in SegmentorModel:
        if text in {candidate.value.lower(), candidate.value.lower().removesuffix(".pt"), candidate.name.lower()}:
            return candidate
    return str(model)


class Segmentor:
    """A wrapper around Ultralytics segmentation models.

    The interface mirrors `Detector`: the same constructor arguments, plus
    `segment()`, `segment_frames()`, `segment_image()`, `segment_batch()`,
    `get_fps()` and `get_frames()`. Results are per-frame dicts (also written as JSON
    lines to `mask_file`) with keys `frame, res, class, conf, mask`.

    Parameters
    ----------
    model : SegmentorModel or str, optional
        Built-in segmentation model to use. Accepts a `SegmentorModel` member, its value /
        file name (`"yolo26x-seg.pt"`), its stem (`"yolo26x-seg"`), or its enum name
        (`"YOLO26xSeg"`). Default is `SegmentorModel.YOLO26xSeg`.
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
    half : bool, optional
        Whether to use half precision inference. Effective on CUDA only. Default is `False`.
    imgsz : int, optional
        Inference image size, forwarded to Ultralytics `model.predict(imgsz=...)`.
        Default is `640`.
    classes : list of int, optional
        Restrict segmentation to these class indices. Default is `None` (all classes).
    class_names : list of str, optional
        Restrict segmentation to these class names (e.g. `["car", "truck"]`), resolved
        against `dnt.shared.util.load_class_dict()` and merged into `classes`.
    rect : bool, optional
        Whether to use rectangular inference. Default is `False`.
    agnostic_nms : bool, optional
        Whether to use class-agnostic NMS. Default is `False`.
    fast : bool, optional
        Whether `segment()` redirects to `segment_fast(batch=self.batch)` (`True`, the
        default) or runs the one-frame-at-a-time pipeline (`False`). Mutate
        `segmentor.fast` after construction to change it for later calls.
    batch : int, optional
        Frames per forward pass when `fast` is True. Default is `8`.
    enable_half : bool, optional
        Deprecated alias of `half`; if given, it overrides `half`.

    """

    def __init__(
        self,
        model: SegmentorModel | str = SegmentorModel.YOLO26xSeg,
        weights: str | None = None,
        conf: float = 0.25,
        nms: float = 0.7,
        max_det: int = 300,
        device: str = "auto",
        half: bool = False,
        imgsz: int = 640,
        classes: list[int] | None = None,
        class_names: list[str] | None = None,
        rect: bool = False,
        agnostic_nms: bool = False,
        enable_half: bool | None = None,
        fast: bool = True,
        batch: int = 8,
    ):
        """Initialize the Segmentor with model weights and inference parameters."""
        model = _coerce_model(model)
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
        self.imgsz = imgsz
        if class_names:
            # deferred: dnt.shared imports dnt.detect, so importing at module level would cycle
            from ...shared.util import load_class_dict

            name_to_idx = load_class_dict()
            try:
                resolved = {name_to_idx[name] for name in class_names}
            except KeyError as e:
                raise ValueError(
                    f"Unknown class name {e.args[0]!r}. See dnt.shared.util.load_class_dict() for valid names."
                ) from e
            classes = sorted({*(classes or []), *resolved})
        self.classes = classes
        self.rect = rect
        self.agnostic_nms = agnostic_nms
        self.fast = fast
        self.batch = batch

        if enable_half is not None:
            half = enable_half

        self.device = resolve_device(device)
        self.half = half_allowed(self.device, half)
        # translates to quantize= on Ultralytics versions that deprecated half= (avoids a
        # per-frame deprecation warning); falls back to half= on older versions
        self._precision_kwargs = predict_precision_kwargs(self.half)

    @staticmethod
    def _resolve_device(device: str) -> str:
        """Resolve a device string (kept for compatibility; see ``dnt._device.resolve_device``).

        Parameters
        ----------
        device : str
            Requested device, e.g. "auto", "cuda", "cpu", or "cuda:0".

        Returns
        -------
        str
            Resolved device string that is available on the host.

        """
        return resolve_device(device)

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

    def _predict(self, image: np.ndarray):
        """Predict one BGR image with this instance's settings; return the Ultralytics Results."""
        return self.model.predict(
            image,
            verbose=False,
            conf=self.conf,
            iou=self.nms,
            max_det=self.max_det,
            device=self.device,
            imgsz=self.imgsz,
            classes=self.classes,
            rect=self.rect,
            agnostic_nms=self.agnostic_nms,
            **self._precision_kwargs,
        )

    def _frame_result(self, image: np.ndarray, frame: int) -> dict | None:
        """Segment one image; return the per-frame result dict, or None if nothing was segmented."""
        detects = self._predict(image)
        return self._result_to_dict(detects[0], frame) if len(detects) > 0 else None

    @staticmethod
    def _result_to_dict(result, frame: int) -> dict | None:
        """Convert one Ultralytics Results to the per-frame dict, or None if it has no masks."""
        if result.masks is None or result.boxes is None:
            return None
        return {
            "frame": frame,
            "res": -1,
            "class": result.boxes.cls.tolist(),
            "conf": result.boxes.conf.tolist(),
            "mask": [x.tolist() for x in result.masks.xy],
        }

    def _ensure_predictor(self):
        """Return an Ultralytics predictor configured like `_predict()`'s per-frame calls.

        Mirrors `Detector._ensure_predictor()`: the first call (or a device change) runs one
        throwaway `model.predict()` on a dummy frame to build the predictor, later calls just
        re-merge the current settings into its args.
        """
        args = {
            "conf": self.conf,
            "iou": self.nms,
            "max_det": self.max_det,
            "device": self.device,
            "imgsz": self.imgsz,
            "classes": self.classes,
            "rect": self.rect,
            "agnostic_nms": self.agnostic_nms,
            "verbose": False,
            **self._precision_kwargs,
        }
        predictor = self.model.predictor
        if predictor is None or resolve_device(predictor.args.device) != resolve_device(self.device):
            dummy = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
            self.model.predict(source=dummy, **args)
            predictor = self.model.predictor
        else:
            # deferred: internal Ultralytics helper, as in Detector._ensure_predictor()
            from ultralytics.cfg import get_cfg

            predictor.args = get_cfg(predictor.args, args)
        return predictor

    @staticmethod
    def _resume_part_file(part: Path) -> int | None:
        """Trim a partial JSON-lines file for resuming; return the frame to restart from.

        Drops a trailing partial line and the last complete line (that frame is simply
        segmented again). Only the file's tail is read. Returns None, leaving an empty
        file, when no complete line exists.
        """
        size = part.stat().st_size
        if size == 0:
            return None
        block = 1 << 20
        with part.open("rb+") as f:
            while True:
                start = max(0, size - block)
                f.seek(start)
                buf = f.read(size - start)
                end = buf.rfind(b"\n")  # end of the last complete line
                if end < 0:
                    if start == 0:
                        f.truncate(0)
                        return None
                    block *= 2
                    continue
                begin = buf.rfind(b"\n", 0, end) + 1  # start of that line (0 if not in buf)
                if begin == 0 and start > 0:
                    block *= 2
                    continue
                frame = int(json.loads(buf[begin:end])["frame"])
                f.truncate(start + begin)
                return frame

    @staticmethod
    def _write_masks(mask_file: str, results: list[dict]) -> None:
        """Write results to `mask_file` as JSON lines, creating its folder if needed."""
        Path(mask_file).parent.mkdir(parents=True, exist_ok=True)
        with open(mask_file, "w", encoding="utf-8") as out_file:
            for r in results:
                out_file.write(json.dumps(r))
                out_file.write("\n")

    def segment(
        self,
        input_video: str,
        mask_file: str | None = None,
        video_index: int | None = None,
        video_tot: int | None = None,
        start_frame: int | None = None,
        end_frame: int | None = None,
        verbose: bool = True,
        message: str | None = None,
        return_df: bool = True,
    ) -> list[dict] | None:
        """Run segmentation on a video and return per-frame masks.

        Parameters
        ----------
        input_video : str
            Path to the input video file.
        mask_file : str, optional
            If provided, results are written to this file as JSON lines.
        video_index : int, optional
            Index of this video in a batch, used only for progress display.
        video_tot : int, optional
            Total number of videos in the batch, used only for progress display.
        start_frame : int, optional
            Frame index to start from. If None or out of range, starts at 0.
        end_frame : int, optional
            Frame index to stop at. If None or out of range, uses the last frame.
        verbose : bool, optional
            Whether to show a progress bar. Default is True.
        message : str, optional
            Optional message shown in the progress bar description.
        return_df : bool, optional
            Whether to return the results (the name follows `Detector.detect`). Default is
            True. With False, results are only written to `mask_file` (then required) and
            None is returned.

        Returns
        -------
        list of dict or None
            One dict per frame with segmentations: `frame, res, class, conf, mask`.
            Empty if the video cannot be opened. None if `return_df` is False.

        """
        if not return_df and not mask_file:
            msg = "return_df=False needs a mask_file; otherwise the masks would be discarded."
            raise ValueError(msg)
        if self.fast:
            return self.segment_fast(
                input_video,
                mask_file=mask_file,
                video_index=video_index,
                video_tot=video_tot,
                start_frame=start_frame,
                end_frame=end_frame,
                verbose=verbose,
                message=message,
                batch=self.batch,
                return_df=return_df,
            )

        cap = cv2.VideoCapture(input_video)
        if not cap.isOpened():
            if verbose:
                print(f"Cannot open video: {input_video}")
            return [] if return_df else None

        start_frame, end_frame = self._normalize_frame_range(cap, start_frame, end_frame)
        if end_frame < start_frame:
            cap.release()
            return [] if return_df else None

        results: list[dict] = []
        frame_total = end_frame - start_frame + 1

        desc = "Segmenting"
        if video_index is not None and video_tot is not None:
            desc += f" {video_index} of {video_tot}"
        if message is not None:
            desc += f" {message}"
        pbar = tqdm(total=frame_total, desc=desc, unit=" frames", disable=not verbose)

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        while cap.isOpened():
            pos_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            ret, frame = cap.read()
            if (not ret) or (pos_frame > end_frame):
                break

            d = self._frame_result(frame, pos_frame)
            if d is not None:
                results.append(d)

            pbar.update(1)

        pbar.close()
        cap.release()

        if mask_file:
            self._write_masks(mask_file, results)
            if verbose:
                print(f"Wrote to {mask_file}")

        return results if return_df else None

    def segment_fast(
        self,
        input_video: str,
        mask_file: str | None = None,
        video_index: int | None = None,
        video_tot: int | None = None,
        start_frame: int | None = None,
        end_frame: int | None = None,
        verbose: bool = True,
        message: str | None = None,
        batch: int | None = None,
        return_df: bool = True,
    ) -> list[dict] | None:
        """Like `segment()`, but batches frames through the Ultralytics predictor directly.

        Reuses Ultralytics' own `preprocess()`/`inference()`/`postprocess()` (so mask
        decoding is unchanged) while skipping `model.predict()`'s per-frame dataset setup.
        See `Detector.detect_fast()` for the speed/precision trade-offs, which apply here
        too (notably, `batch > 1` with `half=True` on CUDA can shift a few results).

        With `mask_file`, each batch's lines are appended to `<mask_file>.part` as soon as
        they are computed, and the file is renamed to `mask_file` only once the whole range
        is done. If `<mask_file>.part` already exists (e.g. after a crash), segmentation
        resumes from it: its last frame is dropped and segmented again. Delete the `.part`
        file to start over.

        Parameters
        ----------
        input_video : str
            Path to the input video file.
        mask_file : str, optional
            If provided, results are written to this file as JSON lines.
        video_index : int, optional
            Index of this video in a batch, used only for progress display.
        video_tot : int, optional
            Total number of videos in the batch, used only for progress display.
        start_frame : int, optional
            Frame index to start from. If None or out of range, starts at 0.
        end_frame : int, optional
            Frame index to stop at. If None or out of range, uses the last frame.
        verbose : bool, optional
            Whether to show a progress bar. Default is True.
        message : str, optional
            Optional message shown in the progress bar description.
        batch : int, optional
            Frames per forward pass. Default is None: use `self.batch`.
        return_df : bool, optional
            Whether to return the results. Default is True. With False, results are only
            written to `mask_file` (then required) and None is returned.

        Returns
        -------
        list of dict or None
            Same as `segment()`.

        """
        batch = self.batch if batch is None else batch
        if batch < 1:
            msg = "batch must be >= 1."
            raise ValueError(msg)
        if not return_df and not mask_file:
            msg = "return_df=False needs a mask_file; otherwise the masks would be discarded."
            raise ValueError(msg)

        cap = cv2.VideoCapture(input_video)
        if not cap.isOpened():
            if verbose:
                print(f"Cannot open video: {input_video}")
            return [] if return_df else None

        start_frame, end_frame = self._normalize_frame_range(cap, start_frame, end_frame)
        if end_frame < start_frame:
            cap.release()
            return [] if return_df else None

        part = None
        resumed = False
        if mask_file:
            Path(mask_file).parent.mkdir(parents=True, exist_ok=True)
            part = Path(f"{mask_file}.part")
            if part.exists():
                resumed = True
                resume_at = self._resume_part_file(part)
                if resume_at is not None:
                    start_frame = max(start_frame, resume_at)
                if verbose:
                    print(f"Resuming {mask_file} from frame {start_frame} (delete {part} to start over).")

        desc = "Segmenting"
        if video_index is not None and video_tot is not None:
            desc += f" {video_index} of {video_tot}"
        if message is not None:
            desc += f" {message}"
        pbar = tqdm(total=max(end_frame - start_frame + 1, 0), desc=desc, unit=" frames", disable=not verbose)

        results: list[dict] = []
        im0s: list[np.ndarray] = []
        frame_ids: list[int] = []
        out = part.open("a", encoding="utf-8") if part is not None else None
        try:
            predictor = self._ensure_predictor() if start_frame <= end_frame else None

            def flush() -> None:
                """Segment the buffered frames and write/keep their results."""
                if not im0s:
                    return
                im = predictor.preprocess(im0s)
                # postprocess() needs a path per image (unused here)
                predictor.batch = ([input_video] * len(im0s), im0s, [""] * len(im0s))
                with torch.inference_mode():
                    preds = predictor.inference(im)
                for r, fid in zip(predictor.postprocess(preds, im, im0s), frame_ids, strict=True):
                    d = self._result_to_dict(r, fid)
                    if d is None:
                        continue
                    if out is not None:
                        out.write(json.dumps(d) + "\n")
                    if return_df:
                        results.append(d)
                if out is not None:
                    out.flush()
                im0s.clear()
                frame_ids.clear()

            if start_frame <= end_frame:
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                while cap.isOpened():
                    pos_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                    ret, frame = cap.read()
                    if (not ret) or pos_frame > end_frame:
                        break
                    im0s.append(frame)
                    frame_ids.append(pos_frame)
                    if len(im0s) >= batch:
                        flush()
                    pbar.update(1)
                flush()  # trailing partial batch
        finally:
            cap.release()
            pbar.close()
            if out is not None:
                out.close()

        if part is not None:
            os.replace(part, mask_file)
            if verbose:
                print(f"Wrote to {mask_file}")
        if not return_df:
            return None
        if resumed:  # lines from the earlier run are only in the file
            with open(mask_file, encoding="utf-8") as f:
                return [json.loads(line) for line in f if line.strip()]
        return results

    def segment_image(self, image: np.ndarray, frame: int = 0) -> dict | None:
        """Run segmentation on one already-decoded BGR image.

        Parameters
        ----------
        image : numpy.ndarray
            BGR image, as read by OpenCV.
        frame : int, optional
            Value written to the ``frame`` key. Default is 0.

        Returns
        -------
        dict or None
            The per-frame result (`frame, res, class, conf, mask`), or None when
            nothing is segmented.

        """
        return self._frame_result(image, frame)

    def segment_frames(self, input_video: str, frames: list[int], verbose: bool = True) -> list[dict]:
        """Run segmentation on specific frames of a video.

        Parameters
        ----------
        input_video : str
            Path to the input video file.
        frames : list of int
            Frame indices to process.
        verbose : bool, optional
            Whether to show a progress bar. Default is True.

        Returns
        -------
        list of dict
            Per-frame results as in `segment()`. Empty if the video cannot be opened.

        """
        if not os.path.exists(input_video):
            if verbose:
                print(f"{input_video} does not exist!")
            return []

        cap = cv2.VideoCapture(input_video)
        if not cap.isOpened():
            if verbose:
                print(f"Cannot open {input_video}")
            return []

        results: list[dict] = []
        pbar = tqdm(total=len(frames), unit=" frames", disable=not verbose)
        for pos_frame in frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos_frame)
            ret, frame = cap.read()
            if not ret:
                continue
            d = self._frame_result(frame, pos_frame)
            if d is not None:
                results.append(d)
            pbar.update()
        pbar.close()
        cap.release()
        return results

    def segment_batch(
        self,
        input_videos: list[str],
        output_path: str | None = None,
        is_overwrite: bool = False,
        is_report: bool = True,
        verbose: bool = True,
        message: str | None = None,
    ) -> list[str]:
        """Run segmentation on multiple videos and optionally write per-video mask files.

        Parameters
        ----------
        input_videos : list of str
            Paths to the input video files.
        output_path : str, optional
            Directory for per-video `<name>_mask.txt` files. If None, nothing is written
            and the returned list is empty.
        is_overwrite : bool, optional
            If False (default), existing mask files are skipped.
        is_report : bool, optional
            If True (default), skipped existing files are still included in the result.
        verbose : bool, optional
            Whether to show progress. Default is True.
        message : str, optional
            Optional message shown in each progress bar description.

        Returns
        -------
        list of str
            Paths of mask files that were created or already existed.

        """
        results: list[str] = []
        total_videos = len(input_videos)

        for idx, input_video in enumerate(input_videos, start=1):
            mask_file = None
            if output_path is not None:
                Path(output_path).mkdir(parents=True, exist_ok=True)
                base_filename = os.path.splitext(os.path.basename(input_video))[0]
                mask_file = os.path.join(output_path, f"{base_filename}_mask.txt")

            if (mask_file is not None) and (not is_overwrite) and os.path.exists(mask_file):
                if is_report:
                    results.append(mask_file)
                continue

            self.segment(
                input_video=input_video,
                mask_file=mask_file,
                video_index=idx,
                video_tot=total_videos,
                verbose=verbose,
                message=message,
                return_df=mask_file is None,
            )

            if mask_file is not None:
                results.append(mask_file)

        return results

    get_fps = staticmethod(Detector.get_fps)
    get_frames = staticmethod(Detector.get_frames)

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
        detects = self._predict(cropped_frame)
        cap.release()

        if len(detects) == 0 or detects[0].masks is None:
            return []
        return [x.tolist() for x in detects[0].masks.xy]
