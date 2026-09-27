"""Detection wrapper around Ultralytics YOLO/RT-DETR for video frames.

Revised by wonstran
    01/28/2026
    11/11/2025.
"""

import os
from collections.abc import Callable
from enum import Enum
from pathlib import Path
from time import time

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from ultralytics import RTDETR, YOLO

from ..._device import half_allowed, predict_precision_kwargs, resolve_device


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


def _coerce_model(model: "DetectorModel | str") -> DetectorModel:
    """Accept a DetectorModel, its value ('rtdetr-x.pt'), its stem ('rtdetr-x') or its name ('RTDETRx')."""
    if isinstance(model, DetectorModel):
        return model
    text = str(model).strip().lower()
    for candidate in DetectorModel:
        if text in {candidate.value.lower(), candidate.value.lower().removesuffix(".pt"), candidate.name.lower()}:
            return candidate
    valid = [m.value.removesuffix(".pt") for m in DetectorModel]
    raise ValueError(f"Unknown detector model {model!r}. Valid: {valid}")


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
    model : DetectorModel or str, optional
        Built-in model weights to use. Accepts a `DetectorModel` member
        (`DetectorModel.YOLO26x`), its value / file name (`"yolo26x.pt"`),
        its stem (`"yolo26x"`), or its enum name (`"YOLO26x"`).
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
    imgsz : int, optional
        Inference image size (longest side, pixels), forwarded to Ultralytics
        `model.predict(imgsz=...)`. Default is `640`, Ultralytics' own default.
    classes : list of int, optional
        Restrict detection to these class indices, forwarded to Ultralytics
        `model.predict(classes=...)`. Default is `None` (all classes), Ultralytics'
        own default.
    class_names : list of str, optional
        Restrict detection to these class names (e.g. `["car", "truck", "bus"]`),
        resolved against `dnt.shared.util.load_class_dict()` (COCO names, matching
        the stock YOLO/RT-DETR weights) and merged into `classes`. Default is `None`.
    rect : bool, optional
        Whether to use rectangular (non-square) inference, forwarded to Ultralytics
        `model.predict(rect=...)`. Default is `False`, Ultralytics' own default.
    agnostic_nms : bool, optional
        Whether to use class-agnostic NMS, forwarded to Ultralytics
        `model.predict(agnostic_nms=...)`. Default is `False`, Ultralytics' own default.
    embed : list of int, optional
        Layer indices to return feature embeddings from, forwarded to Ultralytics
        `model.predict(embed=...)`. Default is `None` (no embeddings), Ultralytics' own
        default.
    fast : bool, optional
        Whether `detect()` redirects to `detect_fast(batch=self.batch)` (`True`, the
        default) or runs the traditional one-frame-at-a-time pipeline (`False`). Mutate
        `detector.fast` after construction to change it for later calls. See
        `detect_fast()` for the speed/behavior trade-off `fast=True` makes.
    batch : int, optional
        Frames per forward pass, used when `fast` is True: by `detect()`'s redirect, and
        as the default for direct `detect_fast()` calls that don't pass their own
        `batch=`. Default is `8`.

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
        model: DetectorModel | str = DetectorModel.YOLO26x,
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
        embed: list[int] | None = None,
        fast: bool = True,
        batch: int = 8,
    ):
        """Initialize a Detector for Ultralytics YOLO/RT-DETR models.

        Parameters
        ----------
        model : DetectorModel or str, optional
            Built-in model to use. Accepts a `DetectorModel` member, its value /
            file name (`"yolo26x.pt"`), its stem (`"yolo26x"`), or its enum name
            (`"YOLO26x"`). Default is "yolo26x".
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
        imgsz : int, optional
            Inference image size, as passed to Ultralytics `model.predict()`.
            Default is 640 (Ultralytics' own default).
        classes : list of int, optional
            Class indices to restrict detection to, as passed to Ultralytics
            `model.predict()`. Default is None, i.e. all classes (Ultralytics' own
            default).
        class_names : list of str, optional
            Class names to restrict detection to (e.g. `["car", "truck"]`), resolved
            via `dnt.shared.util.load_class_dict()` (COCO names) and merged into
            `classes`. Raises `ValueError` for an unknown name. Default is None.
        rect : bool, optional
            Rectangular inference, as passed to Ultralytics `model.predict()`.
            Default is False (Ultralytics' own default).
        agnostic_nms : bool, optional
            Class-agnostic NMS, as passed to Ultralytics `model.predict()`.
            Default is False (Ultralytics' own default).
        embed : list of int, optional
            Layer indices to return feature embeddings from, as passed to
            Ultralytics `model.predict()`. Default is None, i.e. no embeddings
            (Ultralytics' own default).
        fast : bool, optional
            Whether `detect()` redirects to `detect_fast(batch=self.batch)` by default.
            Default is True. See `detect_fast()` for the trade-off this makes; set
            `detector.fast = False` after construction (or per instance here) to keep
            `detect()` on the traditional per-frame pipeline.
        batch : int, optional
            Default frames per forward pass when `fast` is True. Default is 8.

        """
        model = _coerce_model(model)

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
        self.imgsz = imgsz
        if class_names:
            # deferred: dnt.shared imports dnt.detect (Synchronizer), so importing at module level would cycle
            from ...shared.util import load_class_dict

            name_to_idx = load_class_dict()
            try:
                resolved = {name_to_idx[name] for name in class_names}
            except KeyError as e:
                raise ValueError(
                    f"Unknown class name {e.args[0]!r}. "
                    "See dnt.shared.util.load_class_dict() for valid names."
                ) from e
            classes = sorted({*(classes or []), *resolved})
        self.classes = classes
        self.rect = rect
        self.agnostic_nms = agnostic_nms
        self.embed = embed
        self.fast = fast
        self.batch = batch

        self.device = resolve_device(device)
        # half precision only makes sense on CUDA (any index)
        self.half = half_allowed(self.device, half)
        # translates to quantize= on Ultralytics versions that deprecated half= (avoids a
        # per-frame deprecation warning); falls back to half= on older versions
        self._precision_kwargs = predict_precision_kwargs(self.half)

    def _results_to_df(self, results: list[dict]) -> pd.DataFrame:
        """Convert per-box dicts (frame,res,x,y,x2,y2,conf,class) to DET_FIELDS with detect()'s rounding."""
        if not results:
            return pd.DataFrame(columns=self.DET_FIELDS)
        df = pd.DataFrame(results, columns=["frame", "res", "x", "y", "x2", "y2", "conf", "class"])
        df["w"] = (df["x2"] - df["x"]).astype(int)
        df["h"] = (df["y2"] - df["y"]).astype(int)
        df["x"] = df["x"].astype(int)
        df["y"] = df["y"].astype(int)
        df["conf"] = df["conf"].round(2)
        return df[self.DET_FIELDS].reset_index(drop=True)

    def _arrays_to_df(self, frame_ids: list[int], arrays: list[np.ndarray]) -> pd.DataFrame:
        """Vectorized `_results_to_df()` for per-frame `(N, 6)` box arrays.

        Each array holds `x1, y1, x2, y2, conf, cls` for one frame in `frame_ids`. Output
        dtypes and rounding match `_results_to_df()` exactly, so both write identical CSV
        text; this form just avoids one Python dict per box (about 431 vs 64 bytes each).
        """
        counts = [len(a) for a in arrays]
        n = sum(counts)
        if n == 0:
            return pd.DataFrame(columns=self.DET_FIELDS)
        data = np.concatenate(arrays).astype(np.float64)
        x, y, x2, y2 = data[:, 0], data[:, 1], data[:, 2], data[:, 3]
        return pd.DataFrame({
            "frame": np.repeat(np.asarray(frame_ids, dtype=np.int64), counts),
            "res": np.full(n, -1, dtype=np.int64),
            "x": x.astype(np.int64),
            "y": y.astype(np.int64),
            "w": (x2 - x).astype(np.int64),
            "h": (y2 - y).astype(np.int64),
            "conf": np.round(data[:, 4], 2),
            "class": data[:, 5].astype(np.int64),
        })

    @staticmethod
    def _boxes_to_arrays(results: list) -> list[np.ndarray]:
        """Per-frame `(N, 6)` float32 arrays `x1, y1, x2, y2, conf, cls` from Ultralytics Results."""
        out = []
        for r in results:
            boxes = r.boxes
            if boxes is None or len(boxes) == 0:
                out.append(np.empty((0, 6), dtype=np.float32))
                continue
            d = boxes.data.cpu().numpy()  # Boxes.xyxy/.conf/.cls are data[:, :4]/[:, -2]/[:, -1]
            out.append(np.column_stack([d[:, :4], d[:, -2], d[:, -1]]))
        return out

    @staticmethod
    def _predict_arrays(predictor, im0s: list[np.ndarray], path: str) -> list[np.ndarray]:
        """Run one forward pass over `im0s` with Ultralytics' own pre/postprocessing."""
        im = predictor.preprocess(im0s)
        # postprocess() needs a path per image (only used for e.g. Results.plot() / save,
        # both unused here); callers track the real per-frame identity themselves.
        predictor.batch = ([path] * len(im0s), im0s, [""] * len(im0s))
        with torch.inference_mode():
            preds = predictor.inference(im)
        return Detector._boxes_to_arrays(predictor.postprocess(preds, im, im0s))

    @staticmethod
    def _resume_part_file(part: Path) -> int | None:
        """Trim a partial output file for resuming; return the frame to restart from.

        Drops a trailing partial line and every row of the file's last frame (that frame
        may have been cut off mid-write), so it is simply detected again. Only the file's
        tail is read, so this is cheap even for a multi-GB file. Returns None, leaving an
        empty file, when no complete row exists.
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
                first_nl = buf.find(b"\n")
                if start > 0 and first_nl < 0:
                    block *= 2
                    continue
                pos = 0 if start == 0 else first_nl + 1  # a block's first line may be partial
                entries = []  # (absolute offset, frame) of complete lines
                while (nl := buf.find(b"\n", pos)) >= 0:
                    entries.append((start + pos, int(buf[pos : buf.index(b",", pos)])))
                    pos = nl + 1
                if not entries:
                    if start == 0:
                        f.truncate(0)
                        return None
                    block *= 2
                    continue
                last = entries[-1][1]
                i = len(entries) - 1
                while i > 0 and entries[i - 1][1] == last:
                    i -= 1
                if i == 0 and start > 0:  # the last frame's rows may start before this block
                    block *= 2
                    continue
                f.truncate(entries[i][0])
                return last

    def _run_batched(
        self,
        input_video: str,
        prepare: Callable[[], Callable[[list[np.ndarray]], list[np.ndarray]]],
        *,
        iou_file: str | None = None,
        video_index: int | None = None,
        video_tot: int | None = None,
        start_frame: int | None = None,
        end_frame: int | None = None,
        verbose: bool = True,
        message: str | None = None,
        batch: int = 8,
        return_df: bool = True,
    ) -> pd.DataFrame | None:
        """Frame loop of `detect_fast()`: batched inference, streamed output and resume.

        `prepare()` is called once the video is open and returns `infer(frames)`, which
        maps up to `batch` BGR frames to per-frame `(N, 6)` box arrays. Each batch's rows
        are appended to `<iou_file>.part` as soon as they are computed; the file is renamed
        to `iou_file` only when the whole range is done, so a finished `iou_file` is always
        complete. If `<iou_file>.part` already exists, the run resumes from it (see
        `_resume_part_file()`).
        """
        if batch < 1:
            msg = "batch must be >= 1."
            raise ValueError(msg)
        if not return_df and not iou_file:
            msg = "return_df=False needs an iou_file; otherwise the detections would be discarded."
            raise ValueError(msg)

        cap = cv2.VideoCapture(input_video)
        if not cap.isOpened():
            if verbose:
                print(f"Cannot open video: {input_video}")
            return pd.DataFrame(columns=self.DET_FIELDS) if return_df else None

        tot_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if start_frame is None or start_frame < 0 or start_frame >= tot_frames:
            start_frame = 0
        if end_frame is None or end_frame < 0 or end_frame >= tot_frames:
            end_frame = tot_frames - 1
        if start_frame > end_frame:
            cap.release()
            raise ValueError("start_frame must be less than or equal to end_frame.")

        part = None
        resumed = False
        if iou_file:
            Path(iou_file).parent.mkdir(parents=True, exist_ok=True)
            part = Path(f"{iou_file}.part")
            if part.exists():
                resumed = True
                resume_at = self._resume_part_file(part)
                if resume_at is not None:
                    start_frame = max(start_frame, resume_at)
                if verbose:
                    print(f"Resuming {iou_file} from frame {start_frame} (delete {part} to start over).")

        frame_total = max(end_frame - start_frame + 1, 0)
        pbar = None
        if verbose:
            if (video_index is not None) and (video_tot is not None):
                desc = f"Detecting {video_index} of {video_tot}"
            else:
                desc = "Detecting"
            if message is not None:
                desc += f" {message}"
            pbar = tqdm(total=frame_total, desc=desc, unit="frame") if tot_frames > 0 else tqdm(desc=desc, unit="frame")

        chunks: list[pd.DataFrame] = []
        im0s: list[np.ndarray] = []
        frame_ids: list[int] = []
        out = part.open("a") if part is not None else None
        try:
            infer = prepare() if frame_total > 0 else None

            def flush() -> None:
                """Detect the buffered frames and write/keep their rows."""
                if not im0s:
                    return
                chunk = self._arrays_to_df(frame_ids, infer(im0s))
                if len(chunk):
                    if out is not None:
                        chunk.to_csv(out, index=False, header=False)
                        out.flush()
                    if return_df:
                        chunks.append(chunk)
                im0s.clear()
                frame_ids.clear()

            if frame_total > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                while cap.isOpened():
                    pos_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                    ret, frame = cap.read()
                    if not ret or pos_frame > end_frame:
                        break
                    im0s.append(frame)
                    frame_ids.append(pos_frame)
                    if len(im0s) >= batch:
                        flush()
                    if pbar is not None:
                        pbar.update(1)
                flush()  # trailing partial batch
        finally:
            cap.release()
            if pbar is not None:
                pbar.close()
            if out is not None:
                out.close()

        if part is not None:
            os.replace(part, iou_file)
        if not return_df:
            return None
        if resumed:  # rows from the earlier run are only in the file
            try:
                return pd.read_csv(iou_file, header=None, names=self.DET_FIELDS)
            except pd.errors.EmptyDataError:
                return pd.DataFrame(columns=self.DET_FIELDS)
        if not chunks:
            return pd.DataFrame(columns=self.DET_FIELDS)
        return pd.concat(chunks, ignore_index=True)

    def _ensure_predictor(self):
        """Return an Ultralytics predictor configured exactly like `detect()`'s per-frame calls.

        `detect_fast()` uses this so it matches `detect()`'s NMS/class/threshold behavior
        without going through `model.predict()` for every frame (see `detect_fast()`).

        On the first call (or after `self.device` changes), this runs one throwaway
        `model.predict()` call on a dummy frame. That is the same mechanism Ultralytics
        itself uses to build `self.model.predictor` and its underlying `AutoBackend`, and
        it also performs the one-time model warmup. On later calls, it only re-merges the
        current `conf`/`nms`/`classes`/etc. into the existing predictor's args (cheap: no
        model or dataset rebuild), so changes to those attributes after construction still
        take effect, as they do for `detect()`.
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
            "embed": self.embed,
            "verbose": False,
            **self._precision_kwargs,
        }
        predictor = self.model.predictor
        if predictor is None or resolve_device(predictor.args.device) != resolve_device(self.device):
            dummy = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
            self.model.predict(source=dummy, **args)
            predictor = self.model.predictor
        else:
            # deferred: internal Ultralytics helper, mirrors what Model.predict() itself
            # does to refresh an already-set-up predictor's args (engine/model.py).
            from ultralytics.cfg import get_cfg

            predictor.args = get_cfg(predictor.args, args)
        return predictor

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
        message: str | None = None,
        return_df: bool = True,
    ) -> pd.DataFrame | None:
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
            Not supported when `self.fast` is True (`detect_fast()` has no preview); set
            `detector.fast = False` to use `show`.
        message : str | None, optional
            Optional message shown in the progress bar description.
            Default is None.
        return_df : bool, optional
            Whether to return the detections. Default is True. With False, detections are
            only written to `iou_file` (which is then required) and None is returned; for
            long videos this avoids holding every detection in memory, since the `fast`
            path streams rows to `iou_file` as it goes.

        Returns
        -------
        pandas.DataFrame or None
            DataFrame with columns:
            `frame, res, x, y, w, h, conf, class`.
            If the video cannot be opened or no detections are found, an empty DataFrame
            with those columns is returned. None if `return_df` is False.

        Notes
        -----
        With `self.fast`, `iou_file` is written incrementally as
        `<iou_file>.part` and renamed when complete; an interrupted run resumes from that
        file on the next call (see `detect_fast()`).

        """
        if not return_df and not iou_file:
            msg = "return_df=False needs an iou_file; otherwise the detections would be discarded."
            raise ValueError(msg)
        if self.fast:
            if show:
                msg = (
                    "detect(show=True) is not supported while self.fast is True: "
                    "detect_fast() has no preview. Set detector.fast = False for a live preview."
                )
                raise ValueError(msg)
            return self.detect_fast(
                input_video,
                iou_file=iou_file,
                video_index=video_index,
                video_tot=video_tot,
                start_frame=start_frame,
                end_frame=end_frame,
                verbose=verbose,
                message=message,
                batch=self.batch,
                return_df=return_df,
            )

        # validate path
        cap = cv2.VideoCapture(input_video)
        if not cap.isOpened():
            if verbose:
                print(f"Cannot open video: {input_video}")
            return pd.DataFrame(columns=self.DET_FIELDS) if return_df else None

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
            if (video_index is not None) and (video_tot is not None):
                desc = f"Detecting {video_index} of {video_tot}"
            else:
                desc = "Detecting"

            if message is not None:
                desc += f" {message}"

            if tot_frames <= 0:
                pbar = tqdm(desc=desc, unit="frame")
            else:
                pbar = tqdm(total=frame_total, desc=desc, unit="frame")

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
                imgsz=self.imgsz,
                classes=self.classes,
                rect=self.rect,
                agnostic_nms=self.agnostic_nms,
                embed=self.embed,
                **self._precision_kwargs,
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
            return empty_df if return_df else None

        results_df = self._results_to_df(results)

        if iou_file:
            folder = Path(iou_file).parent
            if not folder.exists():
                Path(folder).mkdir(parents=True, exist_ok=True)

            results_df.to_csv(iou_file, index=False, header=False)

        return results_df if return_df else None

    def detect_fast(
        self,
        input_video: str,
        iou_file: str | None = None,
        video_index: int | None = None,
        video_tot: int | None = None,
        start_frame: int | None = None,
        end_frame: int | None = None,
        verbose: bool = True,
        message: str | None = None,
        batch: int | None = None,
        return_df: bool = True,
    ) -> pd.DataFrame | None:
        """Like `detect()`, but batches frames through the Ultralytics `AutoBackend` directly.

        `detect()` calls Ultralytics' high-level `model.predict()` once per frame. That call
        rebuilds an inference dataset around the single frame and forces several CUDA
        synchronizations on every call, regardless of batch size, which is roughly 2x slower
        than running several frames through the backend in one forward pass. This method
        does that instead: it reuses Ultralytics' own model-family-correct
        `preprocess()`/`inference()`/`postprocess()` (so decoding, e.g. RT-DETR's vs YOLO's
        NMS, is unchanged), just without `model.predict()`'s per-frame dataset/callback setup.

        `batch` here means frames within this one video, unlike `detect_batch()`'s `batch`,
        which means a list of videos.

        Parameters
        ----------
        input_video : str
            Path to the input video file.
        iou_file : str, optional
            If provided, detection results are written to this file (CSV without header),
            batch by batch (see Notes).
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
        message : str | None, optional
            Optional message shown in the progress bar description.
            Default is None.
        batch : int, optional
            Number of frames sent through the model in one forward pass. Default is None:
            use `self.batch` (see `Detector`). Larger values usually raise GPU throughput
            at the cost of more GPU memory and higher latency per flush; tune it to your
            GPU and frame resolution.
        return_df : bool, optional
            Whether to return the detections. Default is True. With False, rows are only
            written to `iou_file` (then required) and None is returned, so memory stays
            flat however long the video is.

        Returns
        -------
        pandas.DataFrame or None
            Same columns and semantics as `detect()`: `frame, res, x, y, w, h, conf, class`.
            If the video cannot be opened or no detections are found, an empty DataFrame
            with those columns is returned. None if `return_df` is False.

        Notes
        -----
        - No `show=` preview: batching trades per-frame latency for throughput, so a live
          preview does not fit this path. Use `detect()` for that.
        - `batch=1` reproduces `detect()` exactly (verified frame-for-frame, CPU and CUDA,
          YOLO and RT-DETR): the speed gain at `batch=1` comes purely from skipping
          `model.predict()`'s per-frame dataset/callback setup (see `_ensure_predictor()`),
          not from any change in what gets computed.
        - `batch > 1` on CUDA with `half=True` can shift a small fraction of results:
          running the same image alone vs. as part of a larger batch takes a different
          path through cuDNN/cuBLAS's FP16 kernels, so reduction order (and, rarely, which
          of two near-duplicate boxes NMS keeps) can differ slightly. Measured on an
          RTX 5070 Ti, RT-DETR-x, 200 frames, `batch=8`: `half=True` shifted results on 6%
          of frames (up to ~0.17 confidence, ~67 px on the frames it did shift); `half=False`
          shifted only 2.5% of frames (up to 0.02 confidence, 1 px, i.e. rounding noise).
          This is inherent to batched half-precision GPU inference, not specific to this
          bypass; `detect()` called once per frame never hits it, since it never batches.
          If exact reproducibility with `detect()` matters more than throughput, use
          `half=False` and/or `batch=1`.

        - With `iou_file`, each batch's rows are appended to `<iou_file>.part` as soon as
          they are detected, and the file is renamed to `iou_file` only once the whole range
          is done, so an existing `iou_file` is always complete (and `detect_batch()` can
          keep skipping existing files). If `<iou_file>.part` already exists, e.g. after a
          crash or Ctrl-C, detection resumes from it: the rows of its last frame are dropped
          and that frame is detected again. Delete the `.part` file to start over instead.
          With `batch > 1` and `half=True`, frames after the resume point are grouped into
          batches differently than in an uninterrupted run, so they can differ slightly in
          the way described above.

        """
        batch = self.batch if batch is None else batch

        def prepare():
            predictor = self._ensure_predictor()
            return lambda im0s: self._predict_arrays(predictor, im0s, input_video)

        return self._run_batched(
            input_video,
            prepare,
            iou_file=iou_file,
            video_index=video_index,
            video_tot=video_tot,
            start_frame=start_frame,
            end_frame=end_frame,
            verbose=verbose,
            message=message,
            batch=batch,
            return_df=return_df,
        )

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
                imgsz=self.imgsz,
                classes=self.classes,
                rect=self.rect,
                agnostic_nms=self.agnostic_nms,
                **self._precision_kwargs,
                embed=self.embed,
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

        return self._results_to_df(results)

    def detect_batch(
        self,
        input_videos: list[str],
        output_path: str | None = None,
        is_overwrite: bool = False,
        is_report: bool = True,
        verbose: bool = True,
        message: str | None = None,
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
        message : str | None, optional
            Optional message shown in each progress bar description.
            Default is None.

        Returns
        -------
        list of str
            A list of paths to detection files that were created or already existed.
            If `output_path` is None, this will be an empty list.

        Notes
        -----
        - Each video's `detect()` call uses this `Detector` instance's `fast`/`batch` (see
          `Detector`), the same as calling `detect()` directly; there is no separate
          argument for them here.
        - A video whose `_iou.txt` doesn't exist yet but whose `_iou.txt.part` does (an
          interrupted `fast` run) resumes from the `.part` file; with
          `is_overwrite=True` the `.part` file is deleted and the video starts over.

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

            # regenerating: don't resume from an earlier, possibly different, partial run
            if iou_file is not None and is_overwrite:
                Path(f"{iou_file}.part").unlink(missing_ok=True)

            # run detection (may write to iou_file if not None); the returned table is
            # discarded here, so don't build it when the rows go to a file anyway
            self.detect(
                input_video=input_video,
                iou_file=iou_file,
                video_index=idx,
                video_tot=total_videos,
                verbose=verbose,
                message=message,
                return_df=iou_file is None,
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
