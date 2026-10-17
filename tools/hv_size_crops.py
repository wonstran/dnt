"""Save one crop per vehicle track, largest first, to help choose ``VehReClass(hv_size=...)``.

    .venv/bin/python tools/hv_size_crops.py --tracks T --video V --out DIR
        [--hv-min-count 3] [--min-size 0] [--limit 80]

A track's size is the one ``VehReClass`` uses: the ``hv_min_count``-th largest ``max(w, h)`` over
its real (not interpolated) rows. Each selected track gets one crop of its largest real box,
saved as ``<rank>_size<size>_track<id>.jpg``; ``sizes.csv`` lists every selected track (with an
empty ``crop`` when its frame could not be read), and ``sheet.jpg`` puts the crops on one labeled
contact sheet. Find the gap between the largest vans and pickups and the smallest box trucks,
semi-trailers and buses, and set ``hv_size`` inside it (closer to the vans when unsure: a van
flagged heavy can no longer be car, while a heavy vehicle left unflagged is still asked of the
VLM). No detector is run, so it needs no GPU.
"""

import argparse
import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from dnt.refine.crops import FrameReader, crop_box
from dnt.refine.io import filled_mask
from dnt.track.re_class import _read_frames, _read_tracks, _size

COLUMNS = ["track", "size", "frame", "x", "y", "w", "h"]
CELL_W, CELL_H, LABEL_H, COLS = 240, 160, 24, 8


def select_tracks(
    table: pd.DataFrame, hv_min_count: int = 3, min_size: float = 0.0, limit: int = 80
) -> pd.DataFrame:
    """Return the ``limit`` largest tracks with ``size >= min_size`` and their largest real box."""
    real = table.loc[~filled_mask(table)]
    rows = []
    for t, g in real.groupby("track", sort=False):
        size = _size(g, hv_min_count)
        if math.isnan(size) or size < min_size:
            continue
        best = g.iloc[int(np.argmax((g["w"] * g["h"]).to_numpy(float)))]
        rows.append([t, size, int(best["frame"]), *(float(best[c]) for c in "xywh")])
    sel = pd.DataFrame(rows, columns=COLUMNS)
    sel = sel.sort_values(["size", "track"], ascending=[False, True], kind="stable")
    return sel.head(limit).reset_index(drop=True)


def _id(t) -> str:
    return str(int(t)) if float(t).is_integer() else str(t)


def _sheet(crops: list, labels: list[str]) -> np.ndarray | None:
    """Lay out RGB crops (None for a missing one) in a labeled grid; return a BGR image."""
    if not any(c is not None for c in crops):
        return None
    n_rows = math.ceil(len(crops) / COLS)
    sheet = np.full((n_rows * (CELL_H + LABEL_H), COLS * CELL_W, 3), 40, np.uint8)
    for i, (crop, label) in enumerate(zip(crops, labels, strict=True)):
        x0, y0 = (i % COLS) * CELL_W, (i // COLS) * (CELL_H + LABEL_H)
        if crop is not None:
            h, w = crop.shape[:2]
            scale = min((CELL_W - 8) / w, (CELL_H - 8) / h)
            size = (max(1, round(w * scale)), max(1, round(h * scale)))
            img = cv2.cvtColor(
                cv2.resize(crop, size, interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2BGR
            )
            ox, oy = x0 + (CELL_W - size[0]) // 2, y0 + (CELL_H - size[1]) // 2
            sheet[oy : oy + size[1], ox : ox + size[0]] = img
        cv2.putText(
            sheet,
            label,
            (x0 + 6, y0 + CELL_H + 17),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return sheet


def save_crops(sel: pd.DataFrame, video: str, out: Path) -> pd.DataFrame:
    """Write the crops, ``sizes.csv`` and ``sheet.jpg`` for ``sel`` into ``out``."""
    out.mkdir(parents=True, exist_ok=True)
    want: dict[int, list[int]] = {}
    for i, f in enumerate(sel["frame"]):
        want.setdefault(int(f), []).append(i)
    crops: list = [None] * len(sel)
    names = [""] * len(sel)
    with FrameReader(str(video)) as reader:
        for f, img in _read_frames(reader, want):
            if img is None:
                continue
            for i in want[f]:
                r = sel.iloc[i]
                crop = crop_box(img, (r["x"], r["y"], r["w"], r["h"]))
                if crop is None:
                    continue
                names[i] = f"{i + 1:03d}_size{r['size']:.0f}_track{_id(r['track'])}.jpg"
                cv2.imwrite(str(out / names[i]), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
                crops[i] = crop
    result = sel.assign(crop=names)
    result.insert(0, "rank", range(1, len(sel) + 1))
    result.to_csv(out / "sizes.csv", index=False)
    labels = [f"{s:.0f}px  #{_id(t)}" for s, t in zip(sel["size"], sel["track"], strict=True)]
    sheet = _sheet(crops, labels)
    if sheet is not None:
        cv2.imwrite(str(out / "sheet.jpg"), sheet)
    return result


def main(argv: list[str] | None = None) -> None:
    """Run the command line (see the module docstring)."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--tracks", required=True, help="headerless track file")
    p.add_argument("--video", required=True, help="the video the tracks come from")
    p.add_argument("--out", required=True, type=Path, help="directory for crops and sheet")
    p.add_argument("--hv-min-count", type=int, default=3)
    p.add_argument("--min-size", type=float, default=0.0, help="skip smaller tracks (px)")
    p.add_argument("--limit", type=int, default=80, help="at most this many tracks")
    a = p.parse_args(argv)
    if not Path(a.video).exists():
        p.error(f"video not found: {a.video}")
    sel = select_tracks(_read_tracks(a.tracks), a.hv_min_count, a.min_size, a.limit)
    result = save_crops(sel, a.video, a.out)
    missing = int((result["crop"] == "").sum())
    print(
        f"{len(result)} tracks, sizes {result['size'].max():.0f}-{result['size'].min():.0f} px"
        if len(result)
        else "no tracks selected"
    )
    print(
        f"crops, sizes.csv and sheet.jpg in {a.out}"
        + (f" ({missing} frame(s) could not be read)" if missing else "")
    )


if __name__ == "__main__":
    main()
