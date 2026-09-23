import subprocess
import sys

STRAY = ("detector", "shared", "filter", "tracker", "track", "labeler", "post_process",
         "yolo", "engine", "synhcro", "util", "download", "segmentor")


def test_no_stray_top_level_modules():
    code = (
        "import sys, dnt, dnt.detect, dnt.detect.signal, dnt.track, dnt.label, dnt.filter, dnt.shared, dnt.engine\n"
        f"print(sorted(m for m in {STRAY!r} if m in sys.modules))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.strip()
    assert out == "[]"


def test_labeler_uses_package_util():
    from dnt.label.labeler import load_classes

    assert load_classes.__module__ == "dnt.shared.util"
