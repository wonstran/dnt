import json
import subprocess
import sys

FORBIDDEN = ("dnt.track", "dnt.detect", "dnt.label", "dnt.filter", "boxmot")


def test_refine_imports_nothing_forbidden():
    code = (
        "import importlib, json, pkgutil, sys\n"
        "import dnt.refine\n"
        "for m in pkgutil.walk_packages(dnt.refine.__path__, 'dnt.refine.'):\n"
        "    importlib.import_module(m.name)\n"
        f"forbidden = {FORBIDDEN!r}\n"
        "bad = sorted(m for m in sys.modules\n"
        "             if any(m == p or m.startswith(p + '.') for p in forbidden))\n"
        "print(json.dumps(bad))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert json.loads(out.stdout.strip().splitlines()[-1]) == []
