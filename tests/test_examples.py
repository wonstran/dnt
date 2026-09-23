import ast
from pathlib import Path

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
REMOVED = {"example_batch.py", "example_filter.py", "example_stop.py", "example_shapes.py"}


def test_removed_examples_are_gone():
    assert not REMOVED & {p.name for p in EXAMPLES.glob("*.py")}


def test_examples_compile_and_configs_use_keywords():
    for path in EXAMPLES.glob("*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and getattr(node.func, "id", "").endswith("Config"):
                assert not node.args, f"{path.name}:{node.lineno} positional config argument"
