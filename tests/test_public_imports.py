import importlib
from pathlib import Path

import TorchQML


def test_import_all_public_modules():
    root = Path(TorchQML.__file__).resolve().parent
    skipped = {"__pycache__"}
    failures = []

    for path in root.rglob("*.py"):
        rel = path.relative_to(root)
        if any(part in skipped for part in rel.parts):
            continue
        module_parts = ("TorchQML", *rel.with_suffix("").parts)
        if module_parts[-1] == "__init__":
            module_parts = module_parts[:-1]
        module_name = ".".join(module_parts)
        try:
            importlib.import_module(module_name)
        except Exception as exc:  # pragma: no cover - failure message matters.
            failures.append(f"{module_name}: {type(exc).__name__}: {exc}")

    assert failures == []
