# ---------------------------------------------------------
# IOU Tracker
# Copyright (c) 2017 TU Berlin, Communication Systems Group
# Licensed under The MIT License [see LICENSE for details]
# Written by Erik Bochinski
# ---------------------------------------------------------

"""Utilities for loading class names and creating class mappings.

Provides functions to load class definitions from text files and create
dictionaries and lists for mapping class indices to names.
"""

from pathlib import Path


def load_class_dict(name_file: str | None = None) -> dict[str, int]:
    """Load class names from file and create name-to-index mapping.

    Parameters
    ----------
    name_file : str | None, optional
        Path to class names file (one class per line). If None (default),
        uses 'coco.names' from the shared/data directory.

    Returns
    -------
    dict[str, int]
        Mapping from class name (str) to class index (int).
        Example: {'person': 0, 'car': 2, 'dog': 16}

    Raises
    ------
    FileNotFoundError
        If the class names file does not exist.
    ValueError
        If the file is empty or cannot be parsed.

    Notes
    -----
    The class names file should contain one class name per line.
    The class index is determined by the line number (0-indexed).
    Empty lines are skipped.

    Examples
    --------
    >>> class_dict = load_class_dict()
    >>> idx = class_dict['person']
    >>> print(idx)  # Output: 0

    """
    if not name_file:
        lib_root = Path(__file__).resolve().parents[1]
        name_file = lib_root / "shared/data/coco.names"
    else:
        name_file = Path(name_file)

    if not name_file.exists():
        raise FileNotFoundError(f"Class names file not found: {name_file}")

    with name_file.open("r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines()]

    if not lines:
        raise ValueError(f"Class names file is empty: {name_file}")

    class_dict = {name: idx for idx, name in enumerate(lines) if name}
    return class_dict


def load_classes(name_file: str | None = None) -> list[str]:
    """Load class names from file and return as list.

    Parameters
    ----------
    name_file : str | None, optional
        Path to class names file (one class per line). If None (default),
        uses 'coco.names' from the shared/data directory.

    Returns
    -------
    list[str]
        List of class names in order. Empty strings are filtered out.
        Example: ['person', 'bicycle', 'car', ..., 'toothbrush']

    Raises
    ------
    FileNotFoundError
        If the class names file does not exist.
    ValueError
        If the file is empty.

    Notes
    -----
    The class index in the returned list corresponds to the order of class
    names in the input file (0-indexed). Lines with only whitespace are
    treated as empty and excluded from the result.

    Examples
    --------
    >>> classes = load_classes()
    >>> print(classes[0])  # Output: 'person'
    >>> print(len(classes))  # Output: 80 (for COCO)

    """
    if not name_file:
        lib_root = Path(__file__).resolve().parents[1]
        name_file = lib_root / "shared/data/coco.names"
    else:
        name_file = Path(name_file)

    if not name_file.exists():
        raise FileNotFoundError(f"Class names file not found: {name_file}")

    with name_file.open("r", encoding="utf-8") as f:
        results = [line.strip() for line in f.readlines() if line.strip()]

    if not results:
        raise ValueError(f"Class names file is empty: {name_file}")

    return results
