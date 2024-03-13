from pathlib import Path
from typing import Any
import re

import pint


def is_valid_unit(unit: str, dimensionality: str) -> bool:
    """Verifies that the unit measure has the correct dimensionality"""
    return pint.Unit(unit).dimensionality == dimensionality


def regex_file_list(path: Path, pattern: str) -> list[Path]:
    """Returns a list of files in path, whose regexp matches the pattern"""
    p = path.glob("**/*")
    return [x for x in p if x.is_file() and re.search(".*" + pattern, f"{x}")]


def get_from_id(obj: list, attrib: Any, index: Any) -> Any:
    """Get an element from a homogeneous list, given its attribute"""
    attrib_list = [getattr(value, attrib) for value in obj]
    try:
        object_index = attrib_list.index(index)
    except ValueError:
        raise IndexError(f"Index {index} not found in list:\n\t{attrib_list}")
    return obj[object_index]


def is_id_unique(obj: list, attrib: str = "id") -> bool:
    """Validates that an attribute of elements of a list is unique"""
    id_list = [getattr(value, attrib) for value in obj]
    return len(id_list) == len(set(id_list))
