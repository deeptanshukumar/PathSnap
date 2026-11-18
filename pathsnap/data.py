"""Raw inputs deliberately exclude reference routes from the prediction object."""

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np


@dataclass(frozen=True)
class Trip:
    trip_id: str
    track: np.ndarray  # [observation, longitude/latitude/time_seconds]
    nodes: np.ndarray  # [node, longitude/latitude]
    arcs: np.ndarray  # [arc, source/target_node]


def trip_directories(folder: Path) -> list[Path]:
    if not folder.is_dir():
        raise FileNotFoundError(f"Trip folder does not exist: {folder}")
    return sorted(p for p in folder.iterdir() if p.is_dir() and re.fullmatch(r"\d{8}", p.name))


def load_trip(folder: Path) -> Trip:
    """Validate raw input; no route file is opened, even if one exists."""
    name = folder.name
    if not re.fullmatch(r"\d{8}", name):
        raise ValueError("Trip directories must have eight-digit IDs")
    arrays = {}
    for extension, columns in [("track", 3), ("nodes", 2), ("arcs", 2)]:
        values = np.loadtxt(folder / f"{name}.{extension}", ndmin=2)
        if values.shape[1] != columns or not len(values) or not np.isfinite(values).all():
            raise ValueError(f"Invalid {extension} shape or nonfinite data in {name}")
        arrays[extension] = values
    for key in ["track", "nodes"]:
        coords = arrays[key][:, :2]
        if (np.abs(coords[:, 0]) > 180).any() or (np.abs(coords[:, 1]) >= 90).any():
            raise ValueError(f"Implausible longitude/latitude in {name}.{key}")
    if len(arrays["track"]) < 2 or (np.diff(arrays["track"][:, 2]) <= 0).any():
        raise ValueError(f"At least two GPS points with increasing timestamps required: {name}")
    arcs = arrays["arcs"]
    if (arcs != np.floor(arcs)).any() or arcs.min() < 0 or arcs.max() >= len(arrays["nodes"]):
        raise ValueError(f"Invalid node references in {name}.arcs")
    return Trip(name, arrays["track"], arrays["nodes"], arcs.astype(np.int64))


def load_reference(folder: Path, arc_count: int) -> np.ndarray:
    """Training/scoring only: never call this inside prediction or features."""
    raw = np.loadtxt(folder / f"{folder.name}.route", ndmin=1)
    if raw.ndim != 1 or not len(raw) or not np.isfinite(raw).all():
        raise ValueError("Reference must be a nonempty arc sequence")
    if (raw != np.floor(raw)).any() or raw.min() < 0 or raw.max() >= arc_count:
        raise ValueError("Reference has invalid arc IDs")
    return raw.astype(np.int64)
