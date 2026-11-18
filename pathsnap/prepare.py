"""Development-only inventory, features and independently derived proxy labels."""

from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.sparse.csgraph import connected_components

from .data import load_reference, load_trip, trip_directories
from .features import FEATURE_NAMES, generate_candidates
from .geometry import RoadNetwork
from .reference import align_reference


def prepare(data_root: Path, cache_root: Path, output_root: Path, config: dict) -> None:
    cache_root.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)
    inventory = []
    for split, expected in [("train", 80), ("student_test", 10)]:
        folders = trip_directories(data_root / split)
        if len(folders) != expected:
            raise ValueError(f"Expected {expected} {split} trips, found {len(folders)}")
        for folder in folders:
            start = perf_counter()
            fingerprints = {ext: sha256((folder / f"{folder.name}.{ext}").read_bytes()).hexdigest()
                            for ext in ["track", "nodes", "arcs", "route"]}
            signature = sha256(json.dumps({"raw": fingerprints, "config": config, "features": FEATURE_NAMES,
                                           "cache_version": 1}, sort_keys=True).encode()).hexdigest()
            destination = cache_root / split / f"{folder.name}.npz"
            metadata_path = destination.with_suffix(".json")
            if destination.exists() and metadata_path.exists():
                metadata = json.loads(metadata_path.read_text())
                if metadata.get("signature") == signature:
                    inventory.append(metadata)
                    continue
            trip = load_trip(folder)
            network = RoadNetwork(trip)
            reference = load_reference(folder, len(trip.arcs))
            candidates = generate_candidates(network, config["candidate_count"], config["candidate_radius_m"],
                                               config["index_spacing_m"])
            alignment = align_reference(network, reference)
            labels = (candidates.arc_ids == alignment.arc_ids[:, None]) & candidates.valid
            destination.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(destination, features=candidates.features, arcs=candidates.arc_ids,
                                fraction=candidates.fraction, distance=candidates.distance_m,
                                fallback=candidates.fallback, labels=labels,
                                reference_arcs=alignment.arc_ids, reference_distance=alignment.distance_m)
            dt = np.diff(trip.track[:, 2])
            components, _ = connected_components(network.graph, directed=True, connection="weak")
            degrees = network.indegree + network.outdegree
            metadata = {
                "split": split, "trip": trip.trip_id, "gps_observations": len(trip.track),
                "nodes": len(trip.nodes), "arcs": len(trip.arcs), "reference_route_arcs": len(reference),
                "missing_nonfinite": 0, "non_increasing_timestamps": 0,
                "max_gap_s": float(dt.max()), "one_second_steps": int(np.sum(dt == 1)),
                "total_steps": len(dt), "gaps_over_60s": int(np.sum(dt > 60)),
                "weak_components": int(components), "isolated_nodes": int(np.sum(degrees == 0)),
                "zero_length_arcs": int(np.sum(network.lengths == 0)),
                "reference_endpoint_breaks": int(np.sum(trip.arcs[reference[:-1], 1] != trip.arcs[reference[1:], 0])),
                "proxy_candidate_miss_rate": float(np.mean(~labels.any(axis=1))),
                "candidate_fallback_rate": float(candidates.fallback.mean()),
                "proxy_label_distance_median_m": float(np.median(alignment.distance_m)),
                "proxy_label_distance_p95_m": float(np.quantile(alignment.distance_m, 0.95)),
                "training_quality_excluded_observations": int(np.sum(alignment.distance_m > config["training_label_max_distance_m"])),
                "longitude_extent": [float(trip.nodes[:, 0].min()), float(trip.nodes[:, 0].max())],
                "latitude_extent": [float(trip.nodes[:, 1].min()), float(trip.nodes[:, 1].max())],
                "network_extent_km": (np.ptp(network.xy, axis=0) / 1000).tolist(),
                "raw_sha256": fingerprints, "signature": signature,
                "preparation_seconds": perf_counter() - start,
            }
            metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
            inventory.append(metadata)
            print(f"Prepared {split}/{folder.name}: {len(trip.track)} GPS, miss={metadata['proxy_candidate_miss_rate']:.3f}", flush=True)
    (output_root / "inventory.json").write_text(json.dumps(inventory, indent=2) + "\n")


def training_arrays(cache_root: Path, config: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs, ys, groups = [], [], []
    for file in sorted((cache_root / "train").glob("*.npz")):
        with np.load(file) as data:
            selected = (np.arange(len(data["arcs"])) % config["training_stride"] == 0)
            selected &= data["reference_distance"] <= config["training_label_max_distance_m"]
            valid = (data["arcs"] >= 0) & selected[:, None]
            xs.append(data["features"][valid])
            ys.append(data["labels"][valid].astype(np.int8))
            groups.append(np.full(int(valid.sum()), file.stem))
    if len(xs) != 80:
        raise ValueError("Prepare all 80 training trips before fitting")
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(groups)
