from itertools import product
from pathlib import Path

import numpy as np
import pytest

from pathsnap.data import Trip, load_trip
from pathsnap.decoder import Decoded, DecoderConfig, decode, emission_probabilities, reconstruct_route, viterbi
from pathsnap.experiments import grouped_folds
from pathsnap.features import generate_candidates
from pathsnap.geometry import LocalProjection, RoadNetwork, project_segments
from pathsnap.reference import align_reference


def tiny_network() -> RoadNetwork:
    nodes = np.array([[0., 0.], [.001, 0.], [.002, 0.], [.001, .001]])
    arcs = np.array([[0, 1], [1, 2], [1, 0], [2, 1], [1, 3]])
    track = np.array([[.0001, .00001, 0.], [.0008, .00001, 1.], [.0015, .00001, 2.]])
    return RoadNetwork(Trip("00000001", track, nodes, arcs))


def test_projection_units_roundtrip_and_segment_endpoints():
    coordinates = np.array([[77., 13.], [77.001, 13.]])
    projection = LocalProjection.for_coordinates(coordinates)
    xy = projection.forward(coordinates)
    assert 108 < np.linalg.norm(xy[1] - xy[0]) < 109
    np.testing.assert_allclose(projection.inverse(xy), coordinates)
    f, d = project_segments(np.array([[2., 1.], [-1., 0.], [1., 2.]]),
                            np.zeros((3, 2)), np.array([[1., 0.], [1., 0.], [0., 0.]]))
    np.testing.assert_allclose(f, [1., 0., 0.])
    np.testing.assert_allclose(d, [np.sqrt(2), 1., np.sqrt(5)])


def test_candidate_projection_matches_bruteforce_including_long_edges():
    network = tiny_network()
    candidates = generate_candidates(network, count=5, radius_m=300, spacing_m=10)
    for i, point in enumerate(network.gps):
        _, distances = project_segments(point, network.starts, network.vectors)
        expected = np.lexsort((np.arange(len(distances)), distances))
        np.testing.assert_array_equal(candidates.arc_ids[i], expected)
    assert np.isfinite(candidates.features).all()
    assert not candidates.fallback.any()


def test_missing_radius_candidates_are_explicit_and_padding_has_zero_emission():
    network = tiny_network()
    candidates = generate_candidates(network, count=8, radius_m=.01, spacing_m=10)
    assert candidates.fallback.all()
    assert (~candidates.valid).any()
    probabilities = emission_probabilities(np.zeros(candidates.arc_ids.shape), candidates.valid)
    assert np.all(probabilities[~candidates.valid] == 0)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1)


def test_directed_partial_arc_routing():
    network = tiny_network()
    length = network.lengths[0]
    distances = network.transition_distances(np.array([0]), np.array([.8]),
                                              np.array([0, 1]), np.array([.9, .3]), 300)
    np.testing.assert_allclose(distances, [[.1 * length, .5 * length]], atol=.001)
    # A backwards move >3 m on the same edge must travel around a directed loop.
    backwards = network.transition_distances(np.array([0]), np.array([.8]), np.array([0]), np.array([.2]), 300)
    assert backwards[0, 0] > length
    assert network.bridge(0, 1, 300) == []


def test_viterbi_equals_bruteforce_and_resets_unreachable_sequences():
    emissions = np.array([[.7, .3], [.2, .8], [.6, .4]])
    transitions = [np.log(np.array([[.9, .1], [.1, .9]]))] * 2
    path, resets = viterbi(emissions, transitions)
    def score(sequence):
        return sum(np.log(emissions[t, s]) for t, s in enumerate(sequence)) + sum(
            transitions[t][sequence[t], sequence[t + 1]] for t in range(2))
    expected = max(product(range(2), repeat=3), key=score)
    np.testing.assert_array_equal(path, expected)
    assert resets.tolist() == [True, False, False]
    path, resets = viterbi(emissions, [None, np.full((2, 2), -np.inf)])
    assert resets.all()
    np.testing.assert_array_equal(path, emissions.argmax(axis=1))


def test_reference_alignment_monotonic_and_inference_never_opens_route(tmp_path, monkeypatch):
    network = tiny_network()
    alignment = align_reference(network, np.array([0, 1]))
    assert np.all(np.diff(alignment.route_positions) >= 0)
    np.testing.assert_array_equal(alignment.arc_ids, [0, 0, 1])
    folder = tmp_path / "00000001"
    folder.mkdir()
    for extension, values in [("track", network.trip.track), ("nodes", network.trip.nodes), ("arcs", network.trip.arcs)]:
        np.savetxt(folder / f"00000001.{extension}", values)
    (folder / "00000001.route").write_text("invalid route deliberately not read")
    original = np.loadtxt
    def guarded(path, *args, **kwargs):
        assert Path(path).suffix != ".route", "Prediction attempted reference access"
        return original(path, *args, **kwargs)
    monkeypatch.setattr(np, "loadtxt", guarded)
    loaded = load_trip(folder)
    assert not hasattr(loaded, "route")
    candidates = generate_candidates(RoadNetwork(loaded), count=5, radius_m=300)
    decoded = decode(network, candidates, np.ones(candidates.arc_ids.shape) * .5, DecoderConfig())
    assert len(decoded.arc_ids) == len(loaded.track)
    assert reconstruct_route(network, decoded, DecoderConfig())


def test_trip_group_folds_never_overlap():
    groups = np.repeat(np.arange(9), 10)
    y = np.tile([0, 1], 45)
    folds = grouped_folds(np.ones((90, 2)), y, groups, {"cv_folds": 3, "seed": 357})
    all_validation = []
    for train, validation in folds:
        assert not set(groups[train]) & set(groups[validation])
        all_validation.extend(validation)
    assert sorted(all_validation) == list(range(90))


def test_same_arc_backward_motion_preserves_required_network_loop():
    trip = Trip("00000001", np.array([[.0008, 0., 0.], [.0002, 0., 10.]]),
                np.array([[0., 0.], [.001, 0.]]), np.array([[0, 1], [1, 0]]))
    network = RoadNetwork(trip)
    decoded = Decoded(np.array([0, 0]), np.array([0, 0]), np.array([True, False]), {})
    assert reconstruct_route(network, decoded, DecoderConfig()) == [[0, 1, 0]]


def test_loader_rejects_fractional_arc_node_ids(tmp_path):
    folder = tmp_path / "00000001"
    folder.mkdir()
    (folder / "00000001.track").write_text("0 0 0\n0.001 0 1\n")
    (folder / "00000001.nodes").write_text("0 0\n0.001 0\n")
    (folder / "00000001.arcs").write_text("0 0.5\n")
    with pytest.raises(ValueError, match="Invalid node references"):
        load_trip(folder)
