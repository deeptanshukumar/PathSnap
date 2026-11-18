import numpy as np

from pathsnap.data import Trip
from pathsnap.features import generate_candidates
from pathsnap.geometry import RoadNetwork
from pathsnap.improved import CONTEXT_NAMES, IMPROVED_FEATURE_NAMES, context_features, improved_features, weighted_scores


def straight_road_with_outage() -> RoadNetwork:
    nodes = np.array([[0., 0.], [.001, 0.], [.002, 0.], [.001, .001]])
    arcs = np.array([[0, 1], [1, 2], [1, 0], [2, 1], [1, 3]])
    xs = np.linspace(.0001, .0019, 8)
    times = np.array([0., 1., 2., 3., 200., 201., 202., 203.])  # outage between index 3 and 4
    track = np.column_stack([xs, np.full(8, .00001), times])
    return RoadNetwork(Trip("00000001", track, nodes, arcs))


def test_context_features_shape_finite_and_zero_padding():
    network = straight_road_with_outage()
    candidates = generate_candidates(network, count=5, radius_m=300, spacing_m=10)
    features = improved_features(network, candidates)
    assert features.shape == (8, 5, len(IMPROVED_FEATURE_NAMES))
    assert np.isfinite(features).all()
    assert np.all(features[~candidates.valid] == 0)


def test_forward_arc_beats_reverse_twin_on_context_heading():
    network = straight_road_with_outage()
    candidates = generate_candidates(network, count=5, radius_m=300, spacing_m=10)
    context = context_features(network, candidates)
    heading = context[..., CONTEXT_NAMES.index("ctx_heading_cos_pm5")]
    i = 1  # travelling +x along arc 0 (0 -> 1); arc 2 is the same road reversed
    forward = heading[i, list(candidates.arc_ids[i]).index(0)]
    reverse = heading[i, list(candidates.arc_ids[i]).index(2)]
    assert forward > 0.9 and reverse < -0.9


def test_context_window_never_crosses_an_outage():
    network = straight_road_with_outage()
    candidates = generate_candidates(network, count=5, radius_m=300, spacing_m=10)
    before = context_features(network, candidates)
    shifted = network.gps.copy()
    shifted[4:] += 500.0  # move everything after the outage far away
    network.gps = shifted
    after = context_features(network, candidates)
    np.testing.assert_allclose(before[:4, :, 0], after[:4, :, 0])


def test_emission_power_keeps_probabilities_valid():
    scores = np.array([[0.2, 0.8, 0.0]])
    weighted = weighted_scores(scores, 2.0)
    assert np.all((weighted >= 0) & (weighted <= 1))
    np.testing.assert_allclose(weighted[0, :2], [0.04, 0.64])
