"""Folium maps: reference geometry is opt-in for development reports only."""

import folium
import numpy as np

from .pipeline import Prediction


def make_map(prediction: Prediction, reference_route: np.ndarray | None = None) -> folium.Map:
    network = prediction.network
    trip = network.trip
    center = np.median(trip.track[:, :2], axis=0)[::-1].tolist()
    result = folium.Map(location=center, tiles="OpenStreetMap", control_scale=True, prefer_canvas=True)

    def edge_layer(arc_ids, name, color, weight, opacity, show=True):
        edges = trip.nodes[trip.arcs[np.asarray(arc_ids, dtype=int)]][:, :, ::-1].tolist()
        group = folium.FeatureGroup(name=name, show=show)
        folium.PolyLine(edges, color=color, weight=weight, opacity=opacity, tooltip=name).add_to(group)
        group.add_to(result)

    nearby = np.unique(prediction.candidates.arc_ids[prediction.candidates.valid])
    edge_layer(nearby, "Candidate road network", "#86908c", 2, 0.35)
    if reference_route is not None:
        edge_layer(reference_route, "Reference route (development only)", "#e0a42f", 8, 0.65)
    edge_layer(np.unique(prediction.classifier_arcs), "Classifier only", "#ad55b8", 3, 0.8, False)
    # Draw actual road edges as separate polylines; never connect disconnected pieces by straight lines.
    edge_layer([arc for piece in prediction.route_pieces for arc in piece], "Viterbi reconstructed route", "#087f6a", 5, 0.9)
    gps = folium.FeatureGroup(name="Original GPS observations")
    stride = max(1, len(trip.track) // 1200)
    for i in range(0, len(trip.track), stride):
        point = trip.track[i]
        folium.CircleMarker(location=[float(point[1]), float(point[0])], radius=2, color="#327fca", weight=1,
                            fill=True, fill_opacity=0.8, tooltip=f"GPS {i} | {point[2]:g} s").add_to(gps)
    gps.add_to(result)
    for index in [0, len(trip.track) - 1]:
        point = trip.track[index]
        folium.Marker([float(point[1]), float(point[0])], tooltip="Start" if index == 0 else "Finish",
                      icon=folium.Icon(color="green" if index == 0 else "red", icon="flag")).add_to(result)
    breaks = folium.FeatureGroup(name="Decoder resets")
    for i, reason in prediction.decoded.break_reasons.items():
        point = trip.track[i]
        folium.CircleMarker([float(point[1]), float(point[0])], radius=7, color="#d15243", tooltip=reason).add_to(breaks)
    breaks.add_to(result)
    bounds = trip.track[:, :2]
    result.fit_bounds([bounds.min(axis=0)[::-1].tolist(), bounds.max(axis=0)[::-1].tolist()], padding=(25, 25))
    folium.LayerControl(collapsed=False).add_to(result)
    ref_key = '<span style="color:#c58a12">━</span> Reference (development only)<br>' if reference_route is not None else ""
    result.get_root().html.add_child(folium.Element(
        '<div style="position:fixed;bottom:25px;left:15px;z-index:9999;background:white;padding:12px 16px;'
        'border:1px solid #ddd;border-radius:8px;font:12px sans-serif;line-height:1.8">'
        '<strong>PathSnap layers</strong><br><span style="color:#327fca">●</span> GPS observations<br>'
        '<span style="color:#087f6a">━</span> Predicted road route<br>' + ref_key +
        '<span style="color:#d15243">○</span> Explicit reset / gap</div>'
    ))
    return result
