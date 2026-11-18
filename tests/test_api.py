import json

from fastapi.testclient import TestClient
import joblib
import numpy as np
import pytest
from sklearn.dummy import DummyClassifier

from pathsnap.api import create_app
from pathsnap.features import FEATURE_NAMES


@pytest.fixture
def api_setup(tmp_path):
    root = tmp_path / "data"
    for i in range(10):
        trip_id = f"{i:08d}"
        folder = root / "student_test" / trip_id
        folder.mkdir(parents=True)
        for ext, text in {
            "track": "0.0001 0.00001 0\n0.0008 0.00001 1\n0.0015 0.00001 2\n",
            "nodes": "0 0\n0.001 0\n0.002 0\n",
            "arcs": "0 1\n1 2\n1 0\n2 1\n", "route": "0\n1\n",
        }.items():
            (folder / f"{trip_id}.{ext}").write_text(text)
    artifacts = tmp_path / "artifacts"
    (artifacts / "models").mkdir(parents=True)
    model = DummyClassifier(strategy="prior").fit(np.zeros((2, len(FEATURE_NAMES))), [0, 1])
    joblib.dump({"model": model, "model_name": "synthetic-test-only", "is_tuned": False,
                 "feature_names": FEATURE_NAMES, "config": {"candidate_count": 4, "candidate_radius_m": 300,
                 "index_spacing_m": 30, "decoder": {}}}, artifacts / "models/final.joblib")
    return root, artifacts


def test_api_validation_cors_and_no_reference_exposure(api_setup):
    root, artifacts = api_setup
    with TestClient(create_app(root, artifacts)) as client:
        assert client.get("/health").json()["mode"] == "development"
        assert len(client.get("/trips").json()["trips"]) == 10
        result = client.post("/match", json={"trip_id": "00000000"})
        assert result.status_code == 200
        body = result.json()
        assert len(body["predicted_arc_ids"]) == 3
        for key in ["reference_arcs", "reference_route", "reference_geometry", "reference_labels"]:
            assert key not in json.dumps(body)
        html = client.get(body["map_url"]).text
        assert "Reference route (development only)" not in html
        assert client.post("/match", json={"trip_id": "99999999"}).status_code == 404
        assert client.post("/match", json={"trip_id": "../test"}).status_code == 422
        assert client.post("/match", json={"trip_id": "00000000", "labels": []}).status_code == 422
        response = client.options("/match", headers={"Origin": "http://localhost:8010", "Access-Control-Request-Method": "POST"})
        assert response.status_code == 200
        assert response.headers["access-control-allow-origin"] == "*"


def test_reference_changes_cannot_change_predictions(api_setup):
    root, artifacts = api_setup
    with TestClient(create_app(root, artifacts)) as client:
        first = client.post("/match", json={"trip_id": "00000000"}).json()
        (root / "student_test/00000000/00000000.route").write_text("2\n3\n")
        second = client.post("/match", json={"trip_id": "00000000"}).json()
        assert first["predicted_arc_ids"] == second["predicted_arc_ids"]
        assert first["route_pieces"] == second["route_pieces"]
        assert first["metrics"] != second["metrics"]


def test_evaluator_mode_never_substitutes_student_test(api_setup):
    root, artifacts = api_setup
    with pytest.raises(ValueError, match="requires --evaluation-root"):
        create_app(root, artifacts, "evaluation")
    with pytest.raises(FileNotFoundError):
        create_app(root, artifacts, "evaluation", root / "missing_evaluator_folder")
