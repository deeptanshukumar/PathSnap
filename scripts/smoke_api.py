"""Exercise all active trips over real HTTP and retain small, honest evidence."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import httpx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--output", type=Path, default=Path("reports/evidence/api_smoke.json"))
    args = parser.parse_args()
    records = []
    with httpx.Client(base_url=args.url, timeout=180) as client:
        health = client.get("/health"); health.raise_for_status()
        trips = client.get("/trips"); trips.raise_for_status()
        partition = trips.json()
        assert len(partition["trips"]) == 10
        for item in partition["trips"]:
            start = perf_counter()
            response = client.post("/match", json=item)
            response.raise_for_status()
            result = response.json()
            assert len(result["predicted_arc_ids"]) == result["observations"]
            assert all(isinstance(x, int) and x >= 0 for x in result["predicted_arc_ids"])
            assert result["trip_id"] == item["trip_id"]
            assert "reference_arcs" not in result and "reference_route" not in result
            route_map = client.get(result["map_url"]); route_map.raise_for_status()
            assert "Viterbi reconstructed route" in route_map.text
            assert "Reference route (development only)" not in route_map.text
            records.append({"trip": item["trip_id"], "status_code": response.status_code,
                            "request_id": result["request_id"], "observations": result["observations"],
                            "http_seconds": perf_counter()-start, "server_seconds": result["seconds"],
                            "metrics": result["metrics"]})
            print(f"HTTP 200 | {item['trip_id']} | {result['observations']} points | request={result['request_id']} | {result['seconds']:.2f}s", flush=True)
        assert client.post("/match", json={"trip_id": "99999999"}).status_code == 404
        assert client.post("/match", json={"trip_id": "../bad"}).status_code == 422
        assert client.post("/match", json={"trip_id": records[0]["trip"], "reference_route": [1]}).status_code == 422
        preflight = client.options("/match", headers={"Origin": "http://localhost:8010", "Access-Control-Request-Method": "POST", "Access-Control-Request-Headers": "content-type"})
        assert preflight.status_code == 200
        assert preflight.headers["access-control-allow-origin"] == "*"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"url": args.url, "health": health.json(), "mode": partition["mode"],
                                      "requests": records, "negative_request_checks": "passed", "cors_preflight": "passed",
                                      "evidence_boundary": "Real HTTP requests from this host. This does not establish a physical two-laptop test."}, indent=2) + "\n")


if __name__ == "__main__":
    main()
