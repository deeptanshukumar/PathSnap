"""FastAPI boundary: prediction first, reference-only server scoring second."""

from collections import OrderedDict
from dataclasses import asdict
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from threading import Lock
from time import perf_counter
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from .data import trip_directories
from .evaluation import score_prediction
from .maps import make_map
from .pipeline import Pipeline
from .trace import build_trace, explain_point

logger = logging.getLogger("uvicorn.error")


class MatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    trip_id: str = Field(pattern=r"^\d{8}$")


def create_app(data_root: Path | None, artifacts: Path, mode: str = "development",
               evaluation_root: Path | None = None) -> FastAPI:
    if mode == "evaluation":
        if evaluation_root is None:
            raise ValueError("Evaluation mode requires --evaluation-root with ten reserved trips")
        root = evaluation_root
    elif mode == "development" and data_root is not None:
        root = data_root / "student_test"
    else:
        raise ValueError("Development mode requires --data-root")
    folders = trip_directories(root)
    if len(folders) != 10:
        raise ValueError(f"{mode} mode requires exactly ten trip directories, found {len(folders)}")
    allowed = {folder.name: folder for folder in folders}
    pipeline = Pipeline(artifacts / "models/final.joblib")
    app = FastAPI(title="PathSnap", version="0.1.0")
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=False,
                       allow_methods=["GET", "POST"], allow_headers=["Content-Type"])
    maps: OrderedDict[str, str] = OrderedDict()
    # Read-only UI explanations of finished predictions; bounded like maps.
    traces: OrderedDict[str, dict] = OrderedDict()
    # The trace reports the decoder settings actually used by this pipeline instance.
    trace_config = {**pipeline.config, "decoder": asdict(pipeline.decoder_config)}
    predictions: OrderedDict[str, object] = OrderedDict()
    inference_lock = Lock()
    ui_root = Path(__file__).resolve().parent.parent / "ui"
    if ui_root.exists():
        app.mount("/static", StaticFiles(directory=ui_root), name="static")

    @app.get("/", response_class=FileResponse)
    def home():
        return FileResponse(ui_root / "index.html")

    @app.get("/health")
    def health():
        return {"status": "ready", "mode": mode, "trip_count": len(allowed),
                "model": pipeline.bundle["model_name"], "tuned": pipeline.bundle["is_tuned"]}

    @app.get("/trips")
    def trips():
        return {"mode": mode, "trips": [{"trip_id": trip} for trip in allowed],
                "notice": "Student-test development trips" if mode == "development" else "Reserved final-evaluation trips"}

    @app.post("/match")
    def match(request: MatchRequest):
        if request.trip_id not in allowed:
            raise HTTPException(status_code=404, detail="Trip is not in the active ten-trip partition")
        request_id = uuid4().hex[:12]
        start = perf_counter()
        with inference_lock:
            try:
                prediction = pipeline.predict(allowed[request.trip_id])
                # Reference data is opened only after all predictions are final.
                try:
                    metrics = score_prediction(prediction, allowed[request.trip_id])
                except FileNotFoundError:
                    metrics = None
                map_html = make_map(prediction).get_root().render()
            except (ValueError, OSError) as error:
                logger.exception("PathSnap request %s failed", request_id)
                raise HTTPException(status_code=422, detail=str(error)) from error
            maps[request_id] = map_html
            if len(maps) > 32:
                maps.popitem(last=False)
            traces[request_id] = build_trace(prediction, trace_config, pipeline.bundle["model_name"],
                                             pipeline.bundle["is_tuned"])
            predictions[request_id] = prediction
            for store, limit in ((traces, 32), (predictions, 8)):
                while len(store) > limit:
                    store.popitem(last=False)
        event = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "request_id": request_id,
                 "trip_id": request.trip_id, "mode": mode, "observations": len(prediction.network.gps),
                 "status": "success", "seconds": round(perf_counter() - start, 4)}
        logger.info("PATHSNAP %s", json.dumps(event))
        return {**event, "model": pipeline.bundle["model_name"], "tuned": pipeline.bundle["is_tuned"],
                "predicted_arc_ids": prediction.decoded.arc_ids.tolist(),
                "route_pieces": prediction.route_pieces,
                "breaks": [{"observation": key, "reason": value} for key, value in prediction.decoded.break_reasons.items()],
                "candidate_fallback_observations": int(prediction.candidates.fallback.sum()),
                "metrics": metrics, "timings": prediction.timings, "map_url": f"/maps/{request_id}",
                "trace_url": f"/trace/{request_id}"}

    @app.get("/maps/{request_id}", response_class=HTMLResponse)
    def route_map(request_id: str):
        if request_id not in maps:
            raise HTTPException(status_code=404, detail="Map expired; run matching again")
        return maps[request_id]

    @app.get("/trace/{request_id}")
    def trace(request_id: str):
        if request_id not in traces:
            raise HTTPException(status_code=404, detail="Trace expired; run matching again")
        return traces[request_id]

    @app.get("/explain/{request_id}/{observation}")
    def explain(request_id: str, observation: int):
        if request_id not in predictions:
            raise HTTPException(status_code=404, detail="Explanation expired; run matching again")
        try:
            with inference_lock:
                return explain_point(predictions[request_id], trace_config, observation)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    return app
