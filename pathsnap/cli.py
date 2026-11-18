"""Portable entry points; dataset locations are arguments, never source constants."""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="PathSnap experiment and demonstration tools")
    parser.add_argument("command", choices=["prepare", "train", "evaluate", "report", "serve"])
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--cache", type=Path, default=Path("cache"))
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--config", type=Path, default=Path("configs/baseline.json"))
    parser.add_argument("--reports", type=Path, default=Path("reports"))
    parser.add_argument("--mode", choices=["development", "evaluation"], default="development")
    parser.add_argument("--evaluation-root", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--variant", choices=["default", "improved"], default="default",
                        help="serve the reported model (default) or the opt-in improved variant")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.command in ["prepare", "train", "evaluate", "report"] and args.data_root is None:
        parser.error("--data-root is required")
    if args.command == "prepare":
        from .prepare import prepare
        prepare(args.data_root, args.cache, args.artifacts, config)
    elif args.command == "train":
        from .experiments import run_experiments
        run_experiments(args.data_root, args.cache, args.artifacts, config)
    elif args.command == "evaluate":
        from .evaluation import evaluate_decoder
        evaluate_decoder(args.data_root, args.artifacts, args.reports)
    elif args.command == "report":
        from .reporting import generate_report
        generate_report(args.data_root, args.artifacts, args.reports)
    elif args.command == "serve":
        import uvicorn
        from .api import create_app
        app = create_app(args.data_root, args.artifacts, args.mode, args.evaluation_root, args.variant)
        uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
