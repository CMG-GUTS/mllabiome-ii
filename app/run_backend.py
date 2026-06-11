from __future__ import annotations

import argparse
import sys
from pathlib import Path


repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from app.backend.main import app


def main() -> None:
    import uvicorn

    parser = argparse.ArgumentParser(description="Run mllabiome-ii inference backend")
    parser.add_argument(
        "--port", "-p", type=int, default=8000, help="Port to run the server on"
    )
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    args = parser.parse_args()

    print(f"Starting mllabiome-ii backend on http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
