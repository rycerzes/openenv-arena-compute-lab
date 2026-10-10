# SPDX-License-Identifier: BSD-3-Clause

"""FastAPI application for the Verified Compute Lab environment.

Run locally with `uvicorn server.app:app --host 0.0.0.0 --port 8000`.
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError("openenv is required; install the project dependencies first") from e

try:
    from ..models import ComputeLabAction, ComputeLabObservation
    from .compute_lab_environment import ComputeLabEnvironment
except ImportError:
    from models import ComputeLabAction, ComputeLabObservation
    from server.compute_lab_environment import ComputeLabEnvironment


app = create_app(
    ComputeLabEnvironment,
    ComputeLabAction,
    ComputeLabObservation,
    env_name="compute_lab",
    max_concurrent_envs=8,
)


def main(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(port=args.port)
