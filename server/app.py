"""ASGI app entrypoint expected by OpenEnv packaging checks."""

from __future__ import annotations

import os

import uvicorn

from server.main import app

__all__ = ["app"]


def main() -> None:
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "7860"))
    uvicorn.run("server.main:app", host=host, port=port, factory=False)


if __name__ == "__main__":
    main()
