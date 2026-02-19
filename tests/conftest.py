"""Shared test configuration.

Adds control script directories to sys.path so tests can import
functions directly from the validation scripts.
"""

import sys
from pathlib import Path

CONTROL = Path(__file__).parent.parent / "control"

_CONTROL_SUBDIRS = [
    "surrogate",
    "sequence",
    "transformer",
    "transfer",
    "pinn",
    "deeponet",
    "lenet",
    "lstm_weather",
    "quantized",
    "isomorphic",
]

for subdir in _CONTROL_SUBDIRS:
    path = str(CONTROL / subdir)
    if path not in sys.path:
        sys.path.insert(0, path)
