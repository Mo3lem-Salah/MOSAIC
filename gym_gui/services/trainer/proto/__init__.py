"""Generated gRPC protocol bindings for the trainer daemon."""

from __future__ import annotations

import importlib
import sys

# gRPC tools emit bare absolute imports (e.g. ``import trainer_pb2``).
# Register sys.modules aliases so those imports resolve correctly when this
# package lives under ``gym_gui.services.trainer.proto``.
from . import trainer_pb2 as _trainer_pb2

sys.modules.setdefault("trainer_pb2", _trainer_pb2)
from . import trainer_pb2_grpc as _trainer_pb2_grpc

sys.modules.setdefault("trainer_pb2_grpc", _trainer_pb2_grpc)

trainer_pb2 = _trainer_pb2
trainer_pb2_grpc = _trainer_pb2_grpc

__all__ = [
    "trainer_pb2", "trainer_pb2_grpc",
]

_OPTIONAL_BINDINGS = frozenset(
    {
        "inference_pb2",
        "inference_pb2_grpc",
        "system_pb2",
        "system_pb2_grpc",
    }
)


def __getattr__(name: str):
    """Load optional protocol bindings only when their service is used."""
    if name not in _OPTIONAL_BINDINGS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    if name.endswith("_grpc"):
        # Generated gRPC modules use an absolute import for their paired pb2
        # module. Load and alias it before importing the generated service code.
        __getattr__(name.removesuffix("_grpc"))

    module = importlib.import_module(f".{name}", __name__)
    sys.modules.setdefault(name, module)
    globals()[name] = module
    return module
