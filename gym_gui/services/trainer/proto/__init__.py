"""Generated gRPC protocol bindings for the trainer daemon."""

from __future__ import annotations

import sys

# gRPC tools emit bare absolute imports (e.g. ``import trainer_pb2``).
# Register sys.modules aliases so those imports resolve correctly when this
# package lives under ``gym_gui.services.trainer.proto``.

from . import trainer_pb2 as _trainer_pb2
sys.modules.setdefault("trainer_pb2", _trainer_pb2)
from . import trainer_pb2_grpc as _trainer_pb2_grpc
sys.modules.setdefault("trainer_pb2_grpc", _trainer_pb2_grpc)

from . import inference_pb2 as _inference_pb2
sys.modules.setdefault("inference_pb2", _inference_pb2)
from . import inference_pb2_grpc as _inference_pb2_grpc
sys.modules.setdefault("inference_pb2_grpc", _inference_pb2_grpc)

from . import system_pb2 as _system_pb2
sys.modules.setdefault("system_pb2", _system_pb2)
from . import system_pb2_grpc as _system_pb2_grpc
sys.modules.setdefault("system_pb2_grpc", _system_pb2_grpc)

trainer_pb2      = _trainer_pb2
trainer_pb2_grpc = _trainer_pb2_grpc
inference_pb2      = _inference_pb2
inference_pb2_grpc = _inference_pb2_grpc
system_pb2      = _system_pb2
system_pb2_grpc = _system_pb2_grpc

__all__ = [
    "trainer_pb2", "trainer_pb2_grpc",
    "inference_pb2", "inference_pb2_grpc",
    "system_pb2", "system_pb2_grpc",
]
