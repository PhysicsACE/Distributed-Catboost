from distributed_catboost.main import predict, train
from distributed_catboost.matrix import (
    RayPool,
    RayShardingMode,
)

__version__ = "1.0"

__all__ = [
    "__version__",
    "RayPool",
    "RayShardingMode",
    "train",
    "predict",
]