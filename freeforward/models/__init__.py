# Copyright (c) Xiaomi Corporation.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

__all__ = ["ARCHITECTURES", "FREEFORWARD_MODELS"]


def __getattr__(name):
    # Geometry/motion utilities can be imported on CPU without importing the
    # CUDA rasterizer. The reconstruction registry is loaded when requested.
    if name in __all__:
        from .archs import ARCHITECTURES
        globals()["ARCHITECTURES"] = ARCHITECTURES
        globals()["FREEFORWARD_MODELS"] = ARCHITECTURES["small"]
        return globals()[name]
    raise AttributeError(name)
