# Copyright (c) Xiaomi Corporation.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Architecture registry. Each architecture exposes a ``FREEFORWARD_MODELS`` dict mapping
a backbone-size string (e.g. ``"FreeForward-B/8"``) to a factory callable. To add a new
architecture, drop a ``<name>.py`` here and register it in ``ARCHITECTURES``.
"""

from . import small

ARCHITECTURES = {
    "small": small.FREEFORWARD_MODELS,
}
