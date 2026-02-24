"""Retrieval pipeline package."""

# Import stage modules to trigger @register_stage decorators
from . import scope_stage as _scope_stage  # noqa: F401
from . import precision_stage as _precision_stage  # noqa: F401
from . import similarity_stage as _similarity_stage  # noqa: F401
