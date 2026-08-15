"""Hierarchical Channel-Routed Memory (HCRM) small language model."""

from hcrm.config import HCRMConfig
from hcrm.model import HCRM
from hcrm.table import RuntimeTable

__all__ = ["HCRM", "HCRMConfig", "RuntimeTable"]
__version__ = "0.1.0"
