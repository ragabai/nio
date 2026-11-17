"""NIO, the Nubian Ibex Optimizer."""

from nio.diagnostics import Diagnostics
from nio.optimizer import NIO
from nio.result import Result
from nio.settings import Settings
from nio.space import SearchSpace

__version__ = "1.0.0"

__all__ = ["NIO", "Diagnostics", "Result", "SearchSpace", "Settings", "__version__"]
