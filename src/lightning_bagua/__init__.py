import os

from lightning_bagua.__about__ import *  # noqa: F401, F403
from lightning_bagua.environment import BaguaEnvironment
from lightning_bagua.strategy import BaguaStrategy

_PACKAGE_ROOT = os.path.dirname(__file__)
_PROJECT_ROOT = os.path.dirname(_PACKAGE_ROOT)

__all__ = ["BaguaEnvironment", "BaguaStrategy"]
