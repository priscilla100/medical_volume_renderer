"""
STOP. You should not modify this file unless you KNOW what you are doing.
"""

from abc import ABC, abstractmethod


class Renderable(ABC):
    """
    Abstract class (interface) representing an object-to-render.
    All shapes should inherit this class.
    """
    @abstractmethod
    def render(self, timeElapsedSinceLastFrame: int) -> None:
        pass
