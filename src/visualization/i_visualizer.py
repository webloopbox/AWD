from abc import ABC, abstractmethod

class IVisualizer(ABC):
    @abstractmethod
    def plot(self, *args, **kwargs):
        pass
