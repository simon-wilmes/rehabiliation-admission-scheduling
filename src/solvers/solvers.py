from abc import ABC, abstractmethod
from src.solution import Solution
from src.instance import Instance


class Solver(ABC):
    def __init__(self, instance: Instance):
        self.instance = instance

    @abstractmethod
    def solve_model(self) -> Solution:
        pass

    @abstractmethod
    def create_model(self) -> None:
        pass
