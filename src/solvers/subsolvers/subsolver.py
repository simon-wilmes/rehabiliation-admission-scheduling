#
from abc import ABC, abstractmethod
from src.patients import Patient
from src.treatments import Treatment
from src.solvers import Solver
from src.instance import Instance
from src.logging import logger
from src.solution import Solution, Appointment
from src.utils import slice_dict
from time import time
from collections import defaultdict


class Subsolver(ABC):
    FEASIBLE = 0
    TOO_MANY_TREATMENTS = 1
    MIN_PATIENTS_PROBLEM = 2
    BASE_SOLVER_OPTIONS = {
        "use_day_symmetry": [True, False],
    }
    BASE_SOLVER_DEFAULT_OPTIONS = {
        "use_day_symmetry": True,
    }

    def __init__(self, instance: Instance, solver: Solver, **kwargs):
        self.instance: Instance = instance  # type: ignore
        self.solver: Solver = solver  # type: ignore

        logger.debug(f"Setting Subsolver options: SubSolver")
        for key in Subsolver.BASE_SOLVER_DEFAULT_OPTIONS:
            if key in kwargs:
                setattr(self, key, kwargs[key])
                logger.debug(f" ---- {key} to {kwargs[key]}")
                del kwargs[key]
            else:
                setattr(self, key, self.__class__.BASE_SOLVER_DEFAULT_OPTIONS[key])
                logger.debug(
                    f" ---- {key} to { self.__class__.BASE_SOLVER_DEFAULT_OPTIONS[key]} (default)"
                )
        assert len(kwargs) == 0, f"Unknown options: {kwargs}"

        self.previous_results = dict()

        if self.use_day_symmetry:  # type:ignore
            self.get_day_symmetry = self._get_day_symmetry
            self.days_symmetry = {}
            for d in self.solver.D:
                for d2 in set(self.days_symmetry.values()):
                    if all(
                        [
                            self.solver.av_fdt[key]
                            == self.solver.av_fdt[key[0], d2, key[2]]
                            for key in slice_dict(self.solver.av_fdt, (None, d, None))
                        ]
                    ):
                        # d2 and d are symmetric
                        self.days_symmetry[d] = d2
                        break
                else:
                    self.days_symmetry[d] = d

            days_symmetric_to = defaultdict(set)
            for d in self.solver.D:
                days_symmetric_to[self.days_symmetry[d]].add(d)

            self.days_symmetric_to = {}
            for key, value in days_symmetric_to.items():
                self.days_symmetric_to[key] = sorted(list(value))

        else:
            self.get_day_symmetry = lambda x: x

        self.calls_to_is_day_infeasible = 0
        self.returns_feasible = 0
        self.returns_infeasible = 0
        self.calls_to_solve_subsystem = 0

    @abstractmethod
    def _solve_subsystem(
        self,
        day: int,
        patients: dict[Treatment, dict[Patient, int]],
        max_appointments: dict[Treatment, int] | None = None,
    ) -> dict:
        pass

    def _get_hash(
        self,
        day: int,
        patients: dict[Treatment, dict[Patient, int]],
    ):
        s = str(day) + "|"
        for m in self.solver.M:
            s += (
                str(m.id)
                + ":"
                + str(sorted([patients[m][p] for p in patients[m]]))
                + ","
            )
        return s

    def _get_result(self, day: int, patients: dict[Treatment, dict[Patient, int]]):
        input_value = self._get_hash(self.get_day_symmetry(day), patients)
        if input_value not in self.previous_results:
            return None
        return self.previous_results[input_value]

    def _add_result(
        self, day: int, patients: dict[Treatment, dict[Patient, int]], result: dict
    ):
        input_value = self._get_hash(self.get_day_symmetry(day), patients)
        self.previous_results[input_value] = result

    def _get_day_symmetry(self, d):
        # In the future, this method could be used to calculate a hash for the resource profile of that day
        return self.days_symmetry[d]

    def get_days_symmetric_to(self, d: int) -> set[int]:
        return self.days_symmetric_to[self._get_day_symmetry(d)]

    def is_day_infeasible(
        self, day: int, patients: dict[Treatment, dict[Patient, int]]
    ) -> dict:
        self.calls_to_is_day_infeasible += 1

        # Check if this day has already been seen with the same configuration and report the result

        prev_results = self._get_result(day, patients)
        if prev_results is not None:
            if prev_results.get("status_code", None) == Subsolver.FEASIBLE:
                self.returns_feasible += 1
            else:
                self.returns_infeasible += 1
            return prev_results

        self.calls_to_solve_subsystem += 1
        logger.debug("---------- SOLVE SUBSYSTEM ---------")
        result = self._solve_subsystem(day, patients)
        logger.debug("---------- END SUBSYSTEM ---------")

        # Store the result for future reference
        self._add_result(day, patients, result)
        if result.get("status_code", None) == Subsolver.FEASIBLE:
            self.returns_feasible += 1
        else:
            self.returns_infeasible += 1
        return result

    @abstractmethod
    def _get_day_solution(
        self, day: int, patients: dict[Treatment, dict[Patient, int]]
    ) -> list[Appointment]:
        pass

    def get_day_solution(
        self, day: int, patients: dict[Treatment, dict[Patient, int]]
    ) -> list[Appointment]:

        return self._get_day_solution(day, patients)
