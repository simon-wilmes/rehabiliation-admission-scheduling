#
from abc import ABC, abstractmethod
from src.patients import Patient
from src.treatments import Treatment
from src.solvers import Solver
from src.instance import Instance
from src.logging import logger
from src.solution import Solution, Appointment


class Subsolver(ABC):
    FEASIBLE = 0
    COMPLETELY_INFEASIBLE = 1
    DIFFERENT_TREATMENTS_NEEDED = 2
    BASE_SOLVER_OPTIONS = {
        "store_results": [True, False],
        "store_results_method": ["dict", "hash"],
        "enforce_min_patients_per_treatment": [True, False],
    }
    BASE_SOLVER_DEFAULT_OPTIONS = {
        "store_results": False,
        "store_results_method": "dict",
        "enforce_min_patients_per_treatment": True,
    }

    def __init__(self, instance: Instance, solver: Solver, **kwargs):
        self.instance: Instance = instance  # type: ignore
        self.solver: Solver = solver  # type: ignore

        logger.debug(f"Setting Subsolver options: SubSolver")
        for key in Subsolver.BASE_SOLVER_DEFAULT_OPTIONS:
            if key in kwargs:
                setattr(self, key, kwargs[key])
                logger.debug(f" ---- {key} to {kwargs[key]}")
            else:
                setattr(self, key, self.__class__.BASE_SOLVER_DEFAULT_OPTIONS[key])
                logger.debug(
                    f" ---- {key} to { self.__class__.BASE_SOLVER_DEFAULT_OPTIONS[key]} (default)"
                )

        if self.store_results:  # type:ignore
            if self.store_results_method == "dict":  # type:ignore
                self.get_result = self._get_result_dict
                self.add_result = self._add_result_hash  # type:ignore
                self.feasible_results = {}

            elif self.store_results_method == "hash":  # type:ignore
                self.get_result = self._get_result_hash
                self.add_result = self._add_result_hash
                self.feasible_results = set()
            else:
                raise ValueError("Invalid store_results_method")

    @abstractmethod
    def _solve_subsystem(
        self,
        day: int,
        patients: dict[Treatment, dict[Patient, int]],
        max_appointments: dict[Treatment, int] | None = None,
    ) -> dict:
        pass

    def _get_result_dict(
        self,
        day: int,
        patients: dict[Treatment, dict[Patient, int]],
    ):
        return False

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

    def _get_result_hash(
        self,
        day: int,
        patients: dict[Treatment, dict[Patient, int]],
    ):
        input_value = self._get_hash(day, patients)
        return input_value in self.feasible_results

    def get_day_hash(self, d):
        # In the future, this method could be used to calculate a hash for the resource profile of that day
        return d

    def _add_result_hash(
        self,
        day: int,
        patients: dict[Treatment, dict[Patient, int]],
    ):
        self.feasible_results.add(self._get_hash(day, patients))  # type:ignore

    def is_day_infeasible(
        self, day: int, patients: dict[Treatment, dict[Patient, int]]
    ) -> dict:

        # Check if this day has already been seen with the same configuration and report the result
        if self.store_results:  # type:ignore
            if self.get_result(day, patients):
                logger.debug("Used stored results")
                return {"status_code": Subsolver.FEASIBLE}

        results = self._solve_subsystem(day, patients)
        logger.debug(
            "Subsolver results: "
            + (
                "FEASIBLE"
                if results["status_code"] == Subsolver.FEASIBLE
                else "INFEASIBLE"
            )
        )

        if (
            self.store_results  # type:ignore
            and results["status_code"] == Subsolver.FEASIBLE
        ):

            self.add_result(day, patients)
            logger.debug(f"Adding result to store: {len(self.feasible_results)}")

        return results

    @abstractmethod
    def _get_day_solution(
        self, day: int, patients: dict[Treatment, dict[Patient, int]]
    ) -> list[Appointment]:
        pass

    def get_day_solution(
        self, day: int, patients: dict[Treatment, dict[Patient, int]]
    ) -> list[Appointment]:

        return self._get_day_solution(day, patients)
