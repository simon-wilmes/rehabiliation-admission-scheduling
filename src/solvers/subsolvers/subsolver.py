#
from abc import ABC, abstractmethod
from src.patients import Patient
from src.treatments import Treatment
from src.solvers import Solver
from src.instance import Instance
from src.logging import logger


class Subsolver(ABC):
    FEASIBLE = 0
    COMPLETELY_INFEASIBLE = 1
    MORE_TREATMENTS_NEEDED = 2
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
                self.results = {}

            elif self.store_results_method == "hash":  # type:ignore
                self.get_result = self._get_result_hash
                self.add_result = self._add_result_hash
                self.results = set()
            else:
                raise ValueError("Invalid store_results_method")

    @abstractmethod
    def _solve_subsystem(
        self,
        day: int,
        appointments: dict[Treatment, int],
        patients: dict[Treatment, dict[Patient, int]],
        max_appointments: dict[Treatment, int] | None = None,
    ) -> dict:
        pass

    def _get_result_dict(
        self,
        day: int,
        appointments: dict[Treatment, int],
        patients: dict[Treatment, dict[Patient, int]],
    ):
        return False

    def _get_result_hash(
        self,
        day: int,
        appointments: dict[Treatment, int],
        patients: dict[Treatment, dict[Patient, int]],
    ):
        input_value = frozenset(
            (
                day,
                frozenset(
                    (m, p, patients[m][p]) for m in patients for p in patients[m]
                ),
                frozenset((m, appointments[m]) for m in appointments),
            )
        )
        return input_value in self.results

    def _add_result_dict(
        self,
        day: int,
        appointments: dict[Treatment, int],
        patients: dict[Treatment, dict[Patient, int]],
    ):
        pass

    def get_day_hash(self, d):
        # In the future, this method could be used to calculate a hash for the resource profile of that day
        return d

    def _add_result_hash(
        self,
        day: int,
        appointments: dict[Treatment, int],
        patients: dict[Treatment, dict[Patient, int]],
    ):
        self.results.add(  # type:ignore
            frozenset(
                (
                    day,
                    frozenset(
                        (m, p, patients[m][p]) for m in patients for p in patients[m]
                    ),
                    frozenset((m, appointments[m]) for m in appointments),
                )
            )
        )

    def is_day_infeasible(
        self,
        day: int,
        appointments: dict[Treatment, int],
        patients: dict[Treatment, dict[Patient, int]],
        max_appointments: dict[Treatment, int] | None = None,
    ) -> dict:

        # Check if this day has already been seen with the same configuration and report the result
        if self.store_results:  # type:ignore
            if self.get_result(day, appointments, patients):
                return {"status_code": Subsolver.FEASIBLE}

        results = self._solve_subsystem(day, appointments, patients, max_appointments)
        logger.debug(
            "Subsolver results: "
            + (
                "FEASIBLE"
                if results["status_code"] == Subsolver.FEASIBLE
                else "INFEASIBLE"
            )
        )

        if self.store_results and not results:  # type:ignore
            self.add_result(day, appointments, patients)

        return results
