#
from abc import ABC, abstractmethod
from src.patients import Patient
from src.treatments import Treatment
from src.solvers import Solver
from src.instance import Instance
from src.logging import logger
from src.solution import Solution, Appointment
from src.utils import slice_dict


class Subsolver(ABC):
    FEASIBLE = 0
    COMPLETELY_INFEASIBLE = 1
    DIFFERENT_TREATMENTS_NEEDED = 2
    BASE_SOLVER_OPTIONS = {
        "store_results": [True, False],
        "store_results_method": ["dict", "hash"],
        "enforce_min_patients_per_treatment": [True, False],
        "use_day_symmetry": [True, False],
    }
    BASE_SOLVER_DEFAULT_OPTIONS = {
        "store_results": False,
        "store_results_method": "dict",
        "enforce_min_patients_per_treatment": True,
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
            else:
                setattr(self, key, self.__class__.BASE_SOLVER_DEFAULT_OPTIONS[key])
                logger.debug(
                    f" ---- {key} to { self.__class__.BASE_SOLVER_DEFAULT_OPTIONS[key]} (default)"
                )

        if self.store_results:  # type:ignore
            self.get_result_feasible = self._get_result_feasible_hash
            self.add_result_feasible = self._add_result_feasible_hash
            self.feasible_results = set()

            self.get_result_infeasible = self._get_result_infeasible_hash
            self.add_result_infeasible = self._add_result_infeasible_hash
            self.infeasible_results = set()

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
        else:
            self.get_day_symmetry = lambda x: x

        self.calls_to_is_day_infeasible = 0
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

    def _get_result_infeasible_hash(
        self, day: int, patients: dict[Treatment, dict[Patient, int]]
    ):
        input_value = self._get_hash(self.get_day_symmetry(day), patients)
        return input_value in self.infeasible_results

    def _add_result_infeasible_hash(
        self, day: int, patients: dict[Treatment, dict[Patient, int]]
    ):
        self.infeasible_results.add(
            self._get_hash(self.get_day_symmetry(day), patients)
        )

    def _get_result_feasible_hash(
        self,
        day: int,
        patients: dict[Treatment, dict[Patient, int]],
    ):
        input_value = self._get_hash(self.get_day_symmetry(day), patients)
        return input_value in self.feasible_results

    def _get_day_symmetry(self, d):
        # In the future, this method could be used to calculate a hash for the resource profile of that day
        return self.days_symmetry[d]

    def _add_result_feasible_hash(
        self,
        day: int,
        patients: dict[Treatment, dict[Patient, int]],
    ):
        self.feasible_results.add(
            self._get_hash(self.get_day_symmetry(day), patients)
        )  # type:ignore

    def is_day_infeasible(
        self, day: int, patients: dict[Treatment, dict[Patient, int]]
    ) -> dict:
        self.calls_to_is_day_infeasible += 1
        # Check if this day has already been seen with the same configuration and report the result
        if self.store_results:  # type:ignore
            if self.get_result_feasible(day, patients):
                logger.debug("Used stored results")
                return {"status_code": Subsolver.FEASIBLE}

            if self.get_result_infeasible(day, patients):
                logger.debug("Used stored results")
                return {"status_code": Subsolver.COMPLETELY_INFEASIBLE}
        self.calls_to_solve_subsystem += 1
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

            self.add_result_feasible(day, patients)
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
