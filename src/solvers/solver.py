from abc import ABC, abstractmethod

from src.solution import Solution, Appointment
from src.instance import Instance
from math import ceil, floor
from numpy import arange
from typing import Any
from src.logging import logger
from src.patients import Patient
from src.treatments import Treatment
from src.resource import Resource, ResourceGroup
from src.time import DayHour
from pprint import pprint as pp
from collections import defaultdict
from time import time


class Solver(ABC):
    BASE_SOLVER_OPTIONS = {
        "number_of_threads": [1, 4, 12],
        "treatment_value": (float, 0.0, 10.0),
        "delay_value": (float, 0.0, 10.0),
        "missing_treatment_value": (float, 0.0, 10.0),
        "use_conflict_groups": [True, False],
        "use_resource_loyalty": [True, False],
        "use_even_distribution": [True, False],
        "log_to_console": [True, False],
    }
    BASE_SOLVER_DEFAULT_OPTIONS = {
        "number_of_threads": 4,
        "treatment_value": 4,
        "delay_value": 2,
        "missing_treatment_value": 8,
        "use_conflict_groups": True,
        "use_resource_loyalty": True,
        "use_even_distribution": True,
        "log_to_console": True,
        "no_rel_heur_time": 3,
    }

    def __init__(self, instance: Instance, **kwargs):

        # Set dummy values for weight parameters so that python does not complain that
        # these attr dont exist, as they are set dynamically via the default options
        self.treatment_value = 0.0
        self.delay_value = 0.0
        self.missing_treatment_value = 0.0

        logger.debug(f"Setting options: Solver")
        for key in Solver.BASE_SOLVER_DEFAULT_OPTIONS:
            if key in kwargs:
                setattr(self, key, kwargs[key])
                logger.debug(f" ---- {key} to {kwargs[key]}")
            else:
                setattr(self, key, self.__class__.BASE_SOLVER_DEFAULT_OPTIONS[key])
                logger.debug(
                    f" ---- {key} to { self.__class__.BASE_SOLVER_DEFAULT_OPTIONS[key]} (default)"
                )

        self.instance = instance

    def add_resource_loyal(self):
        return self.use_resource_loyalty  # type: ignore

    def add_even_distribution(self):
        return True  # self.use_even_distribution  # type: ignore

    def add_conflict_groups(self):
        return self.use_conflict_groups  # type: ignore

    def get_avg_treatments_per_e_w(self, p: Patient):
        return sum(self.lr_pm[p, m] for m in self.M_p[p]) * self.e_w / p.length_of_stay

    def solve_model(self, check_better_solution=True) -> Solution:
        logger.info("Solving model: %s", self.__class__.__name__)
        self.time_solve_model = time()

        solution = self._solve_model()

        if check_better_solution and type(solution) is Solution:
            solution.check_other_solvers()

        self.time_solve_model = time() - self.time_solve_model
        if type(solution) is Solution:
            logger.info(
                "Time to find solution: %ss with value %f",
                round(self.time_solve_model, 3),
                solution.value,
            )
            logger.info(
                "Total Time: %ss", round(time() - self.time_create_model_start, 3)
            )
            self.total_time = round(time() - self.time_create_model_start, 3)
        else:
            logger.info(
                "Time to show infeasibility: %ss", round(self.time_solve_model, 3)
            )
        return solution

    def create_model(self) -> None:

        logger.info("Create model: %s", self.__class__.__name__)
        self.time_create_model_start = time()
        self._create_model()
        time_create_model = time() - self.time_create_model_start
        logger.info("Time to create model: %s", round(time_create_model, 3))

    @abstractmethod
    def _solve_model() -> Solution:
        pass

    @abstractmethod
    def _create_model(self) -> None:
        pass

    def _create_parameter_sets(self):
        self.P = list(self.instance.patients.values())
        # Similar to the MIP model, create necessary sets and mappings

        self.max_day = max(
            p.admitted_before_date.day for p in self.instance.patients.values()
        )
        horizon_length = self.instance.horizon_length
        self.D_max = range(self.max_day)  # highest day that a patient could be admitted

        self.h = horizon_length

        self.D = range(
            min(
                horizon_length,
                max(p.admitted_before_date.day + p.length_of_stay - 1 for p in self.P),
            )
        )

        # The time slots are the hours between the workday start and end + plus one, where we
        # set availability to 0, to force the model to stop every treatment before the end of
        # the workday
        self.T: list[float] = list(
            arange(
                self.instance.workday_start.hour,
                self.instance.workday_end.hour + self.instance.time_slot_length.hours,
                self.instance.time_slot_length.hours,
            ).astype(float)
        )

        self.M = list(self.instance.treatments.values())
        self.F = list(self.instance.resources.values())
        self.Fhat = list(self.instance.resource_groups.values())

        self.Fhat_m = {
            m: list(m.resources.keys()) for m in self.instance.treatments.values()
        }
        self.fhat = {
            fhat: [f for f in self.F if f.resource_group == fhat] for fhat in self.Fhat
        }
        p2 = self.P[0]
        self.A_p = {  # For a patient p the days at which a treatment might be scheduled
            p: list(
                range(
                    p.earliest_admission_date.day,
                    min(
                        p.admitted_before_date.day + p.length_of_stay - 1,
                        horizon_length,
                    ),
                ),
            )
            for p in self.instance.patients.values()
        }

        self.D_p = {  # For a patient p the days at which they might be admitted
            p: list(
                range(
                    p.earliest_admission_date.day,
                    p.admitted_before_date.day,
                )
            )
            for p in self.instance.patients.values()
        }

        self.M_p = {
            p: list(p.treatments.keys()) for p in self.instance.patients.values()
        }
        self.k_m = {t: t.num_participants for t in self.instance.treatments.values()}
        self.r_pm = {
            (p, m): int(p.treatments[m])
            for p in self.instance.patients.values()
            for m in self.M_p[p]
        }
        self.lr_pm = defaultdict(
            int,
            {
                (p, m): int(p.treatments[m]) - p.already_scheduled_treatments[m]
                for p in self.instance.patients.values()
                for m in self.M_p[p]
            },
        )

        self.l_p = {p: p.length_of_stay for p in self.instance.patients.values()}
        self.du_m = {
            m: ceil(m.duration / self.instance.time_slot_length)
            for m in self.instance.treatments.values()
        }

        self.b = self.instance.beds_capacity

        # Resource availability
        self.av_fdt = {
            (f, d, t): int(
                f.is_available(DayHour(day=d, hour=t))
            )  # t is not the last time slot
            for f in self.instance.resources.values()
            for d in self.D
            for t in self.T
        }
        self.n_fhatm = {
            (fhat, m): m.resources[fhat] for m in self.M for fhat in self.Fhat_m[m]
        }
        # self.Lhat_m = {}
        # for m in self.M:
        #    self.Lhat_m[m] = [fhat for fhat in self.Fhat_m[m] if m.loyalty[fhat]]

        # Mapping from resource group to treatments requiring that group
        self.M_fhat = {}
        for fhat in self.Fhat:
            self.M_fhat[fhat] = [m for m in self.M if fhat in self.Fhat_m[m]]

        self.C = self.instance.conflict_groups

        # Calculate the number of treatments needed intotal for all patients together
        self.treatment_count = defaultdict(int)
        for p in self.P:
            for m in self.M_p[p]:
                self.treatment_count[m] += p.treatments[m]

        max_treatment_people_specific = defaultdict(int)
        for m in self.M:
            for p in self.P:
                if m in self.M_p[p]:

                    max_treatment_people_specific[m] = max(
                        p.treatments[m] - p.already_scheduled_treatments[m],
                        max_treatment_people_specific[m],
                    )

        self.n_m = {
            m: max(
                ceil(self.treatment_count[m] / self.k_m[m] * 2),  # type: ignore
                max_treatment_people_specific[m],
            )
            for m in self.M
        }

        self.I_m_calc = {
            m: list(
                range(
                    max(
                        ceil(self.treatment_count[m] / self.k_m[m] * 2),  # type: ignore
                        max_treatment_people_specific[m],
                    )
                ),
            )
            for m in self.M
        }

        self.I_m_max = {m: list(range(self.treatment_count[m])) for m in self.M}

        self.I_m = self.I_m_max

        self.e_w: int = int(self.instance.even_scheduling_width)  # type: ignore
        self.e_lb: float = self.instance.even_scheduling_lower  # type: ignore
        self.e_ub: float = self.instance.even_scheduling_upper  # type: ignore

        self.e_w_upper = {
            p: ceil(self.get_avg_treatments_per_e_w(p) * self.e_ub) for p in self.P
        }
        self.e_w_lower = {
            p: floor(self.get_avg_treatments_per_e_w(p) * self.e_lb) for p in self.P
        }
        self.daily_upper = {
            p: ceil(
                self.get_avg_treatments_per_e_w(p)
                / self.e_w
                * self.instance.daily_scheduling_upper
            )
            for p in self.P
        }

    def _assert_patients_arrival_day(self, patient: Patient, day: int):
        logger.error("Assert patients_arrival_day not implemented")
        raise NotImplementedError

    def _assert_appointment(self, appointment: Appointment):
        logger.error("Assert appointment not implemented")
        raise NotImplementedError

    def assert_solution(self, solution: Solution):
        patients_arrival = solution.patients_arrival

        for patient, day in patients_arrival.items():
            self._assert_patients_arrival_day(patient, day.day)

        self._assert_schedule(solution.schedule)

    def _assert_schedule(self, schedule: list[Appointment]):
        for app in schedule:
            self._assert_appointment(app)

    def _min_needed_repetitions(self, m: Treatment):
        return max(
            ceil(self.treatment_count[m] / self.k_m[m]),
            max(self.lr_pm[p, m] for p in self.P if m in self.M_p[p]),
        )
