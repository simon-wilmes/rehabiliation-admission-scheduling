from abc import ABC, abstractmethod
from src.solution import Solution
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


class Solver(ABC):
    BASE_SOLVER_OPTIONS = {
        "number_of_threads": (int, 1, 24),
        "extra_treatments_factor": (float, 1.0, 2.0),
        "use_conflict_groups": (bool,),
        "use_resource_loyalty": (bool,),
        "use_even_distribution": (bool,),
    }
    BASE_SOLVER_DEFAULT_OPTIONS = {
        "number_of_threads": 12,
        "extra_treatments_factor": 1.5,
        "use_conflict_groups": True,
        "use_resource_loyalty": True,
        "use_even_distribution": True,
    }

    def __init__(self, instance: Instance, **kwargs):

        logger.debug(f"Setting options: Solver")
        for key in Solver.BASE_SOLVER_OPTIONS:
            if key in kwargs:
                setattr(self, key, kwargs[key])
                logger.debug(f" ---- {key} to {kwargs[key]}")
            else:
                setattr(self, key, self.__class__.BASE_SOLVER_DEFAULT_OPTIONS[key])
                logger.debug(
                    f" ---- {key} to { self.__class__.BASE_SOLVER_DEFAULT_OPTIONS[key]} (default)"
                )

        self.number_of_threads = kwargs.get("number_of_threads", 12)
        self.extra_treatments_factor = kwargs.get("extra_treatments_factor", 1.5)
        self.instance = instance

    def add_resource_loyal(self):
        return self.use_resource_loyalty  # type: ignore

    def add_even_distribution(self):
        return self.use_even_distribution  # type: ignore

    def add_conflict_groups(self):
        return self.use_conflict_groups  # type: ignore

    @abstractmethod
    def solve_model(self) -> Solution:
        pass

    @abstractmethod
    def create_model(self) -> None:
        pass

    def _create_parameter_sets(self):
        # Similar to the MIP model, create necessary sets and mappings
        self.max_day = max(
            p.admitted_before_date.day + p.length_of_stay
            for p in self.instance.patients.values()
        )
        self.D = range(self.max_day)
        # The time slots are the hours between the workday start and end + plus one, where we
        # set availability to 0, to force the model to stop every treatment before the end of
        # the workday
        self.T = arange(
            self.instance.workday_start.hour,
            self.instance.workday_end.hour + self.instance.time_slot_length.hours,
            self.instance.time_slot_length.hours,
        ).astype(float)

        self.P = list(self.instance.patients.values())
        self.M = list(self.instance.treatments.values())
        self.F = list(self.instance.resources.values())
        self.Fhat = list(self.instance.resource_groups.values())

        self.Fhat_m = {
            m: list(m.resources.keys()) for m in self.instance.treatments.values()
        }
        self.fhat = {
            fhat: [f for f in self.F if f.resource_group == fhat] for fhat in self.Fhat
        }

        self.D_p = {
            p: range(
                p.earliest_admission_date.day,
                p.admitted_before_date.day + p.length_of_stay,
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
        self.lr_pm = {
            (p, m): int(p.treatments[m]) - p.already_scheduled_treatments[m]
            for p in self.instance.patients.values()
            for m in self.M_p[p]
        }

        self.l_p = {p: p.length_of_stay for p in self.instance.patients.values()}
        self.du_m = {
            m: ceil(m.duration / self.instance.time_slot_length)
            for m in self.instance.treatments.values()
        }

        self.b = self.instance.beds_capacity
        self.rw = self.instance.rolling_window_length

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
        self.Lhat_m = {}
        for m in self.M:
            self.Lhat_m[m] = [fhat for fhat in self.Fhat_m[m] if m.loyalty[fhat]]

        # Mapping from resource group to treatments requiring that group
        self.M_fhat = {}
        for fhat in self.Fhat:
            self.M_fhat[fhat] = [m for m in self.M if fhat in self.Fhat_m[m]]

        # Rolling window days
        self.R = self.instance.rolling_window_days

        self.C = self.instance.conflict_groups

        # Calculate the number of treatments needed intotal for all patients together
        self.treatment_count = defaultdict(int)
        for p in self.P:
            for m in self.M_p[p]:
                self.treatment_count[m] += p.treatments[m]

        self.number_treatments_offered = {
            m: ceil(
                sum(p.treatments[m] for p in self.P if m in self.M_p[p])
                / self.k_m[m]
                * self.extra_treatments_factor
            )
            for m in self.M
        }

        week_values = [5, 3, 2, 1, 1, 1, 1, 1, 1]
        self.w_d = {
            d: week_values[floor(d // self.instance.week_length)] for d in self.D
        }
        pass
