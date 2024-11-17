from abc import ABC, abstractmethod
from src.solution import Solution
from src.instance import Instance
from math import ceil
from numpy import arange
from typing import Any
from src.logging import logger
from src.patients import Patient
from src.treatments import Treatment
from src.resource import Resource, ResourceGroup
from src.time import DayHour
from pprint import pprint as pp


class Solver(ABC):
    def __init__(self, instance: Instance, constraints_ignore: list[str] = set()):
        self.all_constraints = set("rest-time", "resource-loyalty", "even-distribution")
        assert constraints_ignore <= self.all_constraints
        self.all_constraints = self.all_constraints - constraints_ignore

        self.instance = instance

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
        
        pass
