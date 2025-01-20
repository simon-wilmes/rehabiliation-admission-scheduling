from gurobipy import Model, GRB
import gurobipy as gp
from src.logging import logger
from src.treatments import Treatment
from src.patients import Patient
from typing import Iterable
from collections import defaultdict
from src.instance import Instance
from src.solvers import Solver
from src.solution import Solution, Appointment
from src.solvers.subsolvers import Subsolver
from src.time import DayHour
from time import time
from math import floor


class MIPSubsolver(Subsolver):

    SOLVER_OPTIONS = {
        "restrict_obj_func_to_1": [True, False],
    }
    # Add any additional options here
    SOLVER_DEFAULT_OPTIONS = {
        "restrict_obj_func_to_1": False,
    }

    def __init__(self, instance: Instance, solver: Solver, **kwargs):
        logger.debug(f"Setting options: {self.__class__.__name__}")

        for key in self.__class__.SOLVER_DEFAULT_OPTIONS:

            if key in kwargs:
                setattr(self, key, kwargs[key])
                logger.debug(f" ---- {key} to {kwargs[key]}")
                del kwargs[key]
            else:
                setattr(self, key, self.__class__.SOLVER_DEFAULT_OPTIONS[key])
                logger.debug(
                    f" ---- {key} to {self.__class__.SOLVER_DEFAULT_OPTIONS[key]} (default)"
                )
        super().__init__(instance, solver, **kwargs)
        self.instance = instance
        self.solver = solver
        self.time_create_model = 0
        self.time_solve_model = 0

    def create_model(self, day: int, patients: dict[Treatment, dict[Patient, int]]):
        logger.debug("SUBSOLVER DAY:" + str(day))
        logger.debug("SUBSOLVER PATIENTS:" + str(patients))

        model = Model("Treatment_Scheduling")

        # Get all relevant appointments and patients
        all_appointments = [m for m in patients.keys() if len(patients[m]) > 0]
        all_patients = {p for m in patients for p in patients[m]}

        # Calculate maximum appointments per treatment
        max_appointments = {
            m: min(
                floor(sum(patients[m].values()) / m.min_num_participants),
                max(self.solver.J_md[m, day]) + 1,
            )
            for m in all_appointments
        }

        # Time periods and feasible start times for each treatment
        T = list(range(len(self.solver.T)))
        feasible_starts = {
            m: list(range(len(self.solver.T) - self.solver.du_m[m] + 1))
            for m in all_appointments
        }

        # DECISION VARIABLES

        # x[m,r,t] = 1 if appointment r of treatment m starts at time t
        x = model.addVars(  # type: ignore
            [  # type: ignore
                (m, r, t)
                for m in all_appointments
                for r in range(max_appointments[m])
                for t in feasible_starts[m]
            ],
            vtype=GRB.BINARY,
            name="treatment_start",
        )

        # y[m,r] = 1 if appointment r of treatment m is scheduled
        y = model.addVars(  # type: ignore
            [(m, r) for m in all_appointments for r in range(max_appointments[m])],  # type: ignore
            vtype=GRB.BINARY,
            name="treatment_scheduled",
        )

        # z[m,r,p] = 1 if patient p is assigned to appointment r of treatment m
        z = model.addVars(  # type: ignore
            [  # type: ignore
                (m, r, p)
                for m in all_appointments
                for r in range(max_appointments[m])
                for p in patients[m]
            ],
            vtype=GRB.BINARY,
            name="patient_assignment",
        )

        # w[m,r,fhat,f] = 1 if resource f is assigned to appointment r of treatment m for resource group fhat
        w = model.addVars(  # type: ignore
            [  # type: ignore
                (m, r, fhat, f)
                for m in all_appointments
                for r in range(max_appointments[m])
                for fhat in self.solver.Fhat_m[m]
                for f in self.solver.fhat[fhat]
            ],
            vtype=GRB.BINARY,
            name="resource_assignment",
        )

        # min_patients_violated[m,r] = 1 if minimum patients constraint is violated
        min_patients_violated = model.addVars(  # type: ignore
            [(m, r) for m in all_appointments for r in range(max_appointments[m])],  # type: ignore
            vtype=GRB.BINARY,
            name="min_patients_violated",
        )

        # CONSTRAINTS

        # 1. Only one start time if scheduled
        for m in all_appointments:
            for r in range(max_appointments[m]):
                model.addConstr(sum(x[m, r, t] for t in feasible_starts[m]) == y[m, r])

        # 2. Patient assignment constraints
        for m in all_appointments:
            for p in patients[m]:
                # Each patient gets required number of treatments
                model.addConstr(
                    gp.quicksum(z[m, r, p] for r in range(max_appointments[m]))
                    == patients[m][p]
                )

                # Can only assign to scheduled treatments
                for r in range(max_appointments[m]):
                    model.addConstr(z[m, r, p] <= y[m, r])

        # 3. Treatment capacity constraints
        for m in all_appointments:
            for r in range(max_appointments[m]):
                # Maximum capacity
                model.addConstr(
                    sum(z[m, r, p] for p in patients[m]) <= self.solver.k_m[m] * y[m, r]
                )

                # Minimum capacity with violation variable
                model.addConstr(
                    sum(z[m, r, p] for p in patients[m])
                    >= self.solver.j_m[m] * y[m, r]
                    - self.solver.k_m[m] * min_patients_violated[m, r]
                )

        # 4. Resource assignment constraints
        for m in all_appointments:
            for r in range(max_appointments[m]):
                for fhat in self.solver.Fhat_m[m]:
                    model.addConstr(
                        sum(w[m, r, fhat, f] for f in self.solver.fhat[fhat])
                        == self.solver.n_fhatm[fhat, m] * y[m, r]
                    )

        # 5. No patient overlap constraints
        for p in all_patients:
            M_p = [m for m in patients if p in patients[m]]
            for t in T:
                # Sum of all treatments covering time t must be <= 1
                model.addConstr(
                    gp.quicksum(
                        z[m, r, p] * x[m, r, t_start]
                        for m in M_p
                        for r in range(max_appointments[m])
                        for t_start in feasible_starts[m]
                        if t_start <= t < t_start + self.solver.du_m[m]
                    )
                    <= 1
                )

        # 6. Resource availability and no overlap constraints
        for f in self.solver.F:
            for t in T:
                # Sum of all treatments using resource f at time t must be <= availability
                model.addConstr(
                    gp.quicksum(
                        w[m, r, fhat, f] * x[m, r, t_start]
                        for m in all_appointments
                        for r in range(max_appointments[m])
                        for fhat in self.solver.Fhat_m[m]
                        if f in self.solver.fhat[fhat]
                        for t_start in feasible_starts[m]
                        if t_start <= t < t_start + self.solver.du_m[m]
                    )
                    <= self.solver.av_fdt[(f, day, self.solver.T[t])]
                )

        # Objective: Minimize minimum patient requirement violations
        if self.restrict_obj_func_to_1:  # type: ignore
            obj_func = model.addVar(name="obj_func", vtype=GRB.BINARY)
            model.addConstrs(
                min_patients_violated[m, r] <= obj_func
                for m in all_appointments
                for r in range(max_appointments[m])
            )
            model.setObjective(
                obj_func,
                GRB.MINIMIZE,
            )
        else:
            model.setObjective(
                sum(
                    min_patients_violated[m, r]
                    for m in all_appointments
                    for r in range(max_appointments[m])
                ),
                GRB.MINIMIZE,
            )

        return model, x, y, z, w

    def _get_day_solution(
        self, day: int, patients: dict[Treatment, dict[Patient, int]]
    ) -> list[Appointment]:
        model, x, y, z, w = self.create_model(day, patients)
        model.optimize()

        if model.Status != GRB.OPTIMAL or model.ObjVal > 0:
            raise ValueError(f"No feasible solution found for day {day}")

        schedule = []
        for m in patients:
            for r in range(len([k for k in y if k[0] == m])):
                if y[m, r].X > 0.5:  # Treatment is scheduled
                    # Find start time
                    start_time = None
                    for t in range(len(self.solver.T) - self.solver.du_m[m] + 1):
                        if (m, r, t) in x and x[m, r, t].X > 0.5:
                            start_time = self.solver.T[t]
                            break

                    # Collect assigned resources by group
                    resources_for_rep = defaultdict(list)
                    for fhat in self.solver.Fhat_m[m]:
                        for f in self.solver.fhat[fhat]:
                            if w[m, r, fhat, f].X > 0.5:
                                resources_for_rep[fhat].append(f)

                    # Collect assigned patients
                    patients_in_treatment = []
                    for p in patients[m]:
                        if z[m, r, p].X > 0.5:
                            patients_in_treatment.append(p)

                    schedule.append(
                        Appointment(
                            patients=patients_in_treatment,
                            treatment=m,
                            solver=self.solver,
                            resources=resources_for_rep,
                            start_date=DayHour(day=day, hour=start_time),
                        )
                    )

        return schedule

    def _solve_subsystem(
        self, day: int, patients: dict[Treatment, dict[Patient, int]]
    ) -> dict:
        self.time_create_model -= time()
        model, x, y, z, w = self.create_model(day, patients)
        self.time_create_model += time()

        self.time_solve_model -= time()
        model.optimize()
        self.time_solve_model += time()

        if model.Status == GRB.OPTIMAL:
            if model.ObjVal == 0:
                return {"status_code": self.FEASIBLE}
            else:
                return {"status_code": self.MIN_PATIENTS_PROBLEM}

        return {"status_code": self.TOO_MANY_TREATMENTS}
