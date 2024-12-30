from .subsolver import (
    Subsolver,
)
from src.logging import logger
from src.treatments import Treatment
from src.patients import Patient
from typing import Iterable
from ortools.sat.python import cp_model
from collections import defaultdict
from src.instance import Instance
from src.solvers import Solver
from src.solution import Solution, Appointment
from src.time import DayHour
from itertools import combinations
from time import time


class CPSubsolver2(Subsolver):

    SOLVER_OPTIONS = Subsolver.BASE_SOLVER_OPTIONS.copy()
    SOLVER_OPTIONS.update({})  # Add any additional options here
    SOLVER_DEFAULT_OPTIONS = {}

    def __init__(self, instance: Instance, solver: Solver, **kwargs):
        logger.debug(f"Setting options: {self.__class__.__name__}")

        for key in kwargs:
            assert (
                key in self.__class__.BASE_SOLVER_DEFAULT_OPTIONS
                or key in self.__class__.SOLVER_DEFAULT_OPTIONS
            ), f"Invalid option: {key}"

        for key in self.__class__.SOLVER_DEFAULT_OPTIONS:

            if key in kwargs:
                setattr(self, key, kwargs[key])
                logger.debug(f" ---- {key} to {kwargs[key]}")
            else:
                setattr(self, key, self.__class__.SOLVER_DEFAULT_OPTIONS[key])
                logger.debug(
                    f" ---- {key} to { self.__class__.SOLVER_DEFAULT_OPTIONS[key]} (default)"
                )

        super().__init__(instance, solver, **kwargs)

        self.create_parameters()
        self.time_create_model = 0
        self.time_solve_model = 0

    def create_parameters(self):
        pass

    def create_model(self, day: int, patients: dict[Treatment, dict[Patient, int]]):
        M_p = {p: [m for m in patients if p in patients[m]] for p in self.solver.P}

        model = cp_model.CpModel()
        all_appointments = [m for m in patients.keys() if len(patients[m]) > 0]
        all_patients = {p for m in patients for p in patients[m]}

        max_appointments = {m: sum(patients[m].values()) for m in all_appointments}

        appointment_vars: dict[Treatment, dict[int, dict]] = {
            m: {} for m in all_appointments
        }

        self.all_is_late_scheduled = {}

        for m in all_appointments:
            du_m = self.solver.du_m[m]
            for r in range(max_appointments[m]):
                start_slot = model.new_int_var_from_domain(
                    cp_model.Domain.from_intervals(  # type:ignore
                        [  # type:ignore
                            (0, len(self.solver.T) - du_m),  # type:ignore
                            (len(self.solver.T), 2 * len(self.solver.T)),  # type:ignore
                        ]  # type:ignore
                    ),
                    f"start_time_m{m.id}_r{r}",
                )

                # Create the constraint that we want to minimize the late scheduled treatments
                is_late_scheduled = model.new_bool_var(
                    f"is_late_scheduled_m{m.id}_r{r}"
                )
                model.add(start_slot <= len(self.solver.T) - 1).only_enforce_if(
                    is_late_scheduled.Not()
                )
                self.all_is_late_scheduled[m, r] = is_late_scheduled

                # If treatment is even scheduled
                is_treatment_scheduled = model.new_bool_var(
                    f"schedule_treatment_m{m.id}_r{r}"
                )
                # Create interval variable
                interval = model.new_optional_fixed_size_interval_var(
                    start=start_slot,
                    size=du_m,
                    is_present=is_treatment_scheduled,
                    name=f"interval_m{m.id}_r{r}",
                )

                # Create Resource variables that assign resources to the treatment
                resource_vars = defaultdict(dict)
                for fhat in self.solver.Fhat_m[m]:
                    for f in self.solver.fhat[fhat]:
                        use_resource = model.new_bool_var(
                            f"use_resource_m{m.id}_r{r}_fhat{fhat.id}_f{f.id}",
                        )
                        resource_vars[fhat][f] = use_resource

                        # CONSTRAINT: resources can only be assigned if is_present is true
                        model.add_implication(use_resource, is_treatment_scheduled)

                    # CONSTRAINT R4: Make sure that every treatment has the required resources
                    model.add(
                        cp_model.LinearExpr.Sum(
                            [resource_vars[fhat][f] for f in self.solver.fhat[fhat]]
                        )
                        == self.solver.n_fhatm[fhat, m] * is_treatment_scheduled
                    )

                appointment_vars[m][r] = {
                    "interval": interval,
                    "start_slot": start_slot,
                    "is_present": is_treatment_scheduled,
                    "resources": resource_vars,
                    "is_late_scheduled": is_late_scheduled,
                }

        # Constraint: Enforce symmetry breaking
        for m in all_appointments:
            for r in range(max_appointments[m]):
                if r == 0:
                    continue
                model.add(
                    appointment_vars[m][r]["start_slot"]
                    >= appointment_vars[m][r - 1]["start_slot"]
                )
                model.add(
                    appointment_vars[m][r]["is_present"]
                    <= appointment_vars[m][r - 1]["is_present"]
                )

        # Create patients variables
        patient_vars = {}
        is_patient_late = {}

        for m in patients:
            patient_vars[m] = {}
            is_patient_late[m] = {}
            for p in patients[m]:
                patient_vars[m][p] = {}
                is_patient_late[m][p] = {}

                expr_list = []
                for r in appointment_vars[m]:
                    patient2treatment_var = model.new_bool_var(
                        f"patient2treatment_{p.id}_m{m.id}_r{r}"
                    )
                    is_patient_late_var = model.new_bool_var(
                        f"patient_late_{p.id}_m{m.id}_r{r}"
                    )

                    patient_vars[m][p][r] = patient2treatment_var
                    is_patient_late[m][p][r] = is_patient_late_var

                    expr_list.append(patient2treatment_var)

                    # Constraint: Treatment must be scheduled if patient is assigned
                    model.add_implication(
                        patient2treatment_var, appointment_vars[m][r]["is_present"]
                    )

                    # Constraint if treatment is late then and patient is assigned to it then patient is late with this treatment
                    model.add_implication(
                        appointment_vars[m][r]["is_late_scheduled"], is_patient_late_var
                    ).only_enforce_if(patient2treatment_var)

                # Constraint: every patient is assigned to exactly the required number of treatments
                model.add(cp_model.LinearExpr().Sum(expr_list) == patients[m][p])

            # Constraint: Every Treatment has at most k_m patients if a patient is scheduled to the treatment
            if patients[m]:
                for r in appointment_vars[m]:
                    model.add(
                        cp_model.LinearExpr.Sum(
                            [patient_vars[m][p][r] for p in patients[m]]
                        )
                        <= self.solver.k_m[m]
                    )

        if self.enforce_min_patients_per_treatment:  # type:ignore
            # Constraint: Every Treatment has at least j_m patients if treatment is scheduled
            # If not set boolean variable min_patients[m] to true
            for m in appointment_vars:
                for r in appointment_vars[m]:
                    model.add(
                        cp_model.LinearExpr.Sum(
                            [patient_vars[m][p][r] for p in patients[m]]
                        )
                        >= self.solver.j_m[m] * appointment_vars[m][r]["is_present"]
                    )

        # CONSTRAINT P2: every patient has only a single treatment at a time
        for p in all_patients:
            intervals = []
            for m in M_p[p]:
                for r in appointment_vars[m]:
                    # Create interval variable with length with rest time
                    interval = model.new_optional_fixed_size_interval_var(
                        start=appointment_vars[m][r]["start_slot"],
                        size=self.solver.du_m[m],
                        is_present=patient_vars[m][p][r],
                        name=f"interval_p{p.id}_m{m.id}_r{r}",
                    )

                    intervals.append(interval)

            model.add_no_overlap(intervals)

        # CONSTRAINT: Resource constraints
        for f in self.solver.F:
            intervals_using_f = []
            demands = []
            for fhat in self.solver.Fhat:
                if fhat not in f.resource_groups:
                    continue

                for m in all_appointments:
                    if fhat not in m.resources:
                        continue
                    for r in appointment_vars[m]:
                        new_interval_var = model.new_optional_fixed_size_interval_var(
                            start=appointment_vars[m][r]["start_slot"],
                            size=self.solver.du_m[m],
                            is_present=appointment_vars[m][r]["resources"][fhat][f],
                            name=f"interval_fhat{fhat.id}_f{f.id}_m{m.id}_r{r}",
                        )
                        intervals_using_f.append(new_interval_var)
                        demands.append(1)
            # Add resource availability constraints
            length = 0
            for t_index, t in enumerate(self.solver.T):
                if self.solver.av_fdt[(f, day, t)] == 0:
                    length += 1
                else:
                    if length != 0:
                        start_point = t_index - length
                        interval_f = model.new_fixed_size_interval_var(
                            int(start_point),
                            length,
                            f"interval_no_avail_f{f.id}_t{t}",
                        )
                        intervals_using_f.append(interval_f)
                        length = 0
            if length != 0:
                start_point = len(self.solver.T) - length
                interval_f = model.new_fixed_size_interval_var(
                    int(start_point),
                    length,
                    f"interval_no_avail_f{f.id}_t{t}",
                )
                intervals_using_f.append(interval_f)

            model.add_no_overlap(intervals_using_f)

        # Actually minimize the late scheduled treatments
        all_patients_late_vars = []
        for m in is_patient_late:
            for p in is_patient_late[m]:
                all_patients_late_vars.extend(is_patient_late[m][p].values())

        model.Minimize(
            cp_model.LinearExpr.Sum(all_patients_late_vars)  # type:ignore
        )

        return model, appointment_vars, patient_vars, is_patient_late

    def _get_day_solution(
        self, day: int, patients: dict[Treatment, dict[Patient, int]]
    ) -> list[Appointment]:

        sub_solver = cp_model.CpSolver()
        sub_solver.parameters.log_search_progress = (
            False  # self.solver.log_to_console  # type: ignore
        )
        model, appointment_vars, patient_vars, _ = self.create_model(day, patients)
        status = sub_solver.Solve(model)
        assert status == cp_model.OPTIMAL, f"Subsolver on day {day} is not optimal"
        assert (
            sub_solver.objective_value == 0
        ), f"Eventhough optimality is claimed, subsystem on day {day} is not feasible"

        schedule = []
        for m in appointment_vars:
            for rep in appointment_vars[m]:
                if sub_solver.Value(appointment_vars[m][rep]["is_present"]):
                    start_time = self.solver.T[
                        sub_solver.Value(appointment_vars[m][rep]["start_slot"])
                    ]
                    resources_for_rep = defaultdict(list)
                    for fhat in appointment_vars[m][rep]["resources"]:
                        for f, var in appointment_vars[m][rep]["resources"][
                            fhat
                        ].items():
                            if sub_solver.Value(var):
                                resources_for_rep[fhat].append(f)

                    patients_in_treatment = []
                    for p in patient_vars[m]:
                        if sub_solver.Value(patient_vars[m][p][rep]):
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
        self,
        day: int,
        patients: dict[Treatment, dict[Patient, int]],
    ) -> dict:
        # Restrict M_p to the daily treatment schedule

        # logger.debug("Solving subsystem with CPSubsolver")
        sub_solver = cp_model.CpSolver()
        sub_solver.parameters.log_search_progress = (
            False  # self.solver.log_to_console  # type: ignore
        )
        self.time_create_model -= time()
        model, appointment_vars, patient_vars, is_patient_late = self.create_model(
            day, patients
        )
        self.time_create_model += time()
        self.time_solve_model -= time()
        status = sub_solver.Solve(model)
        self.time_solve_model += time()

        code_dict = {cp_model.OPTIMAL: "OPTIMAL", cp_model.INFEASIBLE: "INFEASIBLE"}
        code = code_dict[status]  # type:ignore
        # logger.debug(f"Model is {code}")
        if status == cp_model.OPTIMAL:
            if sub_solver.objective_value == 0:
                # logger.debug("Objective value is 0")
                # We can schedule everything fine => feasible

                return {"status_code": Subsolver.FEASIBLE}
            else:
                # Calculate all the treatments that are late scheduled
                late_scheduled_mp = defaultdict(int)
                for m in is_patient_late:
                    for p in is_patient_late[m]:
                        for r in is_patient_late[m][p]:
                            if sub_solver.value(is_patient_late[m][p][r]):
                                late_scheduled_mp[(m, p)] += 1

                return {
                    "status_code": Subsolver.TOO_MANY_TREATMENTS,
                    "late_scheduled": late_scheduled_mp,
                }

        # Model is infeasible this means that the minimum patients per treatment were a problem
        return {"status_code": Subsolver.MIN_PATIENTS_PROBLEM}
