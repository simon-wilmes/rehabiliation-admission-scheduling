from .subsolver import (
    Subsolver,
)
from src.logging import logger
from src.treatments import Treatment
from src.patients import Patient
from typing import Iterable
from collections import defaultdict
from src.instance import Instance
from src.solvers import Solver
from src.solution import Solution, Appointment
from src.time import DayHour
from itertools import combinations
from time import time
from docplex.cp.model import CpoModel
from src.utils import CP_PATH
from math import floor


class CPSubsolver3(Subsolver):

    SOLVER_OPTIONS = Subsolver.BASE_SOLVER_OPTIONS.copy()
    SOLVER_OPTIONS.update(
        {
            "estimated_needed_treatments": [True, False],
            "add_patient_symmetry": [True, False],
        }
    )  # Add any additional options here
    SOLVER_DEFAULT_OPTIONS = {
        "estimated_needed_treatments": True,
        "add_patient_symmetry": False,
    }

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

    def create_model(
        self, day: int, treatment_patient_dict: dict[Treatment, dict[Patient, int]]
    ):
        M_p = {
            p: [m for m in treatment_patient_dict if p in treatment_patient_dict[m]]
            for p in self.solver.P
        }

        model = CpoModel()  # type:ignore
        all_appointments = self.solver.M
        all_patients = {
            p for m in treatment_patient_dict for p in treatment_patient_dict[m]
        }

        max_appointments = {
            m: min(
                floor(sum(treatment_patient_dict[m].values()) / m.min_num_participants),
                max(self.solver.J_md[m, day]) + 1,
            )
            for m in all_appointments
        }

        appointment_vars: dict[Treatment, dict[int, dict]] = {
            m: {} for m in all_appointments
        }

        resource_intervals_f = defaultdict(list)

        for m in all_appointments:
            for r in range(max_appointments[m]):

                # Create interval variable
                interval = model.interval_var(  # type:ignore
                    start=(0, len(self.solver.T) - self.solver.du_m[m] - 1),
                    length=self.solver.du_m[m],
                    optional=True,
                    name=f"interval_m{m.id}_r{r}",
                )

                is_treatment_scheduled = model.presence_of(interval)  # type:ignore
                # Create Resource variables that assign resources to the treatment
                resource_vars = defaultdict(dict)
                for fhat in self.solver.Fhat_m[m]:
                    for f in self.solver.fhat[fhat]:
                        use_resource = model.interval_var(  # type:ignore
                            start=(0, len(self.solver.T) - self.solver.du_m[m] - 1),
                            length=self.solver.du_m[m],
                            optional=True,
                            name=f"interval_resource_m{m.id}_r{r}_fhat{fhat.id}_f{f.id}",
                        )

                        resource_vars[fhat][f] = use_resource
                        resource_intervals_f[f].append(use_resource)

                    # CONSTRAINT R4: Make sure that every treatment has the required resources
                    model.add(
                        model.alternative(  # type:ignore
                            interval,
                            resource_vars[fhat].values(),
                            cardinality=self.solver.n_fhatm[fhat, m],
                        )
                    )

                appointment_vars[m][r] = {
                    "interval": interval,
                    "is_present": is_treatment_scheduled,
                    "resources": resource_vars,
                }

        # Constraint: Enforce symmetry breaking
        for m in appointment_vars:
            for r in appointment_vars[m]:
                if r == 0:
                    continue
                model.add(
                    model.end_before_start(  # type:ignore
                        appointment_vars[m][r - 1]["interval"],
                        appointment_vars[m][r]["interval"],
                    )
                )
                model.add(
                    model.if_then(  # type:ignore
                        appointment_vars[m][r]["is_present"],
                        appointment_vars[m][r - 1]["is_present"],
                    )
                )

        # Create patients variables
        patient_vars = {}
        for m in treatment_patient_dict:
            patient_vars[m] = {}
            for p in treatment_patient_dict[m]:
                patient_vars[m][p] = {}
                expr_list = []
                for r in appointment_vars[m]:
                    patient2treatment_var = model.interval_var(  # type:ignore
                        start=(0, len(self.solver.T) - self.solver.du_m[m] - 1),
                        length=self.solver.du_m[m],
                        optional=True,
                        name=f"interval_patient_p{p.id}_m{m.id}_r{r}",
                    )

                    patient_vars[m][p][r] = patient2treatment_var
                    expr_list.append(patient2treatment_var)

                    # # Constraint: Treatment must be scheduled if patient is assigned
                    model.add(
                        model.if_then(  # type:ignore
                            model.presence_of(patient2treatment_var),  # type:ignore
                            appointment_vars[m][r]["is_present"],
                        )
                    )

                # Constraint: every patient is assigned to exactly the required number of treatments
                model.add(
                    sum(model.presence_of(expr) for expr in expr_list)  # type:ignore
                    == treatment_patient_dict[m][p]
                )

        min_patients = {}
        minimizing_objective = []
        for m in appointment_vars:
            min_patients[m] = {}
            for r in appointment_vars[m]:
                min_patients[m][r] = model.binary_var(  # type:ignore
                    f"min_patients_m{m.id}_r{r}"
                )
                # Create dummy intervals
                num_usable_dummys = m.max_num_participants - m.min_num_participants
                num_unusable_dummys = m.min_num_participants
                all_intervals = [
                    patient_vars[m][p][r] for p in treatment_patient_dict[m]
                ]
                all_intervals.extend(
                    model.interval_var_list(  # type:ignore
                        num_usable_dummys, optional=True, name=f"dummy_m{m.id}_r{r}"
                    )
                )
                for _ in range(num_unusable_dummys):
                    unusable_int_var = model.interval_var(  # type:ignore
                        optional=True, name=f"unusable_dummy_m{m.id}_r{r}"
                    )
                    all_intervals.append(unusable_int_var)
                    minimizing_objective.append(
                        model.presence_of(unusable_int_var)  # type:ignore
                    )
                # Constraint: Every Treatment has at between j_m and k_m patients if treatment is scheduled

                model.add(
                    model.alternative(  # type:ignore
                        appointment_vars[m][r]["interval"],
                        all_intervals,
                        cardinality=m.max_num_participants,
                    )
                )

        model.add(model.minimize(sum(minimizing_objective)))  # type:ignore

        # CONSTRAINT P2: every patient has only a single treatment at a time
        for p in all_patients:
            intervals = []
            for m in M_p[p]:
                for r in patient_vars[m][p]:
                    intervals.append(patient_vars[m][p][r])
            model.add(model.no_overlap(intervals))  # type:ignore

        # CONSTRAINT: Resource constraints
        for f in self.solver.F:
            # Add resource availability constraints
            length = 0
            for t_index, t in enumerate(self.solver.T):
                if self.solver.av_fdt[(f, day, t)] == 0:
                    length += 1
                else:
                    if length != 0:
                        start_point = t_index - length
                        interval_f = model.interval_var(  # type:ignore
                            start=int(start_point),
                            length=length,
                            name=f"interval_no_avail_f{f.id}_t{t}",
                        )
                        resource_intervals_f[f].append(interval_f)
                        length = 0
            if length != 0:
                start_point = len(self.solver.T) - length
                interval_f = model.interval_var(  # type:ignore
                    start=int(start_point),
                    length=length,
                    name=f"interval_no_avail_f{f.id}_t{t}",
                )
                resource_intervals_f[f].append(interval_f)

            model.add(model.no_overlap(resource_intervals_f[f]))  # type:ignore

        return model, appointment_vars, patient_vars

    def _get_day_solution(
        self, day: int, patients: dict[Treatment, dict[Patient, int]]
    ) -> list[Appointment]:

        model, appointment_vars, patient_vars = self.create_model(day, patients)
        results = model.solve(
            execfile=CP_PATH, Workers=self.solver.number_of_threads  # type:ignore
        )
        objective = results.get_objective_bound()  # type:ignore
        assert (
            results.is_solution_optimal()  # type:ignore
        ), f"Subsolver on day {day} is not optimal"

        assert (
            objective == 0
        ), f"Eventhough optimality is claimed, subsystem on day {day} is not feasible"

        schedule = []
        solution = results.get_solution()  # type:ignore

        for m in appointment_vars:
            for rep in appointment_vars[m]:
                name = "interval_m{}_r{}".format(m.id, rep)
                result_interval = solution.get_var_solution(name)
                if not result_interval.presence:
                    continue

                start = result_interval.start

                start_time = self.solver.T[start]

                resources_for_rep = defaultdict(list)
                for fhat in appointment_vars[m][rep]["resources"]:
                    for f, var in appointment_vars[m][rep]["resources"][fhat].items():
                        name = f"interval_resource_m{m.id}_r{rep}_fhat{fhat.id}_f{f.id}"
                        if solution.get_var_solution(name).presence:
                            resources_for_rep[fhat].append(f)

                    patients_in_treatment = []
                    for p in patient_vars[m]:
                        name = f"interval_patient_p{p.id}_m{m.id}_r{rep}"
                        if solution.get_var_solution(name).presence:
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

        self.time_create_model -= time()
        model, appointment_vars, patient_vars = self.create_model(day, patients)
        self.time_create_model += time()
        self.time_solve_model -= time()
        result = model.solve(
            execfile=CP_PATH, Workers=self.solver.number_of_threads  # type:ignore
        )
        self.time_solve_model += time()

        if result.is_solution():  # type:ignore
            if result.get_objective_bound() == 0:
                # logger.debug("Objective value is 0")
                # Print out all varibles
                if False:
                    for m in appointment_vars:
                        for rep in appointment_vars[m]:
                            if sub_solver.value(appointment_vars[m][rep]["is_present"]):
                                resources_for_rep = []
                                for fhat in appointment_vars[m][rep]["resources"]:
                                    for f, var in appointment_vars[m][rep]["resources"][
                                        fhat
                                    ].items():
                                        if sub_solver.value(var):
                                            resources_for_rep.append(f)

                                logger.debug(
                                    f"Appointment {m.id}/{rep} at {self.solver.T[sub_solver.value(appointment_vars[m][rep]['start_slot'])]} using resources {resources_for_rep}"
                                )

                            for p in patient_vars[m]:
                                if sub_solver.value(patient_vars[m][p][rep]):
                                    logger.debug(
                                        f"Patient {p.id} assigned to treatment {m.id}/{rep}"
                                    )
                            pass
                        pass
                return {"status_code": Subsolver.FEASIBLE}
            else:
                # logger.debug(
                #    "Objective value is not 0 {}".format(sub_solver.objective_value)
                # )
                # Is solvable if minimum patients is ignored => generate bad cut from this
                return {"status_code": Subsolver.MIN_PATIENTS_PROBLEM}
            # collect num treatments needed and return

        # Model is infeasible this means that we need to generate a cut, either return informations
        return {"status_code": Subsolver.TOO_MANY_TREATMENTS}
