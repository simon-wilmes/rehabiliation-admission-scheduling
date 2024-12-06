from src.solvers.solver import Solver

from ortools.sat.python import cp_model
from datetime import timedelta
from math import ceil
from collections import defaultdict
from src.logging import logger
from src.instance import Instance
from src.time import DayHour, Duration

from src.solution import Solution, Appointment
from src.solution import NO_SOLUTION_FOUND
from copy import copy
from itertools import product
from src.patients import Patient


class CPSolver2(Solver):

    SOLVER_OPTIONS = Solver.BASE_SOLVER_OPTIONS.copy()

    SOLVER_OPTIONS.update(
        {
            "product_repr": ["only-if", "leq-constraints"],
            "treatment-scheduling": ["2d", "1d"],
            "break_symmetry": ["False", "ordering", "2d-view"],
        }
    )  # Add any additional options here

    SOLVER_DEFAULT_OPTIONS = {
        "product_repr": "only-if",
        "treatment-scheduling": "1d",
        "break_symmetry": "False",
    }

    def __init__(self, instance: Instance, **kwargs):
        logger.debug(f"Setting options: {self.__class__.__name__}")
        for key in self.__class__.SOLVER_DEFAULT_OPTIONS:
            if key in kwargs:
                setattr(self, key, kwargs[key])
                logger.debug(f" ---- {key} to {kwargs[key]}")
            else:
                setattr(self, key, self.__class__.SOLVER_DEFAULT_OPTIONS[key])
                logger.debug(
                    f" ---- {key} to { self.__class__.SOLVER_DEFAULT_OPTIONS[key]} (default)"
                )

        super().__init__(instance, **kwargs)

    def _solve_model(self):
        solver = cp_model.CpSolver()
        solver.parameters.log_search_progress = self.log_to_console  # type: ignore
        solver.parameters.num_search_workers = (
            self.number_of_threads  # type: ignore
        )  # Set the number of threads
        self.solver = solver
        status = solver.Solve(self.model)
        if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
            logger.debug(f"Solution found with status {solver.StatusName(status)}")
            logger.debug(f"Objective Value: {solver.ObjectiveValue()}")
            solution = self._extract_solution(solver)
            return solution
        else:
            logger.info("No solution found.")
            return NO_SOLUTION_FOUND

    def _create_model(self):
        self._create_parameter_sets()

        self.model = cp_model.CpModel()
        self._create_constraints()
        if self.break_symmetry != "False":  # type: ignore
            self._break_symmetry()
        self._set_optimization_goal()

    def _break_symmetry(self):
        for m, r, d in self.treat_rep_day_vars:
            for d2 in range(d):
                self.model.add_implication(
                    self.treat_rep_day_vars[m, r, d]["is_present"],
                    self.treat_rep_day_vars[m, r, d2]["is_present"].Not(),
                )
        for m, r, d in self.treat_rep_day_vars:
            for d2 in range(d):
                self.model.add(
                    self.treat_rep_day_vars[m, r, d]["start_slot"]
                    >= self.treat_rep_day_vars[m, r, d2]["start_slot"]
                )

    def _create_parameter_sets(self):
        super()._create_parameter_sets()
        # Define any additional sets or mappings needed for the model
        # besides the ones created in the parent class
        self.num_time_slots = len(self.T)


    def _set_optimization_goal(self):
        # Objective: Maximize the total number of scheduled treatments, possibly weighted
        # List to store the variables that encode the product w_d[day] * is_assigned_to_treatment

        delay_list = []

        for p, admission_day in self.admission_vars.items():
            delay_list.append(self.delay_value * (admission_day - p.earliest_admission_date.day))  # type: ignore

        treatment_list = []
        for (m, r, d), vars in self.treat_rep_day_vars.items():
            treatment_list.append(self.treatment_value * vars["is_present"])  # type: ignore

        missing_treatment_list = []
        for p in self.P:
            for m in self.M_p[p]:
                patient_vars = self.patient_vars[p][m]

                missing_treatment_list.append(
                    (
                        (
                            self.lr_pm[p, m]
                            - cp_model.LinearExpr.Sum(
                                [
                                    var  # type: ignore
                                    for var in self.patient_treat_vars[p, m].values()
                                ]
                            )
                        )
                        * self.missing_treatment_value  # type: ignore
                    )
                )

        obj_list = delay_list + treatment_list + missing_treatment_list

        self.treat_var = self.model.new_int_var(-10000, 1000000, "t_var")
        self.model.add(self.treat_var == cp_model.LinearExpr.Sum(treatment_list))

        self.missing_var = self.model.new_int_var(-10000, 1000000, "missing_var")
        self.model.add(
            self.missing_var == cp_model.LinearExpr.Sum(missing_treatment_list)
        )

        self.delay_var = self.model.new_int_var(-10000, 1000000, "delay_var")
        self.model.add(self.delay_var == cp_model.LinearExpr.Sum(delay_list))

        self.model.Minimize(cp_model.LinearExpr.Sum(obj_list))

    def slot2time(self, i: int) -> tuple[int, float]:
        return (
            i // self.num_time_slots,
            (i % self.num_time_slots) * self.instance.time_slot_length.hours
            + self.instance.workday_start.hour,
        )

    def time2slot(self, t: tuple[int, int]) -> float:
        return (
            t[0] * self.num_time_slots
            + (t[1] - self.instance.workday_start.hour)
            / self.instance.time_slot_length.hours
        )

    def _create_constraints(self):
        self.treatment_vars = defaultdict(lambda: defaultdict(dict))
        self.treat_rep_vars = defaultdict(dict)
        self.treat_rep_day_vars = defaultdict(dict)
        for d in self.D:
            intervals_using_f = defaultdict(list)
            for m in self.M:
                duration = self.du_m[m]
                for r in self.I_m[m]:
                    start_slot = self.model.new_int_var(
                        0,
                        len(self.T) - duration,
                        f"start_time_m{m.id}_r{r}_d{d}",
                    )
                    is_treatment_scheduled_on_d = self.model.new_bool_var(
                        f"is_schedule_treatment_m{m.id}_r{r}_d{d}"
                    )
                    # Create interval variable
                    interval = self.model.new_optional_fixed_size_interval_var(
                        start=start_slot,
                        size=duration,
                        is_present=is_treatment_scheduled_on_d,
                        name=f"interval_m{m.id}_r{r}",
                    )

                    # Create Resource variables that assign resources to the treatment
                    resource_vars = defaultdict(dict)
                    for fhat in self.Fhat_m[m]:
                        for f in self.fhat[fhat]:
                            use_resource_on_d = self.model.new_bool_var(
                                f"use_resource_m{m.id}_r{r}_f{f.id}_d{d}",
                            )
                            resource_vars[fhat][f] = use_resource_on_d

                            # CONSTRAINT: resources can only be assigned if is_present is true

                            self.model.add_implication(
                                use_resource_on_d, is_treatment_scheduled_on_d
                            )

                            interval_using_f = (
                                self.model.new_optional_fixed_size_interval_var(
                                    start=start_slot,
                                    size=duration,
                                    is_present=use_resource_on_d,
                                    name=f"interval_m{m.id}_r{r}_f{f.id}_d{d}",
                                )
                            )
                            intervals_using_f[f].append(interval_using_f)

                        # CONSTRAINT R4: Make sure that every treatment has the required resources

                        self.model.add(
                            cp_model.LinearExpr.Sum(
                                [resource_vars[fhat][f] for f in self.fhat[fhat]]
                            )
                            == self.n_fhatm[fhat, m] * is_treatment_scheduled_on_d
                        )

                    self.treatment_vars[m][r][d] = {
                        "interval": interval,
                        "start_slot": start_slot,
                        "is_present": is_treatment_scheduled_on_d,
                        "resources": resource_vars,
                    }

                    self.treat_rep_vars[(m, r)] = self.treatment_vars[m][r]
                    self.treat_rep_day_vars[(m, r, d)] = self.treatment_vars[m][r][d]

            # # CONSTRAINT R2, T2: Resources can only be used by one treatment at a time
            for f in intervals_using_f.keys():
                # create intervals for unavailable times
                length = 0
                for ind, t in enumerate(self.T):
                    if self.av_fdt[(f, d, t)] == 0:
                        length += 1
                    else:
                        if length != 0:
                            start_point = ind - length
                            blocked_interval_f = self.model.new_fixed_size_interval_var(
                                int(start_point),
                                length,
                                f"interval_f{f.id}_d{d}_t{t}",
                            )
                            intervals_using_f[f].append(blocked_interval_f)
                            length = 0
                ind += 1
                if length != 0:
                    start_point = ind - length
                    blocked_interval_f = self.model.new_fixed_size_interval_var(
                        int(start_point),
                        length,
                        f"interval_f{f.id}_d{d}_t{t}",
                    )
                    intervals_using_f[f].append(blocked_interval_f)

                # CONSTRAINT R3: Resource availability constraints

                self.model.add_no_overlap(intervals_using_f[f])

        # CONSTRAINT: Every treatment can only be scheduled once
        for (m, r), day_vars in self.treat_rep_vars.items():

            self.model.add_at_most_one(vars["is_present"] for vars in day_vars.values())

            pass

        # Variables for patient admission dates
        self.admission_vars = {}
        for p in self.P:
            # Admission date variable
            admission_day = self.model.new_int_var(
                p.earliest_admission_date.day,
                p.admitted_before_date.day,
                f"admission_day_p{p.id}",
            )
            self.admission_vars[p] = admission_day

        # Variables for treatments and resources
        self.patient_vars = defaultdict(dict)
        self.patient_treat_rep_vars = defaultdict(dict)
        self.patient_treat_vars = defaultdict(dict)
        for p in self.P:
            self.patient_vars[p] = defaultdict(lambda: defaultdict(dict))
            for d in self.A_p[p]:
                intervals_per_day = []
                for m in self.M_p[p]:
                    duration = self.du_m[m]
                    for r in self.I_m[m]:

                        patient2treatment = self.model.new_bool_var(
                            f"patient2treatment_p{p.id}_m{m.id}_r{r}_d{d}",
                        )
                        self.patient_vars[p][m][r][d] = patient2treatment
                        self.patient_treat_rep_vars[(p, m, r)][d] = patient2treatment
                        self.patient_treat_vars[(p, m)][r, d] = patient2treatment

                        # CONSTRAINT: treatment must be scheduled if patient is assigned to it

                        self.model.add_implication(
                            patient2treatment,
                            self.treatment_vars[m][r][d]["is_present"],
                        )

                        # Create interval variable with length with rest time
                        interval = self.model.new_optional_fixed_size_interval_var(
                            start=self.treatment_vars[m][r][d]["start_slot"],
                            size=duration,  # TODO: add rest time
                            is_present=patient2treatment,
                            name=f"interval_p{p.id}_d{d}_m{m.id}_r{r}",
                        )
                        intervals_per_day.append(interval)

                # CONSTRAINT P2: no treatment overlaps for a patient on day d

                self.model.add_no_overlap(intervals_per_day)

        # CONSTRAINT P1: Patient has lr_pm treatments scheduled
        for p in self.P:
            for m in self.M_p[p]:

                self.model.add(
                    cp_model.LinearExpr.Sum(
                        [var for var in self.patient_treat_vars[p, m].values()]
                    )
                    <= self.lr_pm[p, m]
                )
                pass

        # CONSTRAINT A1: A treatment is provided for at most k_m patients
        tmp_treat_rep_day_vars = defaultdict(list)
        for (p, m, r), day_vars in self.patient_treat_rep_vars.items():
            for d, vars in day_vars.items():
                tmp_treat_rep_day_vars[(m, r, d)].append(vars)

        for (m, r, d), patient_vars in tmp_treat_rep_day_vars.items():
            self.model.add(cp_model.LinearExpr.Sum(patient_vars) <= self.k_m[m])

        pass
        # Admission constraints
        for p in self.P:
            admission_day = self.admission_vars[p]
            # If the patient is already admitted, fix the admission day

            if p.already_admitted:
                self.model.add(admission_day == 0)
                pass

        # Ensure that admission is in the correct range
        for p in self.P:
            admission_day = self.admission_vars[p]

            self.model.add(admission_day >= p.earliest_admission_date.day)

            self.model.add(admission_day < p.admitted_before_date.day)

        # Ensure treatments are scheduled within admission period
        for (p, m, r), day_vars in self.patient_treat_rep_vars.items():

            admission_day = self.admission_vars[p]
            for d, var in day_vars.items():
                new_bool_var = self.model.new_bool_var(
                    f"adm_before{p.id}_{m.id}_{r}_{d}"
                )
                self.model.add(admission_day > d).only_enforce_if(new_bool_var)
                self.model.add(admission_day <= d).only_enforce_if(~new_bool_var)
                self.model.add(var == 0).only_enforce_if(new_bool_var)

                new_bool_var = self.model.new_bool_var(
                    f"adm_before{p.id}_{m.id}_{r}_{d}"
                )
                self.model.add(admission_day < d - self.l_p[p]).only_enforce_if(
                    new_bool_var
                )
                self.model.add(admission_day >= d - self.l_p[p]).only_enforce_if(
                    ~new_bool_var
                )
                self.model.add(var == 0).only_enforce_if(new_bool_var)

        # Bed capacity constraints via reservoir constraint
        bed_changes_time = []
        bed_changes_amount = []
        for p in self.P:
            admission_day = self.admission_vars[p]
            bed_changes_time.append(admission_day)
            bed_changes_amount.append(1)

            bed_changes_time.append(admission_day + self.l_p[p])
            bed_changes_amount.append(-1)

        self.model.add_reservoir_constraint(
            times=bed_changes_time,
            level_changes=bed_changes_amount,
            min_level=0,
            max_level=self.instance.beds_capacity,
        )

        # Even distribution of treatments
        if self.add_even_distribution():
            logger.warning(
                "Even distribution constraint not implemented for CP solver."
            )

        # Conflict groups
        if self.add_conflict_groups():
            logger.warning("Conflict Groups constraint not implemented for CP solver.")

    def _extract_solution(self, solver):
        appointments_dict = {}

        for (m, r, d), vars in self.treat_rep_day_vars.items():

            if solver.value(vars["is_present"]) == 0:
                continue

            start_slot = solver.value(vars["start_slot"])
            start_time = self.T[start_slot]
            start_day_hour = DayHour(d, start_time)  # type: ignore
            # Collect resources assigned
            assigned_resources = defaultdict(list)
            for fhat, resources_var in vars["resources"].items():
                for f, resource_var in resources_var.items():
                    if solver.value(resource_var):
                        assigned_resources[fhat].append(f)

            appointment_value = {
                "treatment": m,
                "start_date": start_day_hour,
                "resources": assigned_resources,
            }
            appointments_dict[(m, r)] = {
                "appointment_parameter": appointment_value,
                "patients": [],
            }

        for (p, m), rep_vars in self.patient_treat_vars.items():
            for (r, d), vars in rep_vars.items():

                if solver.value(vars):
                    appointments_dict[(m, r)]["patients"].append(p)

        appointments = []
        for key in appointments_dict:
            value = appointments_dict[key]
            appointment_parameter = value["appointment_parameter"]
            patients = value["patients"]
            # logger.debug(patients)
            # logger.debug(appointment_parameter)
            # logger.debug(key)
            if len(patients) == 0:
                logger.warning("Treatment scheduled without patients. ")
                continue
            appointments.append(Appointment(patients=patients, **appointment_parameter))
        patients_arrival = {}
        for p in self.P:
            admission_day = self.admission_vars[p]
            admission_day_value = solver.value(admission_day)
            patients_arrival[p] = DayHour(
                day=admission_day_value, hour=int(self.instance.workday_start.hour)
            )
        solution = Solution(
            instance=self.instance,
            schedule=appointments,
            patients_arrival=patients_arrival,
            solver=self,
            test_even_distribution=self.add_even_distribution(),
            test_conflict_groups=self.add_conflict_groups(),
            test_resource_loyalty=self.add_resource_loyal(),
            solution_value=solver.objective_value,
        )
        return solution

    def _assert_patients_arrival_day(self, patient: Patient, day: int):
        self.model.add(self.admission_vars[patient] == day)

    def _assert_appointment(self, appointment: Appointment):
        if not hasattr(self, "patient_rep"):
            self.patient_rep = defaultdict(int)
            self.treatment_rep = defaultdict(int)  #

        m = appointment.treatment
        resources = appointment.resources
        d = appointment.start_date.day

        for p in appointment.patients:
            r = self.patient_rep[p, m]
            self.patient_rep[p] += 1
            self.model.add(self.patient_treat_rep_vars[p, m, r][d] == 1)

        r = self.treatment_rep[m]
        self.treatment_rep[m] += 1
        for fhat, fs in resources.items():
            for f in fs:
                self.model.add(self.treatment_vars[m][r][d]["resources"][fhat][f] == 1)
