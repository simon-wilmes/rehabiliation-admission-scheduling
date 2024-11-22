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

a = []
b = []


class CPSolver(Solver):

    SOLVER_OPTIONS = Solver.BASE_SOLVER_OPTIONS.copy()
    SOLVER_OPTIONS.update(
        {"product_repr": ["only-if", "leq-constraints"]}
    )  # Add any additional options here

    SOLVER_DEFAULT_OPTIONS = {
        "product_repr": "only-if",
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

    def create_model(self):

        self._create_model()
        self._set_optimization_goal()

    def solve_model(self):
        solver = cp_model.CpSolver()
        solver.parameters.num_search_workers = (
            self.number_of_threads
        )  # Set the number of threads
        status = solver.Solve(self.model)
        if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
            logger.debug(f"Solution found with status {solver.StatusName(status)}")
            logger.debug(f"Objective Value: {solver.ObjectiveValue()}")
            solution = self._extract_solution(solver)
            return solution
        else:
            logger.error("No solution found.")
            return NO_SOLUTION_FOUND

    def _create_model(self):
        self._create_parameter_sets()

        self.model = cp_model.CpModel()
        self._create_constraints()

    def _create_parameter_sets(self):
        super()._create_parameter_sets()
        # Define any additional sets or mappings needed for the model
        # besides the ones created in the parent class
        self.num_time_slots = len(self.T)

    def _set_optimization_goal(self):
        # Objective: Maximize the total number of scheduled treatments, possibly weighted
        # List to store the variables that encode the product w_d[day] * is_assigned_to_treatment
        product_list = []
        for p in self.P:
            for m in self.M_p[p]:
                vars = self.patient_vars[p][m]
                pass

                for rep in range(self.number_treatments_offered[m]):
                    patient_assigned_to_treatment = vars[
                        "patient_treatment_assignment"
                    ][rep]
                    day_slot = self.treatment_vars[m][rep]["day_slot"]

                    # make w_d into list by sorted keys
                    list_w_d = list(self.w_d.items())
                    list_w_d.sort(key=lambda x: x[0])
                    list_w_d = [x[1] for x in list_w_d]

                    # Define an auxiliary variable to represent w[d]
                    w_d_var = self.model.new_int_var(
                        min(list_w_d), max(list_w_d), f"w_d_m{m.id}_r{rep}"
                    )

                    self.model.add_element(day_slot, list_w_d, w_d_var)

                    product = self.model.new_int_var(
                        0, max(list_w_d), f"product_p{p.id}_m{m.id}_r{rep}"
                    )

                    if self.product_repr == "only-if":  # type: ignore

                        self.model.add(product == w_d_var).only_enforce_if(
                            patient_assigned_to_treatment
                        )
                        self.model.add(product == 0).only_enforce_if(
                            patient_assigned_to_treatment.Not()
                        )
                    elif self.product_repr == "leq-constraints":  # type: ignore

                        # This only works because we want to maximize the (sum over) product
                        self.model.add(product <= w_d_var)
                        self.model.add(
                            product <= patient_assigned_to_treatment * max(list_w_d)
                        )

                    else:
                        logger.error("Invalid product representation option.")
                        raise ValueError
                    product_list.append(product)

        self.model.Maximize(cp_model.LinearExpr.Sum(product_list))

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
        self.treatment_vars = defaultdict(dict)
        # Add variables for all the treatments to schedule
        for m, rt in self.number_treatments_offered.items():
            for r in range(rt):
                duration = self.du_m[m]

                start_slot = self.model.new_int_var(
                    0,
                    self.num_time_slots * len(self.D),
                    f"start_time_m{m.id}_r{r}",
                )
                is_treatment_scheduled = self.model.new_bool_var(
                    f"schedule_treatment_m{m.id}_r{r}"
                )
                # Create interval variable
                interval = self.model.new_optional_fixed_size_interval_var(
                    start=start_slot,
                    size=duration,
                    is_present=is_treatment_scheduled,
                    name=f"interval_m{m.id}_r{r}",
                )

                # Create Resource variables that assign resources to the treatment
                resource_vars = defaultdict(dict)
                for fhat in self.Fhat_m[m]:
                    for f in self.fhat[fhat]:
                        use_resource = self.model.new_bool_var(
                            f"use_resource_m{m.id}_r{r}_f{f.id}",
                        )
                        resource_vars[fhat][f] = use_resource

                        # CONSTRAINT: resources can only be assigned if is_present is true
                        self.model.add_implication(use_resource, is_treatment_scheduled)

                    # CONSTRAINT R4: Make sure that every treatment has the required resources

                    self.model.add(
                        cp_model.LinearExpr.Sum(
                            [resource_vars[fhat][f] for f in self.fhat[fhat]]
                        )
                        == self.n_fhatm[fhat, m]
                    )

                # Add variables that store the day of the treatment # TODO: ugly
                day_slot = self.model.new_int_var(
                    0, len(self.D) - 1, f"start_day_m{m.id}_r{r}"
                )

                self.model.add_division_equality(
                    day_slot, start_slot, self.num_time_slots
                )

                self.treatment_vars[m][r] = {
                    "interval": interval,
                    "start_slot": start_slot,
                    "day_slot": day_slot,
                    "is_present": is_treatment_scheduled,
                    "resources": resource_vars,
                }

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
        for p in self.P:
            for m in self.M_p[p]:

                # Create integer variable for admission
                duration = self.du_m[m]

                # Which Treatment variables
                patient_treatment_assignment = {}
                intervals = {}

                for reps, var_dict in self.treatment_vars[m].items():
                    patient2treatment = self.model.new_bool_var(
                        f"patient2treatment_p{p.id}_m{m.id}_r{reps}",
                    )
                    patient_treatment_assignment[reps] = patient2treatment

                    # CONSTRAINT: patient can only have treatment if it is scheduled

                    self.model.add_implication(
                        patient2treatment, var_dict["is_present"]
                    )

                    interval_start_slot = var_dict["start_slot"]
                    # Create interval variable with length with rest time
                    interval = self.model.new_optional_fixed_size_interval_var(
                        start=interval_start_slot,
                        size=duration,  # TODO: add rest time
                        is_present=patient2treatment,
                        name=f"interval_p{p.id}_m{m.id}_r{reps}",
                    )
                    intervals[reps] = interval

                self.patient_vars[p][m] = {
                    "intervals": intervals,
                    "patient_treatment_assignment": patient_treatment_assignment,
                }
        # CONSTRAINT P1: Patient has lr_pm treatments scheduled

        for p in self.P:
            for m in self.M_p[p]:
                patient_vars = self.patient_vars[p][m]
                self.model.add(
                    cp_model.LinearExpr.Sum(
                        [
                            var
                            for rep, var in patient_vars[
                                "patient_treatment_assignment"
                            ].items()
                        ]
                    )
                    == self.lr_pm[p, m]
                )

        # CONSTRAINT A1: A treatment is provided for at most k_m patients
        for m in self.M:
            # loop over all patients that could attend this treatment

            for rep in range(self.number_treatments_offered[m]):
                patients_list = []
                for p in self.P:
                    if m not in self.M_p[p]:
                        continue

                    patients_list.append(
                        self.patient_vars[p][m]["patient_treatment_assignment"][rep]
                    )

                self.model.add(cp_model.LinearExpr.Sum(patients_list) <= self.k_m[m])

        # CONSTRAINT P2: every patient has only a single treatment at a time
        for p, treatment_vars in self.patient_vars.items():
            all_intervals = []
            for m, vars in treatment_vars.items():
                all_intervals.extend(vars["intervals"].values())

            self.model.add_no_overlap(all_intervals)

        # Admission constraints
        for p in self.P:
            admission_day = self.admission_vars[p]
            # If the patient is already admitted, fix the admission day

            if p.already_admitted:
                self.model.add(admission_day == 0)

        # Ensure that admission is in the correct range
        for p in self.P:
            admission_day = self.admission_vars[p]

            self.model.add(admission_day >= p.earliest_admission_date.day)
            self.model.add(admission_day < p.admitted_before_date.day)

        # Ensure treatments are scheduled within admission period
        for p, treatment_vars in self.patient_vars.items():
            for m, vars in treatment_vars.items():
                interval_var = vars["intervals"]
                patient_treatment_assignment = vars["patient_treatment_assignment"]
                for rep in interval_var:
                    # Treatment must start after admission
                    self.model.add(
                        interval_var[rep].start_expr()
                        >= admission_day * self.num_time_slots
                    ).only_enforce_if(patient_treatment_assignment[rep])

                    # Treatment must end before admission end
                    self.model.add(
                        interval_var[rep].start_expr()
                        < (admission_day + self.l_p[p]) * self.num_time_slots
                    ).only_enforce_if(patient_treatment_assignment[rep])

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

        # Resource capacity constraints
        for f in self.F:
            fhat = f.resource_group
            # CONSTRAINT R2, T2
            # Resource can only be used by a single treatment at a time

            # Collect intervals for f
            intervals_using_f = []

            for m, rep in self.treatment_vars.items():
                # Skip treatments that do not use resource f
                if f.resource_group not in self.Fhat_m[m]:
                    continue

                for r, vars in rep.items():
                    interval, start_slot, resources = (
                        vars["interval"],
                        vars["start_slot"],
                        vars["resources"],
                    )

                    interval_f = self.model.new_optional_fixed_size_interval_var(
                        interval.start_expr(),
                        self.du_m[m],
                        resources[fhat][f],
                        f"interval_p{p.id}_m{m.id}_f{f.id}_r{r}",
                    )
                    if f.id == 4:
                        a.append((interval_f, resources[fhat][f]))

                    intervals_using_f.append(interval_f)

            # CONSTRAINT R3: Resource availability constraints
            length = 0
            for d in self.D:
                for t in self.T:
                    if self.av_fdt[(f, d, t)] == 0:
                        length += 1
                    else:
                        if length != 0:
                            start_point = self.time2slot((d, t)) - length
                            interval_f = self.model.new_fixed_size_interval_var(
                                int(start_point),
                                length,
                                f"interval_f{f.id}_d{d}_t{t}",
                            )
                            intervals_using_f.append(interval_f)
                            length = 0
                            if f.id == 4:
                                b.append(interval_f)

            if length != 0:
                start_point = self.time2slot((d, t)) - length
                interval_f = self.model.new_fixed_size_interval_var(
                    int(start_point),
                    length,
                    f"interval_f{f.id}_d{d}_t{t}",
                )
                intervals_using_f.append(interval_f)
                if f.id == 4:
                    b.append(interval_f)
                length = -1
            # No overlap on resource f
            self.model.add_no_overlap(intervals_using_f)
            # a.append(intervals_using_f)

        # Even distribution of treatments
        if self.add_even_distribution():
            logger.warning(
                "Even distribution constraint not implemented for CP solver."
            )

        # Conflict groups
        if self.add_conflict_groups():
            for p in self.P:
                for conflict_group in self.instance.conflict_groups:
                    conflicting_vars = []
                    for (p2, m, r), vars in self.patient_vars.items():
                        if p2 is not p:
                            continue
                        if m in conflict_group:
                            interval, start_slot, resources = vars.values()
                            conflicting_vars.append(start_slot)

                    all_diff_vars = []
                    for var in conflicting_vars:
                        div_var = self.model.new_int_var(
                            0, len(self.D), f"conflict_group{conflict_group}_{var}"
                        )

                        self.model.add_division_equality(
                            div_var, var, self.num_time_slots
                        )

                        all_diff_vars.append(div_var)
                    self.model.add_all_different(all_diff_vars)

    def _extract_solution(self, solver):
        appointments_dict = {}

        for m, var_dict in self.treatment_vars.items():
            for rep, vars in var_dict.items():
                if solver.value(vars["is_present"]) is False:
                    continue

                start_slot = solver.value(vars["start_slot"])
                start_day, start_time = self.slot2time(start_slot)
                start_day_hour = DayHour(start_day, start_time)
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
                appointments_dict[(m, rep)] = {
                    "appointment_parameter": appointment_value,
                    "patients": [],
                }

        for p in self.P:
            for m in self.M_p[p]:
                vars = self.patient_vars[p][m]
                for rep in range(self.number_treatments_offered[m]):
                    if solver.value(vars["patient_treatment_assignment"][rep]):
                        appointments_dict[(m, rep)]["patients"].append(p)

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
        """
        for i in a:
            logger.debug("Constraint")
            logger.debug(i)
            logger.debug(solver.value(i[0].start_expr()))
            logger.debug(solver.value(i[0].size_expr()))
            logger.debug(solver.value(i[0].end_expr()))

            logger.debug(solver.value(i[1]))
        for i in b:
            logger.debug("Constraint")
            logger.debug(i)
            logger.debug(solver.value(i.start_expr()))
            logger.debug(solver.value(i.size_expr()))
        """
        solution = Solution(
            instance=self.instance,
            schedule=appointments,
            patients_arrival=patients_arrival,
            test_even_distribution=self.add_even_distribution(),
            test_conflict_groups=self.add_conflict_groups(),
            test_resource_loyalty=self.add_resource_loyal(),
        )
        return solution
