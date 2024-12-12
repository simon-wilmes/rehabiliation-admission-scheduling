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
from src.patients import Patient
from copy import copy
from math import floor, gcd
from pprint import pformat


class CPSolver(Solver):

    SOLVER_OPTIONS = Solver.BASE_SOLVER_OPTIONS.copy()
    SOLVER_OPTIONS.update(
        {
            "break_symmetry": [True, False],
            "add_knowledge": [True, False],
            "min_repr": ["cumulative", "reservoir", "day-variables"],
            "max_repr": ["cumulative", "reservoir"],
        }
    )  # Add any additional options here

    SOLVER_DEFAULT_OPTIONS = {
        "break_symmetry": True,
        "min_repr": "reservoir",
        "max_repr": "cumulative",
        "add_knowledge": True,
    }

    def __init__(self, instance: Instance, **kwargs):
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
            logger.debug("Sub Objectives:")
            logger.debug(f"(SOLVER):Minimize Treatments: {solver.value(self.mt)}")
            logger.debug(f"(SOLVER):Minimize Delay: {solver.value(self.md)}")
            logger.debug(
                f"(SOLVER):Minimize Missing Treatments: {solver.value(self.mmt)}"
            )

            solution = self._extract_solution(solver)
            return solution
        else:
            logger.info("No solution found.")
            return NO_SOLUTION_FOUND

    def _create_model(self):
        self._create_parameter_sets()

        self.model = cp_model.CpModel()
        self._create_constraints()
        if self.break_symmetry:  # type: ignore
            self._break_symmetry()

        if self.add_knowledge:  # type:ignore
            self._add_knowledge_num_needed_treatments()
        self._set_optimization_goal()

    def _create_parameter_sets(self):
        super()._create_parameter_sets()
        # Define any additional sets or mappings needed for the model
        # besides the ones created in the parent class
        self.num_time_slots = len(self.T) * 2

    def _break_symmetry(self):
        logger.info("Symmetry is being broken")
        for m, rep_dict in self.treatment_vars.items():
            for r in rep_dict:
                if r == 0:
                    continue
                self.model.add(
                    rep_dict[r]["start_slot"] >= rep_dict[r - 1]["start_slot"]
                )
                self.model.add_implication(
                    rep_dict[r]["is_present"], rep_dict[r - 1]["is_present"]
                )

    def _add_knowledge_num_needed_treatments(self):
        # Calculate minimum number of repetitions needed for each treatment
        for m in self.M:
            # min_reps = self._min_needed_repetitions(m)
            # self.model.add(self.treatment_vars[m][min_reps - 1]["is_present"] == 1)
            pass

    def _set_optimization_goal(self):
        self.obj_factor = 1
        for p in self.P:
            if self.obj_factor % p.length_of_stay != 0:
                self.obj_factor = gcd(self.obj_factor, p.length_of_stay)

        delay_list = []
        for p, admission_day in self.admission_vars.items():
            delay_list.append(
                self.obj_factor
                * self.delay_value
                * (admission_day - p.earliest_admission_date.day)
            )

        treatment_list = []
        for m, rep_vars in self.treatment_vars.items():
            for r, vars in rep_vars.items():
                treatment_list.append(
                    self.treatment_value * self.obj_factor * vars["is_present"]
                )

        missing_treatment_list = []

        for p in self.P:
            for m in self.M_p[p]:
                missing_treatment_list.extend(
                    [
                        -self.missing_treatment_value * var["patient2treatment"]
                        for var in self.patient_vars[p][m].values()
                    ]
                )

            total_treatments = sum(
                self.missing_treatment_value * self.lr_pm[p, m] for m in self.M_p[p]
            )
            missing_treatment_list += [total_treatments]

        self.mt = self.model.new_int_var(-10000, 100000, name="minimiz_treatments")
        self.md = self.model.new_int_var(-10000, 100000, name="minimiz_delay")
        self.mmt = self.model.new_int_var(
            -10000, 100000, name="minimiz_missing_treatments"
        )
        self.model.add(self.mt == cp_model.LinearExpr.Sum(treatment_list))
        self.model.add(self.md == cp_model.LinearExpr.Sum(delay_list))
        self.model.add(self.mmt == cp_model.LinearExpr.Sum(missing_treatment_list))

        obj_list = delay_list + treatment_list + missing_treatment_list
        self.model.Minimize(cp_model.LinearExpr.Sum(obj_list))

    def slot2time(self, i: int) -> tuple[int, float]:
        # If outside of scheduling slots then return a time that is always unavailable
        if i % self.num_time_slots >= self.num_time_slots / 2:
            return (i // self.num_time_slots, self.instance.workday_end.hour)

        return (
            i // self.num_time_slots,
            (i % self.num_time_slots) * self.instance.time_slot_length.hours
            + self.instance.workday_start.hour,
        )

    def time2slot(self, t: tuple[int, float]) -> float:
        return (
            t[0] * self.num_time_slots
            + (t[1] - self.instance.workday_start.hour)
            / self.instance.time_slot_length.hours
        )

    def _create_constraints(self):
        self.treatment_vars = defaultdict(dict)
        # Add variables for all the treatments to schedule
        for m, rep_range in self.I_m.items():
            for r in rep_range:
                duration = self.du_m[m]

                start_slot = self.model.new_int_var(
                    0,
                    self.num_time_slots * len(self.D) - 1,
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
                        == self.n_fhatm[fhat, m] * is_treatment_scheduled
                    )

                # Add variables that store the day of the treatment # TODO: ugly
                # day_slot = self.model.new_int_var(
                #     0, len(self.D) - 1, f"start_day_m{m.id}_r{r}"
                # )
                # self.model.add_division_equality(
                #     day_slot, start_slot, self.num_time_slots
                # )

                self.treatment_vars[m][r] = {
                    "interval": interval,
                    "start_slot": start_slot,
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
                reps_dict = defaultdict(dict)

                for reps, var_dict in self.treatment_vars[m].items():
                    patient2treatment = self.model.new_bool_var(
                        f"patient2treatment_p{p.id}_m{m.id}_r{reps}",
                    )
                    reps_dict[reps]["patient2treatment"] = patient2treatment

                    # CONSTRAINT: patient can only have treatment if it is scheduled
                    self.model.add_implication(
                        patient2treatment, var_dict["is_present"]
                    )

                    interval_start_slot = var_dict["start_slot"]
                    # Create interval variable with length with rest time
                    interval = self.model.new_optional_fixed_size_interval_var(
                        start=interval_start_slot,
                        size=duration,
                        is_present=patient2treatment,
                        name=f"interval_p{p.id}_m{m.id}_r{reps}",
                    )
                    reps_dict[reps]["interval"] = interval

                self.patient_vars[p][m] = reps_dict

        # CONSTRAINT P1: Patient has lr_pm treatments scheduled
        for p in self.P:
            for m in self.M_p[p]:
                patient_vars = self.patient_vars[p][m]

                self.model.add(
                    cp_model.LinearExpr.Sum(
                        [var["patient2treatment"] for var in patient_vars.values()]
                    )
                    <= self.lr_pm[p, m]
                )

        # CONSTRAINT A1: A treatment is provided for at most k_m patients
        for m in self.M:
            # loop over all patients that could attend this treatment

            for rep in self.I_m[m]:
                patients_list = []
                for p in self.P:
                    if m not in self.M_p[p]:
                        continue

                    patients_list.append(
                        self.patient_vars[p][m][rep]["patient2treatment"]
                    )

                self.model.add(cp_model.LinearExpr.Sum(patients_list) <= self.k_m[m])

        # CONSTRAINT P2: every patient has only a single treatment at a time
        for p, ptreatment_vars in self.patient_vars.items():
            all_intervals = []
            for m, vars in ptreatment_vars.items():
                all_intervals.extend([var["interval"] for var in vars.values()])

            self.model.add_no_overlap(all_intervals)

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
        for p, ptreatment_vars in self.patient_vars.items():
            admission_day = self.admission_vars[p]
            for m, vars in ptreatment_vars.items():
                for rep in vars.keys():
                    interval = vars[rep]["interval"]
                    patient2treatment = vars[rep]["patient2treatment"]
                    # Treatment must start after admission
                    self.model.add(
                        interval.start_expr() >= admission_day * self.num_time_slots
                    ).only_enforce_if(patient2treatment)

                    # Treatment must end before admission end
                    self.model.add(
                        interval.start_expr()
                        < (admission_day + self.l_p[p]) * self.num_time_slots
                    ).only_enforce_if(patient2treatment)

        # Bed capacity constraints via reservoir constraint
        intervals = []
        for p in self.P:
            admission_day = self.admission_vars[p]
            interval = self.model.new_fixed_size_interval_var(
                admission_day, self.l_p[p], f"bed_interval_p{p.id}"
            )
            intervals.append(interval)
        self.model.add_cumulative(
            intervals=intervals,
            demands=[1] * len(intervals),
            capacity=self.instance.beds_capacity,
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

                    intervals_using_f.append(interval_f)

            # CONSTRAINT R3: Resource availability constraints
            length = 0
            for d in self.D:
                for t_index in range(self.num_time_slots):
                    t = self.T[t_index] if t_index < len(self.T) else self.T[-1]
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

            if length != 0:
                # Handle this edge case where the last time slot is unavailable, this is always the case
                start_point = self.time2slot((self.D[-1], self.T[-1]))
                interval_f = self.model.new_fixed_size_interval_var(
                    int(start_point),
                    length,
                    f"interval_f{f.id}_d{d}_t{t}",
                )
                intervals_using_f.append(interval_f)
                length = -1
            # No overlap on resource f
            self.model.add_no_overlap(intervals_using_f)

        # Even distribution of treatments
        if self.max_repr == "reservoir":  # type:ignore
            for p in self.P:
                # Daily even scheduling
                time_slots = []
                demands = []
                active_treatments = []
                for m in self.M_p[p]:
                    for rep in self.I_m[m]:
                        time_slots.append(
                            self.patient_vars[p][m][rep]["interval"].start_expr()
                        )
                        demands.append(1)

                        active_treatments.append(
                            self.patient_vars[p][m][rep]["patient2treatment"]
                        )

                        time_slots.append(
                            self.patient_vars[p][m][rep]["interval"].start_expr()
                            + int(self.num_time_slots / 2)
                        )
                        demands.append(-1)
                        active_treatments.append(
                            self.patient_vars[p][m][rep]["patient2treatment"]
                        )
                self.model.add_reservoir_constraint_with_active(
                    time_slots,
                    demands,
                    active_treatments,
                    min_level=0,
                    max_level=self.daily_upper[p],
                )
                # Even window scheduling not needed when length of stay is too short
                if p.length_of_stay <= self.e_w:
                    continue
                # Create new intervals for e_w scheduling
                time_slots = []
                demands = []
                active_treatments = []

                for m in self.M_p[p]:
                    for rep in self.I_m[m]:
                        time_slots.append(
                            self.patient_vars[p][m][rep]["interval"].start_expr()
                        )
                        demands.append(1)

                        active_treatments.append(
                            self.patient_vars[p][m][rep]["patient2treatment"]
                        )

                        time_slots.append(
                            self.patient_vars[p][m][rep]["interval"].start_expr()
                            + (
                                (self.e_w - 1) * self.num_time_slots
                                + int(self.num_time_slots / 2)
                            )
                        )
                        demands.append(-1)
                        active_treatments.append(
                            self.patient_vars[p][m][rep]["patient2treatment"]
                        )

                self.model.add_reservoir_constraint_with_active(
                    time_slots,
                    demands,
                    active_treatments,
                    min_level=0,
                    max_level=self.e_w_upper[p],
                )

        elif self.max_repr == "cumulative":  # type:ignore
            for p in self.P:
                # Handle the even window scheduling
                intervals = []
                for m in self.M_p[p]:
                    for rep in self.I_m[m]:
                        new_int_var = self.model.new_optional_fixed_size_interval_var(
                            start=self.patient_vars[p][m][rep]["interval"].start_expr(),
                            size=(self.e_w - 1) * self.num_time_slots
                            + int(self.num_time_slots / 2),
                            is_present=self.patient_vars[p][m][rep][
                                "patient2treatment"
                            ],
                            name=f"max_treatment_int_p{p.id}_m{m.id}",
                        )
                        intervals.append(new_int_var)

                self.model.add_cumulative(
                    intervals=intervals,
                    demands=[1] * len(intervals),
                    capacity=self.e_w_upper[p],
                )
                # Handle the daily even scheduling
                intervals = []
                for m in self.M_p[p]:
                    for rep in self.I_m[m]:
                        new_int_var = self.model.new_optional_fixed_size_interval_var(
                            start=self.patient_vars[p][m][rep]["interval"].start_expr(),
                            size=int(self.num_time_slots / 2),
                            name=f"max_treatment_int_p{p.id}_m{m.id}",
                            is_present=self.patient_vars[p][m][rep][
                                "patient2treatment"
                            ],
                        )
                        intervals.append(new_int_var)

                self.model.add_cumulative(
                    intervals=intervals,
                    demands=[1] * len(intervals),
                    capacity=self.daily_upper[p],
                )

        else:
            logger.warning(
                f"Invalid max_repr value: {self.max_repr}"  # type:ignore
            )  # type:ignore
        if self.enforce_min_treatments_per_day:  # type:ignore

            if self.min_repr == "reservoir":  # type:ignore
                for p in self.P:
                    # Create new intervals
                    time_slots = [
                        self.admission_vars[p] * self.num_time_slots
                        + int((self.e_w - 0.5) * self.num_time_slots - 1),
                        (self.admission_vars[p] * self.num_time_slots)
                        + int(self.num_time_slots * (self.l_p[p] - 0.5)),
                    ]
                    demands = [-self.e_w_lower[p], self.e_w_lower[p]]
                    active_treatments = [True, True]
                    for m in self.M_p[p]:
                        for rep in self.I_m[m]:
                            time_slots.append(
                                self.patient_vars[p][m][rep]["interval"].start_expr()
                            )
                            demands.append(1)
                            active_treatments.append(
                                self.patient_vars[p][m][rep]["patient2treatment"]
                            )

                            time_slots.append(
                                self.patient_vars[p][m][rep]["interval"].start_expr()
                                + (
                                    (self.e_w - 1) * self.num_time_slots
                                    + int(self.num_time_slots / 2)
                                )
                            )
                            demands.append(-1)
                            active_treatments.append(
                                self.patient_vars[p][m][rep]["patient2treatment"]
                            )

                    self.model.add_reservoir_constraint_with_active(
                        time_slots,
                        demands,
                        active_treatments,
                        min_level=0,
                        max_level=100,
                    )
                pass
            elif self.min_repr == "cumulative":  # type:ignore
                for p in self.P:
                    start_slot = 0
                    end_slot = (len(self.D) + 2) * self.num_time_slots
                    # Create new intervals
                    intervals = [
                        self.model.new_fixed_size_interval_var(
                            start=self.admission_vars[p] * self.num_time_slots
                            + int(self.num_time_slots / 2 - 1),
                            size=int(self.l_p[p] * self.num_time_slots + 1),
                            name=f"min_treatment_int_dummy_p{p.id}",
                        )
                    ]
                    demands = [self.daily_lower[p]]

                    # demands = []
                    for m in self.M_p[p]:

                        for rep in self.I_m[m]:
                            new_int_var_start = self.model.new_interval_var(
                                start=start_slot,
                                size=self.patient_vars[p][m][rep][
                                    "interval"
                                ].start_expr(),
                                end=self.patient_vars[p][m][rep][
                                    "interval"
                                ].start_expr(),
                                name=f"min_treatment_int_start_p{p.id}_m{m.id}",
                            )

                            new_int_var_middle = (
                                self.model.new_optional_fixed_size_interval_var(
                                    start=self.patient_vars[p][m][rep][
                                        "interval"
                                    ].start_expr(),
                                    size=int(self.num_time_slots * 1.5),
                                    is_present=self.patient_vars[p][m][rep][
                                        "patient2treatment"
                                    ].Not(),
                                    name=f"min_treatment_int_middle_p{p.id}_m{m.id}",
                                )
                            )

                            end_length = self.model.new_int_var(
                                0,
                                (len(self.D) + 2) * self.num_time_slots,
                                f"min_cumulative_end_p{p.id}_m{m.id}",
                            )

                            new_int_var_end = self.model.new_interval_var(
                                start=self.patient_vars[p][m][rep][
                                    "interval"
                                ].start_expr()
                                + (int(self.num_time_slots * 1.5)),
                                size=end_length,
                                end=end_slot,
                                name=f"min_treatment_int_end_p{p.id}_m{m.id}",
                            )

                            intervals.append(new_int_var_start)
                            intervals.append(new_int_var_middle)
                            intervals.append(new_int_var_end)

                    demands += [1] * (len(intervals) - len(demands))
                    total_treatments_p = sum(self.lr_pm[p, m] for m in self.M_p[p])
                    self.model.add_cumulative(
                        intervals=intervals,
                        demands=demands,
                        capacity=total_treatments_p,
                    )
            elif self.min_repr == "day-variables":  # type:ignore
                pass
            else:
                logger.warning(
                    f"Invalid min_repr value: {self.min_repr}"  # type:ignore
                )  #

    def _extract_solution(self, solver):
        appointments_dict = {}

        for m, var_dict in self.treatment_vars.items():
            for rep, vars in var_dict.items():
                if solver.value(vars["is_present"]) == 0:
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
                for rep in self.I_m[m]:
                    if solver.value(vars[rep]["patient2treatment"]):
                        appointments_dict[(m, rep)]["patients"].append(p)

        appointments = []
        test = [(str(r), len(d["patients"])) for (m, r), d in appointments_dict.items()]

        for key in appointments_dict:
            value = appointments_dict[key]
            appointment_parameter = value["appointment_parameter"]
            patients = value["patients"]
            # logger.debug(patients)
            # logger.debug(appointment_parameter)
            # logger.debug(key)
            # if len(patients) == 0:
            #    logger.warning("Treatment scheduled without patients. ")
            #    continue
            appointments.append(Appointment(patients=patients, **appointment_parameter))
        patients_arrival = {}
        for p in self.P:
            admission_day = self.admission_vars[p]
            admission_day_value = solver.value(admission_day)
            patients_arrival[p] = DayHour(
                day=admission_day_value, hour=int(self.instance.workday_start.hour)
            )

        for m, mvars in self.treatment_vars.items():
            break
            for r, vars in mvars.items():
                if not solver.value(vars["is_present"]):
                    pass
                    # continue
                logger.info("Treatment")
                logger.debug(f"{m}: {r}")
                logger.debug(f"start:{solver.value(vars["interval"].start_expr())}")
                logger.debug(f"is_present: {solver.value(vars["is_present"])}")
                for fhat, resources_var in vars["resources"].items():
                    for f, resource_var in resources_var.items():
                        logger.debug(f"resource:{f}")
                        logger.debug(f"r2t: {solver.value(resource_var)}")
                for p in self.patient_vars:

                    logger.debug("Patient")
                    logger.debug(f"p{p}")
                    logger.debug(
                        f"p2t: {solver.value(self.patient_vars[p][m][r]['patient2treatment'])}"
                    )

        solution = Solution(
            instance=self.instance,
            schedule=appointments,
            patients_arrival=patients_arrival,
            solver=self,
            solution_value=solver.objective_value,
        )
        return solution

    def _assert_patients_arrival_day(self, patient: Patient, day: int):
        self.model.add(self.admission_vars[patient] == day)

    def _assert_schedule(self, schedule: list[Appointment]):
        schedule.sort(key=lambda x: x.start_date)
        for app in schedule:
            self._assert_appointment(app)

    def _assert_appointment(self, appointment: Appointment):

        if not hasattr(self, "patient_rep"):
            self.patient_rep = defaultdict(int)
            self.treatment_rep = defaultdict(int)  #

        m = appointment.treatment
        resources = appointment.resources
        d = appointment.start_date.day

        r = self.treatment_rep[m]
        self.treatment_rep[m] += 1

        for p in appointment.patients:
            pr = self.patient_rep[p, m]
            self.patient_rep[p] += 1
            self.model.add(
                self.patient_vars[p][m]["patient_treatment_assignment"][r] == 1
            )

        self.model.add(
            self.treatment_vars[m][r]["start_slot"]
            == int(self.time2slot((d, appointment.start_date.hour)))
        )

        for fhat, fs in resources.items():
            for f in fs:
                self.model.add(self.treatment_vars[m][r]["resources"][fhat][f] == 1)
