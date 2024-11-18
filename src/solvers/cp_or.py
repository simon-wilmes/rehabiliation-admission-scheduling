from src.solvers.solvers import Solver

from ortools.sat.python import cp_model
from datetime import timedelta
from math import ceil
from collections import defaultdict
import logging
from src.instance import Instance
from src.time import DayHour, Duration
from src.solution import Solution, Appointment

logger = logging.getLogger(__name__)


class CPSolver(Solver):
    def __init__(self, instance: Instance, constraints_ignore: set[str] = set()):
        super().__init__(instance, constraints_ignore)

        self.algo_options = {
            "even-scheduling": ["sum", "reservoir"]
        }
        
    def create_model(self):

        self._create_model()
        # self._set_optimization_goal()

    def solve_model(self):
        solver = cp_model.CpSolver()
        solver.parameters.num_search_workers = (
            self.number_of_threads
        )  # Set the number of threads
        status = solver.Solve(self.model)
        if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
            logger.info("Solution found.")
            solution = self._extract_solution(solver)
            return solution
        else:
            logger.error("No solution found.")
            return None

    def _create_model(self):
        self._create_parameter_sets()

        self.model = cp_model.CpModel()
        self._create_constraints()

    def _create_parameter_sets(self):
        super()._create_parameter_sets()
        # Define any additional sets or mappings needed for the model
        # besides the ones created in the parent class
        self.num_time_slots = len(self.T) + 1

    def _set_optimization_goal(self):
        # Objective: Maximize the total number of scheduled treatments, possibly weighted
        total_treatments = []
        for vars_list in self.treatment_vars.values():
            for var in vars_list:
                total_treatments.append(var["interval"].PresenceLiteral())
        self.model.Maximize(cp_model.LinearExpr.Sum(total_treatments))

    def slot2time(self, i: int) -> tuple[int, float]:
        return (
            i // self.num_time_slots,
            (i % self.num_time_slots) * self.instance.time_slot_length.hours
            + self.instance.workday_start.hour,
        )

    def time2slot(self, t: tuple[int, int]) -> int:
        return t[0] * self.num_time_slots + t[1]

    def _create_constraints(self):

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
        self.treatment_vars = defaultdict(dict)
        for p in self.P:
            for m in self.M_p[p]:
                # Resource variables for loyal resources use the same variable for multiple repetitions thereby enforcing loyalty
                resource_loyal_var = {
                    fg: {
                        f: self.model.new_bool_var(
                            f"use_resource{p.id}_m{m.id}_loyal_f{f.id}",
                        )
                        for f in self.fhat[fg]
                    }
                    for fg in self.Lhat_m[m]
                }
                num_reps = self.lr_pm[(p, m)]
                for r in range(num_reps):
                    # Create integer variable for admission
                    duration = self.du_m[m]

                    start_slot = self.model.new_int_var(
                        0,
                        self.num_time_slots * len(self.D),
                        f"start_time_p{p.id}_m{m.id}_r{r}",
                    )

                    # Create interval variable
                    interval = self.model.new_fixed_size_interval_var(
                        start=start_slot,
                        size=duration,
                        name=f"interval_p{p.id}_m{m.id}_r{r}",
                    )
                    # Resource variables
                    resource_vars = {}
                    for fhat in self.Fhat_m[m]:
                        # If resource is loyal, use the same variable for all repetitions
                        if fhat in self.Lhat_m[m] and self.add_resource_loyal():
                            for f in self.fhat[fhat]:
                                resource_vars[f] = resource_loyal_var[fhat][f]
                        else:
                            for f in self.fhat[fhat]:
                                use_resource = self.model.new_bool_var(
                                    f"use_resource{p.id}_m{m.id}_r{r}_f{f.id}",
                                )
                                resource_vars[f] = use_resource

                    # Enforcing existing loyalty
                    if self.add_resource_loyal():
                        for (m2, fhat), resources in p.already_resource_loyal.items():
                            # Skip if not the same treatment
                            if m is not m2:
                                continue
                            for f in resources:
                                self.model.add(resource_loyal_var[fhat][f] == 1)

                    # Make sure that every treatment has the required resources
                    for fhat in self.Fhat_m[m]:
                        if fhat not in self.Lhat_m[m]:

                            self.model.add(
                                cp_model.LinearExpr.Sum(
                                    [resource_vars[f] for f in self.fhat[fhat]]
                                )
                                == self.n_fhatm[fhat, m]
                            )

                    self.treatment_vars[(p, m, r)] = {
                        "interval": interval,
                        "start_slot": start_slot,
                        "resources": resource_vars,
                    }

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
        for (p, m, r), vars in self.treatment_vars.items():
            interval, start_slot, resources = vars.values()
            admission_day = self.admission_vars[p]

            # Treatment can only start after admission
            self.model.add(
                start_slot >= admission_day * self.num_time_slots,
            )

            # Treatment must end before admission end
            self.model.add(
                start_slot <= (admission_day + self.l_p[p] - 1) * self.num_time_slots,
            )

        # Patient can have only one treatment at a time
        for p in self.P:
            intervals = []
            for m in self.M_p[p]:
                for r in range(self.lr_pm[(p, m)]):
                    intervals.append(self.treatment_vars[(p, m, r)]["interval"])
            self.model.add_no_overlap(intervals)

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
            # Collect intervals for f
            intervals_using_f = []

            for (p, m, r), vars in self.treatment_vars.items():

                # Skip treatments that do not use resource f
                if f.resource_group not in self.Fhat_m[m]:
                    continue

                interval, start_slot, resources = vars.values()

                interval_f = self.model.new_optional_fixed_size_interval_var(
                    interval.start_expr(),
                    self.du_m[m],
                    resources[f],
                    f"interval_p{p.id}_m{m.id}_f{f.id}_r{r}",
                )
                intervals_using_f.append(interval_f)

            # Resource availability constraints
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

            if length != 0:
                start_point = self.time2slot((d, t)) - length
                interval_f = self.model.new_fixed_size_interval_var(
                    int(start_point),
                    length,
                    f"interval_f{f.id}_d{d}_t{t}",
                )
                intervals_using_f.append(interval_f)
                length = -1
            # No overlap on resource f
            self.model.add_no_overlap(intervals_using_f)

        for f in self.F:
            availability = []
            for d in self.D:
                for t in self.T:
                    if self.av_fdt.get((f, d, t), 0):
                        availability.append((d * 24 + t, 1))

        # Even distribution of treatments
        if self.add_even_distribution():
            logger.debug("Even distribution constraint not implemented for CP solver.")

        # Conflict groups
        if self.add_conflict_groups():
            for p in self.P:
                for conflict_group in self.instance.conflict_groups:
                    conflicting_vars = []
                    for (p2, m, r), vars in self.treatment_vars.items():
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
        appointments_dict = defaultdict(list)
        for (p, m, rep), var_dict in self.treatment_vars.items():
            interval = var_dict["interval"]

            start_slot = solver.value(interval.start_expr())
            start_day, start_time = self.slot2time(start_slot)

            # Collect resources assigned
            assigned_resources = defaultdict(list)
            for resource, resource_var in var_dict["resources"].items():
                if solver.value(resource_var):
                    assigned_resources[resource.resource_group].append(resource)

            # Create appointment key
            resource_ids = []
            for res_list in assigned_resources.values():
                resource_ids.extend([res.id for res in res_list])

            appointment_key = (m, start_day, start_time, frozenset(resource_ids))

            appointments_dict[appointment_key].append((p, assigned_resources))

        appointments = []
        for (m, d, t, resource_ids), patient_info_list in appointments_dict.items():
            patients = []
            resources = {}
            for p, resources_used in patient_info_list:
                patients.append(p)
                resources = (
                    resources_used  # Assuming resources are the same for all patients
                )
            start_date = DayHour(day=d, hour=t)
            appointment = Appointment(
                patients=patients,
                start_date=start_date,
                treatment=m,
                resources=resources,
            )
            appointments.append(appointment)

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
        )
        return solution
