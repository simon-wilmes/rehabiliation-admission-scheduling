from src.solvers.solvers import Solver

from ortools.sat.python import cp_model
from datetime import timedelta
from math import ceil
from collections import defaultdict
import logging
from src.instance import Instance

logger = logging.getLogger(__name__)


class CPSolver(Solver):
    def __init__(self, instance: Instance):
        self.instance = instance
        self.model = cp_model.CpModel()

    def create_model(self):

        self._create_model()
        treatment_vars, patient_vars = self.vars
        self._set_optimization_goal(treatment_vars, patient_vars)

    def solve_model(self):
        solver = cp_model.CpSolver()
        solver.parameters.num_search_workers = 12  # Set the number of threads
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

        vars = self._create_variables()
        self.vars = vars
        self._create_constraints(*vars)

    def _create_parameter_sets(self):
        # Define any additional sets or mappings needed for the model besides the ones created in the parent class

        super()._create_parameter_sets()

        self.num_time_slots = len(self.T) + 1

    def _set_optimization_goal(self, treatment_vars, patient_vars):
        # Objective: Maximize the total number of scheduled treatments, possibly weighted
        total_treatments = []
        for vars_list in treatment_vars.values():
            for var in vars_list:
                total_treatments.append(var["interval"].PresenceLiteral())
        self.model.Maximize(cp_model.LinearExpr.Sum(total_treatments))

    def _create_variables(self):
        # Variables for patient admission dates
        patient_vars = {}
        for p in self.P:
            # Admission date variable
            admission_day = self.model.new_int_var(
                p.earliest_admission_date.day,
                p.admitted_before_date.day,
                f"admission_day_p{p.id}",
            )
            patient_vars[p] = admission_day

        # Variables for treatments and resources
        treatment_vars = defaultdict(list)
        for p in self.P:
            for m in self.M_p[p]:
                num_reps = self.r_pm[(p, m)]
                for r in range(num_reps):
                    # Create interger variable for admission
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
                        for f in self.fhat[fhat]:
                            use_resource = self.model.new_bool_var(
                                f"use_resource{p.id}_m{m.id}_r{r}_f{f.id}",
                            )
                            resource_vars[f] = use_resource

                    treatment_vars[(p, m, r)] = {
                        "interval": interval,
                        "start_slot": start_slot,
                        "resources": resource_vars,
                    }

        return treatment_vars, patient_vars

    def _create_constraints(self, treatment_vars, patient_admission_vars):
        # Admission constraints
        for p in self.P:
            admission_day = patient_admission_vars[p]
            # If the patient is already admitted, fix the admission day
            if p.already_admitted:
                self.model.add(admission_day == p.earliest_admission_date.day)

        # Ensure that admssion is in the correct range
        for p in self.P:
            self.model.add(admission_day >= p.earliest_admission_date.day)
            self.model.add(admission_day <= p.admitted_before_date.day)

        # Ensure treatments are scheduled within admission period
        for (p, m, r), vars in treatment_vars.items():
            interval, start_slot, resources = vars.values()
            admission_day = patient_admission_vars[p]

            # Treatment can only start after admission
            self.model.add(
                start_slot >= admission_day * self.num_time_slots,
            )

            # Treatment must end before admission end
            self.model.add(
                start_slot <= (admission_day + self.l_p[p] - 1) * 37,
            )

        # Patient can have only one treatment at a time
        for p in self.P:
            intervals = []
            for m in self.M_p[p]:
                for r in range(self.r_pm[(p, m)]):
                    intervals.append(treatment_vars[(p, m, r)]["interval"])
            self.model.add_no_overlap(intervals)

        # Bed capacity constraints via reservoir constraint
        bed_changes_time = []
        bed_changes_amount = []
        for p in self.P:
            admission_day = patient_admission_vars[p]
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

            for (p, m, r), vars in treatment_vars.items():

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
            # No overlap on resource f
            self.model.add_no_overlap(intervals_using_f)

        # Resource availability constraints
        for f in self.F:
            availability = []
            for d in self.D:
                for t in self.T:
                    if self.av_fdt.get((f, d, t), 0):
                        availability.append((d * 24 + t, 1))

        # Restric resources at the end of the day

        # Even distribution of treatments
        for p in self.P:
            admission_day = patient_admission_vars[p]
            length_of_stay = self.l_p[p]
            num_windows = ceil(length_of_stay / self.rw)
            for m in self.M_p[p]:
                total_reps = self.r_pm[(p, m)]
                for w in range(num_windows):
                    window_start = w * self.rw
                    window_end = min(window_start + self.rw, length_of_stay)
                    window_days = range(window_start, window_end)
                    treatments_in_window = []
                    for rep in treatment_vars[(p, m)]:
                        interval = rep["interval"]
                        start_day = rep["start_day"]
                        in_window = self.model.NewBoolVar(
                            f"in_window_p{p.id}_m{m.id}_w{w}"
                        )
                        self.model.Add(
                            start_day - admission_day >= window_start
                        ).OnlyEnforceIf(in_window)
                        self.model.Add(
                            start_day - admission_day < window_end
                        ).OnlyEnforceIf(in_window)
                        self.model.AddBoolOr(
                            [
                                start_day - admission_day < window_start,
                                start_day - admission_day >= window_end,
                            ]
                        ).OnlyEnforceIf(in_window.Not())
                        self.model.AddImplication(in_window, interval.PresenceLiteral())
                        treatments_in_window.append(in_window)
                    max_in_window = ceil(total_reps / num_windows)
                    self.model.Add(
                        cp_model.LinearExpr.Sum(treatments_in_window) <= max_in_window
                    )

        # Resource loyalty constraints
        for p in self.P:
            for m in self.M_p[p]:
                for fhat in self.Lhat_m.get(m, []):
                    # All repetitions of treatment m for patient p must use the same resource in fhat
                    first_resource_var = None
                    for rep in treatment_vars[(p, m)]:
                        resource_vars = rep["resources"][fhat]
                        for resource_var in resource_vars:
                            if first_resource_var is None:
                                first_resource_var = resource_var
                            else:
                                self.model.Add(resource_var == first_resource_var)

    def _extract_solution(self, solver):
        appointments_dict = defaultdict(list)
        for (p, m), reps in self.vars[0].items():
            for rep in reps:
                interval = rep["interval"]
                if solver.Value(interval.PresenceLiteral()):
                    start_time = solver.Value(interval.StartExpr())
                    start_day = start_time // 24
                    start_hour = start_time % 24
                    # Collect resources assigned
                    resources_used = {}
                    for fhat, resource_vars in rep["resources"].items():
                        resources = self.fhat[fhat]
                        assigned_resources = []
                        for resource_var in resource_vars:
                            resource_idx = solver.Value(resource_var)
                            assigned_resources.append(resources[resource_idx])
                        resources_used[fhat] = assigned_resources
                    # Create appointment key
                    resource_ids = []
                    for res_list in resources_used.values():
                        resource_ids.extend([res.id for res in res_list])
                    resource_ids.sort()
                    appointment_key = (m, start_day, start_hour, tuple(resource_ids))
                    appointments_dict[appointment_key].append((p, resources_used))

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
            admission_day = self.vars[1][p]["admission_day"]
            admission_day_value = solver.Value(admission_day)
            patients_arrival[p] = DayHour(
                day=admission_day_value, hour=int(self.instance.workday_start.hours)
            )

        solution = Solution(
            instance=self.instance,
            schedule=appointments,
            patients_arrival=patients_arrival,
        )
        return solution
