from src.solvers.solvers import Solver

from ortools.sat.python import cp_model
from datetime import timedelta
from math import ceil
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class CPSolver(Solver):
    def __init__(self, instance):
        self.instance = instance
        self.model = cp_model.CpModel()

    def create_model(self):
        self._create_parameter_sets()
        vars = self._create_variables()
        self._create_constraints(*vars)
        self._set_optimization_goal(*vars)
        self.vars = vars

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

    def _set_optimization_goal(self, treatment_vars, patient_vars):
        # Objective: Maximize the total number of scheduled treatments, possibly weighted
        total_treatments = []
        for vars_list in treatment_vars.values():
            for var in vars_list:
                total_treatments.append(var["interval"].PresenceLiteral())
        self.model.Maximize(cp_model.LinearExpr.Sum(total_treatments))

    def _create_parameter_sets(self):
        # Similar to the MIP model, create necessary sets and mappings
        self.max_day = max(
            p.latest_admission_date.day + p.length_of_stay + 1
            for p in self.instance.patients.values()
        )
        self.D = range(self.max_day)
        self.T = list(
            range(
                int(self.instance.workday_start.hours),
                int(self.instance.workday_end.hours),
            )
        )
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
                p.latest_admission_date.day + 1 + p.length_of_stay,
            )
            for p in self.instance.patients.values()
        }
        self.M_p = {
            p: list(p.treatments.keys()) for p in self.instance.patients.values()
        }
        self.k_m = {t: t.num_participants for t in self.instance.treatments.values()}
        self.r_pm = {
            (p, m): int(p.treatments[m] * p.length_of_stay / self.instance.week_length)
            for p in self.instance.patients.values()
            for m in self.M_p[p]
        }
        self.l_p = {p: p.length_of_stay for p in self.instance.patients.values()}
        self.du_m = {
            m: ceil(m.duration / self.instance.time_slot_length)
            for m in self.instance.treatments.values()
        }

        self.b = self.instance.num_beds
        self.rw = self.instance.rolling_window_length

        # Resource availability
        self.av_fdt = {
            (f, d, t): int(f.is_available(d + t))
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

    def _create_variables(self):
        # Variables for patient admission dates
        patient_vars = {}
        for p in self.P:
            # Admission date variable
            admission_day = self.model.new_int_var(
                p.earliest_admission_date.day,
                p.latest_admission_date.day,
                f"admission_day_p{p.id}",
            )
            patient_vars[p] = {
                "admission_day": admission_day,
            }

        # Variables for treatments and resources
        treatment_vars = {}
        for p in self.P:
            for m in self.M_p[p]:
                num_reps = self.r_pm[(p, m)]
                treatment_vars[(p, m)] = []
                for r in range(num_reps):
                    # Create interval variable for each treatment repetition
                    start_day = self.model.new_int_var(
                        p.earliest_admission_date.day,
                        p.latest_admission_date.day + self.l_p[p] - 1,
                        f"start_day_p{p.id}_m{m.id}_r{r}",
                    )

                    # Break symetry by enforcing order amongst repetitions

                    duration = self.du_m[m]
                    # Assuming time slots are hours for simplicity
                    start_time = self.model.new_int_var(
                        int(self.instance.workday_start.hours),
                        int(self.instance.workday_end.hours - duration),
                        f"start_time_p{p.id}_m{m.id}_r{r}",
                    )

                    # Create interval variable
                    interval = self.model.new_fixed_size_interval_var(
                        start_day * 24 + start_time,
                        duration,
                        (start_day + duration // 24) * 24 + start_time + duration % 24,
                        self.model.new_bool_var(f"is_scheduled_p{p.id}_m{m.id}_r{r}"),
                        f"interval_p{p.id}_m{m.id}_r{r}",
                    )
                    # Resource variables
                    resource_vars = {}
                    for fhat in self.Fhat_m[m]:
                        num_resources_needed = self.n_fhatm[(fhat, m)]
                        resources = self.fhat[fhat]
                        resource_assignments = []
                        for n in range(num_resources_needed):
                            resource_var = self.model.new_int_var(
                                0,
                                len(resources) - 1,
                                f"resource_p{p.id}_m{m.id}_r{r}_fhat{fhat.id}_{n}",
                            )
                            resource_assignments.append(resource_var)
                        resource_vars[fhat] = resource_assignments

                    treatment_vars[(p, m, r)].append(
                        {
                            "interval": interval,
                            "start_day": start_day,
                            "start_time": start_time,
                            "resources": resource_vars,
                        }
                    )

        return treatment_vars, patient_vars

    def _create_constraints(self, treatment_vars, patient_vars):
        # Admission constraints
        for p in self.P:
            admission_day = patient_vars[p]["admission_day"]
            # If the patient is already admitted, fix the admission day
            if p.already_admitted:
                self.model.Add(admission_day == p.earliest_admission_date.day)

        # Ensure treatments are scheduled within admission period
        for p in self.P:
            admission_day = patient_vars[p]["admission_day"]
            for m in self.M_p[p]:
                for rep in treatment_vars[(p, m)]:
                    interval = rep["interval"]
                    start_day = rep["start_day"]
                    # Treatment can only start after admission
                    self.model.Add(start_day >= admission_day).OnlyEnforceIf(
                        interval.PresenceLiteral()
                    )
                    # Treatment must end before admission end
                    self.model.Add(
                        start_day <= admission_day + self.l_p[p] - 1
                    ).OnlyEnforceIf(interval.PresenceLiteral())

        # Patient can have only one treatment at a time
        for p in self.P:
            intervals = []
            for m in self.M_p[p]:
                for rep in treatment_vars[(p, m)]:
                    intervals.append(rep["interval"])
            self.model.AddNoOverlap(intervals)

        # Bed capacity constraints
        for d in self.D:
            patients_present = []
            for p in self.P:
                admission_day = patient_vars[p]["admission_day"]
                # Patient is present if admission_day <= d < admission_day + length_of_stay
                is_present = self.model.NewBoolVar(f"is_present_p{p.id}_d{d}")
                self.model.Add(admission_day <= d).OnlyEnforceIf(is_present)
                self.model.Add(admission_day + self.l_p[p] > d).OnlyEnforceIf(
                    is_present
                )
                self.model.AddBoolOr(
                    [admission_day > d, admission_day + self.l_p[p] <= d]
                ).OnlyEnforceIf(is_present.Not())
                patients_present.append(is_present)
            self.model.Add(cp_model.LinearExpr.Sum(patients_present) <= self.b)

        # Resource capacity constraints
        for f in self.F:
            # Collect intervals that use resource f
            intervals_using_f = []
            for p in self.P:
                for m in self.M_p[p]:
                    for rep in treatment_vars[(p, m)]:
                        for fhat in self.Fhat_m[m]:
                            resources = rep["resources"][fhat]
                            resources_list = self.fhat[fhat]
                            for resource_var in resources:
                                # Add constraint to link resource assignment to resource f
                                is_assigned_to_f = self.model.NewBoolVar(
                                    f"is_assigned_p{p.id}_m{m.id}_f{f.id}"
                                )
                                self.model.Add(
                                    resource_var == resources_list.index(f)
                                ).OnlyEnforceIf(is_assigned_to_f)
                                self.model.Add(
                                    resource_var != resources_list.index(f)
                                ).OnlyEnforceIf(is_assigned_to_f.Not())
                                # If treatment is scheduled and resource is assigned to f, add interval
                                interval = rep["interval"]
                                interval_f = self.model.NewOptionalIntervalVar(
                                    interval.StartExpr(),
                                    interval.SizeExpr(),
                                    interval.EndExpr(),
                                    is_assigned_to_f,
                                    f"interval_p{p.id}_m{m.id}_f{f.id}",
                                )
                                intervals_using_f.append(interval_f)
            # No overlap on resource f
            self.model.AddNoOverlap(intervals_using_f)

        # Resource availability constraints
        for f in self.F:
            availability = []
            for d in self.D:
                for t in self.T:
                    if self.av_fdt.get((f, d, t), 0):
                        availability.append((d * 24 + t, 1))
            # Note: In this simplified model, we assume resources are always available when needed
            # For a more detailed model, you can adjust the intervals to match resource availability

        # Conflict group constraints
        for p in self.P:
            for d in self.D:
                conflict_intervals = []
                for c in self.C:
                    treatments_in_c = self.M_c[c]
                    for m in treatments_in_c:
                        if m not in self.M_p[p]:
                            continue
                        for rep in treatment_vars[(p, m)]:
                            interval = rep["interval"]
                            start_day = rep["start_day"]
                            is_on_day_d = self.model.NewBoolVar(
                                f"is_on_day_p{p.id}_m{m.id}_d{d}"
                            )
                            self.model.Add(start_day == d).OnlyEnforceIf(is_on_day_d)
                            self.model.Add(start_day != d).OnlyEnforceIf(
                                is_on_day_d.Not()
                            )
                            self.model.AddImplication(
                                is_on_day_d, interval.PresenceLiteral()
                            )
                            conflict_intervals.append(is_on_day_d)
                self.model.Add(cp_model.LinearExpr.Sum(conflict_intervals) <= 1)

        # Even distribution of treatments
        for p in self.P:
            admission_day = patient_vars[p]["admission_day"]
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


# Auxiliary classes (Assuming these are defined elsewhere in your codebase)
class DayHour:
    def __init__(self, day, hour):
        self.day = day
        self.hour = hour


class Appointment:
    def __init__(self, patients, start_date, treatment, resources):
        self.patients = patients
        self.start_date = start_date
        self.treatment = treatment
        self.resources = resources


class Solution:
    def __init__(self, instance, schedule, patients_arrival):
        self.instance = instance
        self.schedule = schedule
        self.patients_arrival = patients_arrival
