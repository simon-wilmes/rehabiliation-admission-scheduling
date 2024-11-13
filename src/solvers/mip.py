from src.instance import Instance
import gurobipy as gp
from src.time import Duration
from math import ceil
from numpy import arange
from src.logging import logger, print
from itertools import product
from src.solution import Appointment, Solution
from src.time import DayHour
from src.patients import Patient


class MIPSolver:
    def __init__(self, instance: Instance):
        self.instance = instance

    def create_model(self):
        self._create_parameter_sets()
        # Create the model
        model = gp.Model("PatientAdmissionScheduling")
        vars = self._create_variables(model)
        self._create_constraints(model, *vars)
        self._set_optimization_goal(model, *vars)

        self.vars = vars
        self.model = model

    def solve_model(self):
        self.model.optimize()
        if self.model.status == gp.GRB.OPTIMAL:
            logger.info("Optimal solution found.")
            solution = self._extract_solution()

        else:
            logger.error("No optimal solution found.")
            return

    def _set_optimization_goal(
        self, model: gp.Model, x_pmdt, z_pmfdt, u_mfdt, v_pmf, a_pd
    ):
        objective = gp.quicksum(
            (d - p.earliest_admission_date.day) * a_pd[p, d]
            for p in self.P
            for d in self.D_p[p]
        )
        model.setObjective(objective, gp.GRB.MINIMIZE)

    def _create_parameter_sets(self):
        max_day = max(
            p.latest_admission_date.day + p.length_of_stay + 1
            for p in self.instance.patients.values()
        )
        self.D = range(max_day)
        self.T = arange(
            self.instance.workday_start.hours,
            self.instance.workday_end.hours,
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
                p.latest_admission_date.day + 1 + p.length_of_stay,
            )
            for p in self.instance.patients.values()
        }
        self.M_p = {
            p: list(p.treatments.keys()) for p in self.instance.patients.values()
        }
        self.k_m = {t: t.num_participants for t in self.instance.treatments.values()}
        self.r_pm = {
            (p, m): p.treatments[m]
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

    def _create_variables(self, model: gp.Model):
        #####################################
        # Create variables
        #####################################
        # Create x_pmdt
        x_pmdt_keys = []
        for p in self.P:
            x_pmdt_keys.extend(product([p], self.M_p[p], self.D_p[p], self.T))

        x_pmdt = model.addVars(x_pmdt_keys, vtype=gp.GRB.BINARY, name="x_pmdt")
        # Create z_pmfdt
        z_pmfdt_keys = []
        for p in self.P:
            for m in self.M_p[p]:
                for fhat in self.Fhat_m[m]:
                    z_pmfdt_keys.extend(
                        product([p], [m], self.fhat[fhat], self.D_p[p], self.T)
                    )

        z_pmfdt = model.addVars(z_pmfdt_keys, vtype=gp.GRB.BINARY, name="z_pmfdt")

        # Create u_mfdt
        u_mfdt_keys = []
        for m in self.M:
            for fhat in self.Fhat_m[m]:
                u_mfdt_keys.extend(product([m], self.fhat[fhat], self.D, self.T))
        u_mfdt = model.addVars(u_mfdt_keys, vtype=gp.GRB.BINARY, name="u_mfdt")

        # Create v_pmf
        v_pmf_keys = []
        for p in self.P:
            for m in self.M_p[p]:
                for fhat in self.Fhat_m[m]:
                    v_pmf_keys.extend(product([p], [m], self.fhat[fhat]))
        v_pmf = model.addVars(v_pmf_keys, vtype=gp.GRB.BINARY, name="v_pmf")

        # Create a_pd
        a_pd_keys = []
        for p in self.P:
            a_pd_keys.extend(product([p], self.D_p[p]))
        a_pd = model.addVars(a_pd_keys, vtype=gp.GRB.BINARY, name="a_pd")

        self.vars = {
            "x_pmdt": x_pmdt,
            "z_pmfdt": z_pmfdt,
            "u_mfdt": u_mfdt,
            "v_pmf": v_pmf,
            "a_pd": a_pd,
        }
        return x_pmdt, z_pmfdt, u_mfdt, v_pmf, a_pd

    def _create_constraints(
        self, model: gp.Model, x_pmdt, z_pmfdt, u_mfdt, v_pmf, a_pd
    ):
        #####################################
        # Create constraints
        #####################################

        # Constraint (p1): Sum over all scheduled treatments equals total repetitions left
        for p in self.P:
            for m in self.M_p[p]:
                model.addConstr(
                    gp.quicksum(x_pmdt[p, m, d, t] for d in self.D_p[p] for t in self.T)
                    == self.r_pm[p, m],
                    name=f"constraint_p1_p{p.id}_m{m.id}",
                )

        # Constraint (p2): Patients not admitted have no treatments scheduled
        for p in self.P:
            for m in self.M_p[p]:
                for d in self.D_p[p]:
                    for t in self.T:
                        delta_set = [
                            delta
                            for delta in self.D_p[p]
                            if d - self.l_p[p] < delta <= d
                        ]
                        model.addConstr(
                            x_pmdt[p, m, d, t]
                            <= gp.quicksum(a_pd[p, delta] for delta in delta_set),
                            name=f"constraint_p2_p{p.id}_m{m.id}_d{d}_t{t}",
                        )

        # Constraint (p3): Only one treatment at a time per patient
        for p in self.P:
            for d in self.D_p[p]:
                for t in self.T:
                    expr = gp.LinExpr()
                    for m in self.M_p[p]:
                        tau_set = [tau for tau in self.T if t - self.du_m[m] < tau <= t]
                        expr += gp.quicksum(x_pmdt[p, m, d, tau] for tau in tau_set)
                    model.addConstr(expr <= 1, name=f"constraint_p3_p{p.id}_d{d}_t{t}")

        # Constraint (p4): Total admitted patients cannot exceed total beds
        for d in self.D:
            expr = gp.LinExpr()
            for p in self.P:
                delta_set = [
                    delta for delta in self.D_p[p] if d - self.l_p[p] < delta <= d
                ]
                expr += gp.quicksum(a_pd[p, delta] for delta in delta_set)
            model.addConstr(expr <= self.b, name=f"constraint_p4_d{d}")

        # Constraint (p5): Stress limit per patient per day
        """for p in self.P:
            for d in self.D_p[p]:
                expr = gp.LinExpr()
                for m in self.M_p[p]:
                    b_m_m = self.b_m[m]
                    expr += b_m_m * gp.quicksum(x_pmdt[p, m, d, t] for t in self.T)
                model.addConstr(expr <= self.s_p[p], name=f"constraint_p5_p{p.id}_d{d}")
        """
        # Constraint (p6): Patient admitted exactly once within the specified time
        for p in self.P:
            D_p_early = [d for d in self.D_p[p] if d <= max(self.D_p[p]) - self.l_p[p]]
            model.addConstr(
                gp.quicksum(a_pd[p, d] for d in D_p_early) == 1,
                name=f"constraint_p6_p{p.id}",
            )

        # Constraint (p7): Patient is admitted exactly once
        for p in self.P:
            model.addConstr(
                gp.quicksum(a_pd[p, d] for d in self.D_p[p]) == 1,
                name=f"constraint_p7_p{p.id}",
            )

        # Constraint (r2): Resource availability and utilization
        for fhat in self.Fhat:
            for f in self.fhat[fhat]:
                for d in self.D:
                    for t in self.T:
                        expr = gp.LinExpr()
                        for m in self.M_fhat[fhat]:
                            du_m_m = self.du_m[m]
                            tau_set = [tau for tau in self.T if t - du_m_m < tau <= t]
                            expr += gp.quicksum(
                                u_mfdt[m, f, d, tau]
                                for tau in tau_set
                                if (m, f, d, tau) in u_mfdt
                            )
                        model.addConstr(
                            expr <= self.av_fdt[f, d, t],
                            name=f"constraint_r2_fhat{fhat.id}_f{f.id}_d{d}_t{t}",
                        )

        # Constraint (r3): Linking z and u variables
        for p in self.P:
            for m in self.M_p[p]:
                for fhat in self.Fhat_m[m]:
                    for f in self.fhat[fhat]:
                        for d in self.D_p[p]:
                            for t in self.T:
                                model.addConstr(
                                    z_pmfdt[p, m, f, d, t] <= u_mfdt[m, f, d, t],
                                    name=f"constraint_r3_p{p.id}_m{m.id}_f{f.id}_d{d}_t{t}",
                                )

        # Constraint (r4): Assign required number of resources for each treatment
        for p in self.P:
            for m in self.M_p[p]:
                for fhat in self.Fhat_m[m]:
                    for d in self.D_p[p]:
                        for t in self.T:
                            lhs = gp.quicksum(
                                z_pmfdt[p, m, f, d, t] for f in self.fhat[fhat]
                            )
                            rhs = self.n_fhatm[fhat, m] * x_pmdt[p, m, d, t]
                            model.addConstr(
                                lhs == rhs,
                                name=f"constraint_r4_p{p.id}_m{m.id}_fhat{fhat.id}_d{d}_t{t}",
                            )

        # Constraint (a1): Limit the number of patients per treatment
        for m in self.M:
            for fhat in self.Fhat_m[m]:
                for f in self.fhat[fhat]:
                    for d in self.D:
                        for t in self.T:
                            lhs = gp.quicksum(
                                z_pmfdt[p, m, f, d, t]
                                for p in self.P
                                if (p, m, f, d, t) in z_pmfdt
                            )
                            model.addConstr(
                                lhs <= self.k_m[m],
                                name=f"constraint_a1_m{m.id}_f{f.id}_d{d}_t{t}",
                            )

        # Constraint (a2-help): Resource loyalty linking
        for p in self.P:
            for m in self.M_p[p]:
                for fhat in self.Fhat_m[m]:
                    for f in self.fhat[fhat]:
                        for d in self.D_p[p]:
                            for t in self.T:
                                model.addConstr(
                                    z_pmfdt[p, m, f, d, t] <= v_pmf[p, m, f],
                                    name=f"constraint_a2_help_p{p.id}_m{m.id}_f{f.id}_d{d}_t{t}",
                                )

        # Constraint (a2): Resource loyalty constraint
        for p in self.P:
            for m in self.M_p[p]:
                for fhat in self.Lhat_m[m]:
                    model.addConstr(
                        gp.quicksum(v_pmf[p, m, f] for f in self.fhat[fhat])
                        <= self.n_fhatm[fhat, m],
                        name=f"constraint_a2_p{p.id}_m{m.id}_fhat{fhat.id}",
                    )

        # Constraint (a3): Conflict groups (if applicable)
        if hasattr(self, "M_c") and hasattr(self, "C"):
            for p in self.P:
                for d in self.D_p[p]:
                    for c in self.C:
                        model.addConstr(
                            gp.quicksum(
                                x_pmdt[p, m, d, t]
                                for m in self.M_c[c]
                                for t in self.T
                                if (p, m, d, t) in x_pmdt
                            )
                            <= 1,
                            name=f"constraint_a3_p{p.id}_d{d}_c{c}",
                        )

        # Constraint (a4): Even distribution of treatments over rolling windows
        if hasattr(self, "rw"):
            for p in self.P:
                for m in self.M_p[p]:
                    num_windows = ceil(len(self.D_p[p]) / self.rw)
                    for w in range(num_windows):
                        window_start = self.D_p[p][0] + w * self.rw
                        window_end = min(window_start + self.rw - 1, self.D_p[p][-1])
                        window_days = [
                            d for d in self.D_p[p] if window_start <= d <= window_end
                        ]
                        expr = gp.quicksum(
                            x_pmdt[p, m, d, t]
                            for d in window_days
                            for t in self.T
                            if (p, m, d, t) in x_pmdt
                        )
                        model.addConstr(
                            expr <= ceil(self.r_pm[p, m] / num_windows),
                            name=f"constraint_a4_p{p.id}_m{m.id}_w{w}",
                        )

        # Constraint: Make sure that already admitted patients are admitted on the first day of the planning horizon
        for p in self.P:
            if p.already_admitted:
                model.addConstr(
                    a_pd[p, 0] == 1, name=f"constraint_already_admitted_p{p.id}"
                )

        # Constraint: Make sure that already_resource_loyal is respected
        # Constraint: Ensure already resource loyal assignments are set to 1
        for p in self.P:
            if hasattr(p, "already_resource_loyal"):
                for (m, fhat), resources in p.already_resource_loyal.items():
                    for f in resources:
                        model.addConstr(
                            v_pmf[p, m, f] == 1,
                            name=f"constraint_loyal_resource_p{p.id}_m{m.id}_f{f.id}",
                        )

    def _extract_solution(self):
        """
        Extracts the solution from the MIP model and constructs a Solution object.

        Returns:
            Solution: The constructed solution with appointments.
        """
        from collections import defaultdict

        x_pmdt, z_pmfdt, u_mfdt, v_pmf, a_pd = self.vars

        # appointments_dict: key=(m, d, t, frozenset of resource IDs), value=list of patients
        appointments_dict = defaultdict(list)

        # Collect scheduled treatments and group patients based on resources used
        for (p, m, d, t), var in x_pmdt.items():
            if var.X > 0.5:
                # Determine the resources used by patient p for treatment m at (d, t)
                resources_used = defaultdict(
                    list
                )  # dict[ResourceGroup, list[Resource]]
                for fhat in m.resources.keys():
                    # Get the required number of resources for this resource group
                    required_amount = m.resources[fhat]

                    # Find resources assigned to patient p for this treatment at (d, t)
                    resources_in_group = []
                    for f in self.instance.resources.values():
                        if f.resource_group != fhat:
                            continue
                        z_key = (p, m, f, d, t)
                        if z_key in z_pmfdt and z_pmfdt[z_key].X > 0.5:
                            resources_in_group.append(f)

                    # Sort resources to ensure consistent ordering
                    resources_in_group.sort(key=lambda res: res.id)

                    # Check if the required number of resources are assigned
                    if len(resources_in_group) != required_amount:
                        print(
                            f"Warning: Patient {p.id} expected {required_amount} resources from {fhat.name} "
                            f"but found {len(resources_in_group)} for treatment {m.id} at ({d}, {t})."
                        )

                    resources_used[fhat] = resources_in_group

                # Create a key that uniquely identifies the appointment, including resources
                # We'll use the resource IDs to create a frozenset for the key
                resource_ids = []
                for res_list in resources_used.values():
                    resource_ids.extend([res.id for res in res_list])
                resource_ids.sort()
                appointment_key = (m, d, t, frozenset(resource_ids))

                # Add patient to the appropriate appointment group
                appointments_dict[appointment_key].append((p, resources_used))

        appointments = []

        # Create appointments from the grouped data
        for (m, d, t, resource_ids), patient_info_list in appointments_dict.items():
            # Collect patients and ensure resources are consistent
            patients = []
            resources = defaultdict(list)
            first_patient_resources = None

            for p, resources_used in patient_info_list:
                patients.append(p)

                # For the first patient, store the resources used
                if first_patient_resources is None:
                    first_patient_resources = resources_used
                else:
                    # Ensure that resources used are the same for all patients in the appointment
                    for fhat in first_patient_resources:
                        if first_patient_resources[fhat] != resources_used[fhat]:
                            print(
                                f"Warning: Inconsistent resources for patients in appointment {(m.id, d, t)}."
                            )

                # Since resources should be the same, we can use first_patient_resources
                resources = first_patient_resources

            # Create the appointment
            start_date = DayHour(day=d, hour=t)
            appointment = Appointment(
                patients=patients,
                start_date=start_date,
                treatment=m,
                resources=resources,
            )
            appointments.append(appointment)

        patients_arrival: dict[Patient, DayHour] = {}
        for p in self.P:
            for d in self.D_p[p]:
                if a_pd[p, d].X > 0.5:
                    patients_arrival[p] = DayHour(
                        day=d, hour=self.instance.workday_start.hours
                    )
        # Create the solution
        solution = Solution(
            instance=self.instance,
            schedule=appointments,
            patients_arrival=patients_arrival,
        )

        return solution
