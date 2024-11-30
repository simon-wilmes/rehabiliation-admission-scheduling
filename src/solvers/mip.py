from src.instance import Instance
import gurobipy as gp
from src.time import Duration
from math import ceil
from numpy import arange
from src.logging import logger, print
from itertools import product

from src.solution import Appointment, Solution, NO_SOLUTION_FOUND
from src.time import DayHour
from src.patients import Patient
from src.solvers.solver import Solver

PRINT_VARIABLES = False


class MIPSolver(Solver):
    SOLVER_OPTIONS = Solver.BASE_SOLVER_OPTIONS.copy()
    SOLVER_OPTIONS.update([])  # Add any additional options here

    SOLVER_DEFAULT_OPTIONS = {}

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

    def _create_model(self):
        self._create_parameter_sets()
        # Create the model
        model = gp.Model("PatientAdmissionScheduling")
        model.setParam("LogToConsole", int(self.log_to_console))  # type: ignore
        model.setParam("Threads", self.number_of_threads)  # type: ignore
        model.setParam("Cuts", 0)
        model.setParam("CutPasses", 3)
        vars = self._create_variables(model)
        self._create_constraints(model, *vars)
        self._set_optimization_goal(model, *vars)

        self.vars = vars
        self.model = model

    def _solve_model(self) -> Solution | int:
        self.model.optimize()
        if (
            self.model.status == gp.GRB.OPTIMAL
            or self.model.status == gp.GRB.TIME_LIMIT
            or self.model.status == gp.GRB.INTERRUPTED
        ):
            logger.debug("Optimal solution found.")
            solution = self._extract_solution()
            return solution
        else:
            logger.error("No optimal solution found.")
            # Compute IIS
            self.model.computeIIS()

            # Print IIS
            logger.debug("\nIIS Constraints:")
            for constr in self.model.getConstrs():
                if constr.IISConstr:
                    logger.debug(f"Constraint {constr.ConstrName} is in the IIS")

            logger.debug("\nIIS Variables:")
            for var in self.model.getVars():
                if var.IISLB or var.IISUB:
                    logger.debug(f"Variable {var.VarName} is in the IIS with bounds")
            return NO_SOLUTION_FOUND

    def _set_optimization_goal(
        self, model: gp.Model, x_pmdt, z_pmfdt, u_mfdt, v_pmf, a_pd
    ):

        o_pmdt_keys = []
        for p in self.P:
            o_pmdt_keys.extend(product([p], self.M_p[p], self.A_p[p], self.T))
        o_pmdt = model.addVars(o_pmdt_keys, vtype=gp.GRB.BINARY, name="o_pmdt")

        for p in self.P:
            for m, d, t in product(self.M_p[p], self.A_p[p], self.T):
                for fhat in self.Fhat_m[m]:
                    for f in self.fhat[fhat]:
                        model.addConstr(
                            o_pmdt[p, m, d, t]
                            >= z_pmfdt[p, m, f, d, t]
                            - gp.quicksum(
                                z_pmfdt[p_prime, m, f, d, t]
                                for p_prime in self._get_smaller_patients(p)
                                if (p_prime, m, f, d, t) in z_pmfdt
                            )
                        )
                        pass
        logger.debug("Constraint (optimization) created.")
        minimize_treatments = gp.quicksum(
            o_pmdt[p, m, d, t] for p, m, d, t in o_pmdt_keys
        )

        minimize_delay = gp.quicksum(
            (d - p.earliest_admission_date.day) * a_pd[p, d]
            for p in self.P
            for d in self.A_p[p]
        )

        minimize_missing_treatment = gp.quicksum(
            self.lr_pm[p, m]
            - gp.quicksum(x_pmdt[p, m, d, t] for d in self.A_p[p] for t in self.T)
            for p in self.P
            for m in self.M_p[p]
        )

        # Set objective
        objective = (
            self.treatment_value * minimize_treatments  # type: ignore
            + self.delay_value * minimize_delay  # type: ignore
            + self.missing_treatment_value * minimize_missing_treatment  # type: ignore
        )

        model.setObjective(objective, gp.GRB.MINIMIZE)

    def _create_parameter_sets(self):
        # Define any additional sets or mappings needed for the model
        super()._create_parameter_sets()

    def _get_smaller_patients(self, p: Patient) -> list[Patient]:
        ind1 = self.P.index(p)
        return self.P[:ind1]

    def _create_variables(self, model: gp.Model):
        #####################################
        # Create variables
        #####################################
        # Create x_pmdt
        x_pmdt_keys = []
        for p in self.P:
            x_pmdt_keys.extend(product([p], self.M_p[p], self.A_p[p], self.T))

        x_pmdt = model.addVars(x_pmdt_keys, vtype=gp.GRB.BINARY, name="x_pmdt")
        # Create z_pmfdt
        z_pmfdt_keys = []
        for p in self.P:
            for m in self.M_p[p]:
                for fhat in self.Fhat_m[m]:
                    z_pmfdt_keys.extend(
                        product([p], [m], self.fhat[fhat], self.A_p[p], self.T)
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
            a_pd_keys.extend(product([p], self.A_p[p]))
        a_pd = model.addVars(a_pd_keys, vtype=gp.GRB.BINARY, name="a_pd")

        self.vars = {
            "x_pmdt": x_pmdt,
            "z_pmfdt": z_pmfdt,
            "u_mfdt": u_mfdt,
            "v_pmf": v_pmf,
            "a_pd": a_pd,
        }
        logger.debug("Variables created.")
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
                    gp.quicksum(x_pmdt[p, m, d, t] for d in self.A_p[p] for t in self.T)
                    <= self.lr_pm[p, m],
                    name=f"constraint_p1_p{p.id}_m{m.id}",
                )
        logger.debug("Constraint (p1) created.")
        # Constraint (p2): Patients not admitted have no treatments scheduled
        for p in self.P:
            for m in self.M_p[p]:
                for d in self.A_p[p]:
                    for t in self.T:
                        delta_set = [
                            delta
                            for delta in self.A_p[p]
                            if d - self.l_p[p] < delta <= d
                        ]
                        model.addConstr(
                            x_pmdt[p, m, d, t]
                            <= gp.quicksum(a_pd[p, delta] for delta in delta_set),
                            name=f"constraint_p2_p{p.id}_m{m.id}_d{d}_t{t}",
                        )
        logger.debug("Constraint (p2) created.")
        # Constraint (p3): Only one treatment at a time per patient
        for p in self.P:
            for d in self.A_p[p]:
                for t in self.T:
                    expr = gp.LinExpr()
                    for m in self.M_p[p]:
                        tau_set = [tau for tau in self.T if t - self.du_m[m] * self.instance.time_slot_length.hours < tau <= t]
                        expr += gp.quicksum(x_pmdt[p, m, d, tau] for tau in tau_set)
                    model.addConstr(expr <= 1, name=f"constraint_p3_p{p.id}_d{d}_t{t}")
        logger.debug("Constraint (p3) created.")
        # Constraint (p4): Total admitted patients cannot exceed total beds
        for d in self.D:
            expr = gp.LinExpr()
            for p in self.P:
                delta_set = [
                    delta for delta in self.A_p[p] if d - self.l_p[p] < delta <= d
                ]
                expr += gp.quicksum(a_pd[p, delta] for delta in delta_set)
            model.addConstr(expr <= self.b, name=f"constraint_p4_d{d}")
        logger.debug("Constraint (p4) created.")
        # Constraint (p6): Patient admitted exactly once within the specified time
        for p in self.P:
            D_p_early = [d for d in self.A_p[p] if d <= max(self.A_p[p]) - self.l_p[p]]
            model.addConstr(
                gp.quicksum(a_pd[p, d] for d in D_p_early) == 1,
                name=f"constraint_p6_p{p.id}",
            )
        logger.debug("Constraint (p6) created.")
        # Constraint (p7): Patient is admitted exactly once
        for p in self.P:
            model.addConstr(
                gp.quicksum(a_pd[p, d] for d in self.A_p[p]) == 1,
                name=f"constraint_p7_p{p.id}",
            )
        logger.debug("Constraint (p7) created.")
        # Constraint (r2): Resource availability and utilization
        for fhat in self.Fhat:
            for f in self.fhat[fhat]:
                for d in self.D:
                    for t in self.T:
                        expr = gp.LinExpr()
                        for m in self.M_fhat[fhat]:
                            du_m_m = self.du_m[m]
                            tau_set = [
                                tau
                                for tau in self.T
                                if t - du_m_m * self.instance.time_slot_length.hours
                                < tau
                                <= t
                            ]
                            expr += gp.quicksum(
                                u_mfdt[m, f, d, tau]
                                for tau in tau_set
                                if (m, f, d, tau) in u_mfdt
                            )
                        model.addConstr(
                            expr <= self.av_fdt[f, d, t],
                            name=f"constraint_r2_fhat{fhat.id}_f{f.id}_d{d}_t{t}",
                        )
        logger.debug("Constraint (r2) created.")
        # Constraint (r3): Linking z and u variables
        for p in self.P:
            for m in self.M_p[p]:
                for fhat in self.Fhat_m[m]:
                    for f in self.fhat[fhat]:
                        for d in self.A_p[p]:
                            for t in self.T:
                                model.addConstr(
                                    z_pmfdt[p, m, f, d, t] <= u_mfdt[m, f, d, t],
                                    name=f"constraint_r3_p{p.id}_m{m.id}_f{f.id}_d{d}_t{t}",
                                )
        logger.debug("Constraint (r3) created.")
        # Constraint (r4): Assign required number of resources for each treatment
        for p in self.P:
            for m in self.M_p[p]:
                for fhat in self.Fhat_m[m]:
                    for d in self.A_p[p]:
                        for t in self.T:
                            lhs = gp.quicksum(
                                z_pmfdt[p, m, f, d, t] for f in self.fhat[fhat]
                            )
                            rhs = self.n_fhatm[fhat, m] * x_pmdt[p, m, d, t]
                            model.addConstr(
                                lhs == rhs,
                                name=f"constraint_r4_p{p.id}_m{m.id}_fhat{fhat.id}_d{d}_t{t}",
                            )
        logger.debug("Constraint (r4) created.")
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
        logger.debug("Constraint (a1) created.")
        if self.add_resource_loyal():
            # Constraint (a2-help): Resource loyalty linking
            for p in self.P:
                for m in self.M_p[p]:
                    for fhat in self.Fhat_m[m]:
                        for f in self.fhat[fhat]:
                            for d in self.A_p[p]:
                                for t in self.T:
                                    model.addConstr(
                                        z_pmfdt[p, m, f, d, t] <= v_pmf[p, m, f],
                                        name=f"constraint_a2_help_p{p.id}_m{m.id}_f{f.id}_d{d}_t{t}",
                                    )

        # Constraint (a2): Resource loyalty constraint
        if self.add_resource_loyal():
            for p in self.P:
                for m in self.M_p[p]:
                    for fhat in self.Lhat_m[m]:
                        model.addConstr(
                            gp.quicksum(v_pmf[p, m, f] for f in self.fhat[fhat])
                            <= self.n_fhatm[fhat, m],
                            name=f"constraint_a2_p{p.id}_m{m.id}_fhat{fhat.id}",
                        )

        # Constraint (a4): Even distribution of treatments over rolling windows
        if self.add_even_distribution():
            for d in self.R:
                for p in self.P:
                    for m in self.M_p[p]:
                        num_windows = ceil(len(self.A_p[p]) / self.rw)
                        for w in range(num_windows):
                            window_start = self.A_p[p][0] + w * self.rw
                            window_end = min(
                                window_start + self.rw - 1, self.A_p[p][-1]
                            )
                            window_days = [
                                d
                                for d in self.A_p[p]
                                if window_start <= d <= window_end
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
        logger.debug("Constraint (already_admitted) created.")
        # Constraint: Make sure that already_resource_loyal is respected
        # Constraint: Ensure already resource loyal assignments are set to 1
        if self.add_resource_loyal():
            for p in self.P:
                if hasattr(p, "already_resource_loyal"):
                    for (m, fhat), resources in p.already_resource_loyal.items():
                        for f in resources:
                            model.addConstr(
                                v_pmf[p, m, f] == 1,
                                name=f"constraint_loyal_resource_p{p.id}_m{m.id}_f{f.id}",
                            )

        # Constraint: Group people into sessions, enforce transitive property amongs resource usage
        p1_index = 0
        for p1, p2 in product(self.P, self.P):
            if self.P.index(p1) > p1_index:
                p1_index = self.P.index(p1)
                logger.debug(f"New patient index seen: {p1_index}")
            if self.P.index(p1) >= self.P.index(p2):
                continue
            # Find iterate over common treatments
            for m in set(self.M_p[p1]) & set(self.M_p[p2]):
                common_resources = set()
                for fhat in self.Fhat_m[m]:
                    common_resources |= set(self.fhat[fhat])

                for f1, f2 in product(common_resources, common_resources):
                    if self.F.index(f1) >= self.F.index(f2):
                        continue
                    D_p1p2 = set(self.A_p[p1]) & set(self.A_p[p2])
                    for d, t in product(D_p1p2, self.T):
                        model.addConstr(
                            z_pmfdt[p1, m, f1, d, t]
                            >= z_pmfdt[p2, m, f1, d, t]
                            + z_pmfdt[p1, m, f2, d, t]
                            + z_pmfdt[p2, m, f2, d, t]
                            - 2
                        )
        logger.debug("Constraint (transitive) created.")

    def _extract_solution(self):
        """
        Extracts the solution from the MIP model and constructs a Solution object.

        Returns:
            Solution: The constructed solution with appointments.
        """
        from collections import defaultdict

        x_pmdt, z_pmfdt, u_mfdt, v_pmf, a_pd = self.vars
        # Print out all variables that are one
        if PRINT_VARIABLES:
            for key in x_pmdt:
                if x_pmdt[key].X > 0.5:  # type: ignore
                    logger.debug(f"x_pmdt{key} = {x_pmdt[key].X}")  # type: ignore

            for key in z_pmfdt:
                if z_pmfdt[key].X > 0.5:  # type: ignore
                    logger.debug(f"z_pmfdt{key} = {z_pmfdt[key].X}")  # type: ignore

            for key in u_mfdt:
                if u_mfdt[key].X > 0.5:  # type: ignore
                    logger.debug(f"u_mfdt{key} = {u_mfdt[key].X}")  # type: ignore

            for key in v_pmf:
                if v_pmf[key].X > 0.5:  # type: ignore
                    logger.debug(f"v_pmf{key} = {v_pmf[key].X}")  # type: ignore

            for key in a_pd:
                if a_pd[key].X > 0.5:  # type: ignore
                    logger.debug(f"a_pd{key} = {a_pd[key].X}")  # type: ignore

        # appointments_dict: key=(m, d, t, frozenset of resource IDs), value=list of patients
        appointments_dict = defaultdict(list)

        # Collect scheduled treatments and group patients based on resources used
        for (p, m, d, t), var in x_pmdt.items():  # type: ignore
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
                        if z_key in z_pmfdt and z_pmfdt[z_key].X > 0.5:  # type: ignore
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
            for d in self.A_p[p]:
                if a_pd[p, d].X > 0.5:  # type: ignore
                    patients_arrival[p] = DayHour(
                        day=d, hour=self.instance.workday_start.hour
                    )

        # Create the solution
        solution = Solution(
            instance=self.instance,
            schedule=appointments,
            patients_arrival=patients_arrival,
            solver=self,
            test_even_distribution=self.add_even_distribution(),
            test_conflict_groups=self.add_conflict_groups(),
            test_resource_loyalty=self.add_resource_loyal(),
            solution_value=self.model.objVal,
        )

        return solution
