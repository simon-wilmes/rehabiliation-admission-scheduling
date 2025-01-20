from src.instance import Instance
import gurobipy as gp
from src.time import Duration
from math import ceil
from numpy import arange
from src.logging import logger, print
from itertools import product
from src.utils import slice_dict
from math import floor, ceil
from src.solution import Appointment, Solution, NO_SOLUTION_FOUND
from src.time import DayHour
from src.patients import Patient
from src.solvers.solver import Solver
from src.treatments import Treatment
from src.resource import Resource, ResourceGroup
from itertools import combinations

PRINT_VARIABLES = False


class MIPSolver(Solver):
    SOLVER_OPTIONS = Solver.BASE_SOLVER_OPTIONS.copy()
    SOLVER_OPTIONS.update(
        {
            "use_lazy_constraints": [True, False],
            "substitute_x_pmdt": [True, False],
            "substitute_x_small": [True, False],
        }
    )

    SOLVER_DEFAULT_OPTIONS = {
        "use_lazy_constraints": True,
        "substitute_x_pmdt": False,
        "substitute_x_small": False,
    }

    def __init__(self, instance: Instance, **kwargs):
        logger.debug(f"Setting options: {self.__class__.__name__}")
        for key in self.__class__.SOLVER_DEFAULT_OPTIONS:

            if key in kwargs:
                setattr(self, key, kwargs[key])
                logger.debug(f" ---- {key} to {kwargs[key]}")
                del kwargs[key]
            else:
                setattr(self, key, self.__class__.SOLVER_DEFAULT_OPTIONS[key])
                logger.debug(
                    f" ---- {key} to { self.__class__.SOLVER_DEFAULT_OPTIONS[key]} (default)"
                )
        super().__init__(instance, **kwargs)
        self._create_parameter_sets()

    def _create_model(self):

        # Create the self.model
        self.model = gp.Model("PatientAdmissionScheduling")
        self.model.setParam("LogToConsole", int(self.log_to_console))  # type: ignore
        self.model.setParam("Threads", self.number_of_threads)  # type: ignore

        # self.model.setParam("CutPasses", 3)
        # self.model.setParam("NoRelHeurTime", self.no_rel_heur_time)  # type: ignore
        self.model.setParam("LazyConstraints", int(self.use_lazy_constraints))  # type: ignore
        vars = self._create_variables()
        self._create_constraints()
        self._set_optimization_goal()

        self.vars = vars

    def _get_expr_value_model(self, model, expr):
        value = expr.getConstant()
        for i in range(expr.size()):
            value += expr.getCoeff(i) * model.cbGetSolution(expr.getVar(i))
        return value

    def _solve_model(self):
        self.num_lazy_added = 0
        self.num_conflicts_found = 0

        def _lazy_constraint_callback(model: gp.Model, where):
            if where == gp.GRB.Callback.MIPSOL:
                # Get all x_pmdt and z_pmgfdt values
                z_pmgfdt = model.cbGetSolution(self.z_pmgfdt)
                if self.substitute_x_pmdt:  # type: ignore
                    x_pmdt = {}
                    for p in self.P:
                        for m, d, t in product(self.M_p[p], self.A_p[p], self.T):
                            x_pmdt[p, m, d, t] = self._get_expr_value_model(
                                model, self.x_pmdt[p, m, d, t]
                            )
                            pass
                else:
                    x_pmdt = model.cbGetSolution(self.x_pmdt)

                for d, t, m in product(self.D, self.T, self.M):
                    patients = set()
                    for p in self.P:
                        if (p, m, d, t) in x_pmdt and x_pmdt[p, m, d, t] > 0.5:
                            patients.add(p)

                    groups: list[tuple[set[Patient], set[Resource]]] = []
                    if len(patients) > 1:

                        for p in sorted(patients):
                            resources_by_patient = {
                                f
                                for fhat in self.Fhat_m[m]
                                for f in self.fhat[fhat]
                                if z_pmgfdt[p, m, fhat, f, d, t] > 0.5
                            }
                            added = False
                            for patients_in_group, resources_in_group in groups:
                                if len(resources_by_patient & resources_in_group) > 0:
                                    # if patient shares a resource with the group, add patient to the group
                                    patients_in_group.add(p)
                                    resources_in_group |= resources_by_patient
                                    added = True
                            if not added:
                                groups.append((set([p]), resources_by_patient))

                    for group_p, group_f in groups:
                        if len(group_f) > self.total_num_resources_m[m]:
                            self.num_conflicts_found += 1
                            for (p1, p2), ((fhat1, f1), (fhat2, f2)) in product(
                                combinations(sorted(list(group_p)), 2),
                                product(
                                    self.all_resources_possibly_used_by_m[m], repeat=2
                                ),
                            ):

                                if f1 == f2:
                                    continue

                                self.num_lazy_added += 1
                                model.cbLazy(
                                    self.z_pmgfdt[p1, m, fhat1, f1, d, t]
                                    >= self.z_pmgfdt[p2, m, fhat1, f1, d, t]
                                    + self.z_pmgfdt[p1, m, fhat2, f2, d, t]
                                    + self.z_pmgfdt[p2, m, fhat2, f2, d, t]
                                    - 2
                                )

        if self.use_lazy_constraints:  # type: ignore
            self.model.optimize(_lazy_constraint_callback)
        else:
            self.model.optimize()
        if (
            self.model.status == gp.GRB.OPTIMAL
            or self.model.status == gp.GRB.TIME_LIMIT
            or self.model.status == gp.GRB.INTERRUPTED
        ):
            logger.info("Optimal solution found.")
            logger.debug("Sub Objectives:")
            logger.debug(f"(SOLVER):Minimize Delay: {self.md.X}")
            logger.debug(f"(SOLVER):Minimize Missing Treatments: {self.mmt.X}")

            self.solution_found = True

        else:
            logger.info("No optimal solution found.")
            self.solution_found = False
            return
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

    def _set_optimization_goal(self):

        minimize_delay = gp.quicksum(
            (d - p.earliest_admission_date.day) * self.a_pd[p, d]
            for p in self.P
            for d in self.D_p[p]
        )

        minimize_missing_treatment = gp.LinExpr()
        for p in self.P:
            # Sum up the number of all treatments for patient p
            scheduled_treatments = gp.quicksum(
                value
                for (p_prime, _, _, _), value in self.x_pmdt.items()
                if p_prime == p
            )
            total_treatments = sum(self.lr_pm[p, m] for m in self.M_p[p])

            minimize_missing_treatment += total_treatments - scheduled_treatments

        objective = (
            +self.delay_value * minimize_delay  # type: ignore
            + self.missing_treatment_value * minimize_missing_treatment  # type: ignore
        )
        # Create variables for easier extraction of individual values
        self.md = self.model.addVar(name="minimize_delay", vtype=gp.GRB.CONTINUOUS)
        self.mmt = self.model.addVar(
            name="minimize_missing_treatment", vtype=gp.GRB.CONTINUOUS
        )

        self.model.addConstr(self.md == self.delay_value * minimize_delay)
        self.model.addConstr(
            self.mmt == self.missing_treatment_value * minimize_missing_treatment
        )

        self.model.setObjective(objective, gp.GRB.MINIMIZE)

    def _create_parameter_sets(self):
        # Define any additional sets or mappings needed for the self.model
        super()._create_parameter_sets()
        self.total_num_resources_m = {m: sum(m.resources.values()) for m in self.M}

        self.all_resources_possibly_used_by_m: dict[
            Treatment, list[tuple[ResourceGroup, Resource]]
        ] = {}
        for m in self.M:
            resources = set()
            for fhat in self.Fhat_m[m]:
                resources |= set((fhat, f) for f in self.fhat[fhat])
            self.all_resources_possibly_used_by_m[m] = sorted(resources)
        pass

    def _get_smaller_patients(self, p: Patient) -> list[Patient]:
        ind1 = self.P.index(p)
        return self.P[:ind1]

    def _create_variables(self):
        #####################################
        # Create variables
        #####################################
        # Create self.z_pmfdt
        z_pmfhatfdt_keys = []
        for p in self.P:
            for m in self.M_p[p]:
                for fhat in self.Fhat_m[m]:
                    z_pmfhatfdt_keys.extend(
                        product([p], [m], [fhat], self.fhat[fhat], self.A_p[p], self.T)
                    )
        self.z_pmgfdt = self.model.addVars(
            z_pmfhatfdt_keys, vtype=gp.GRB.BINARY, name="z_pmgfdt"
        )
        # Create self.x_pmdt
        if self.substitute_x_pmdt:  # type: ignore
            self.x_pmdt = {}
            for p in self.P:
                for m, d, t in product(self.M_p[p], self.A_p[p], self.T):
                    self.x_pmdt[p, m, d, t] = gp.LinExpr()
                    num_added = 0
                    for fhat in sorted(self.Fhat_m[m], key=lambda fhat: fhat.id):
                        self.x_pmdt[p, m, d, t] += (
                            gp.quicksum(
                                self.z_pmgfdt[p, m, fhat, f, d, t]
                                for f in self.fhat[fhat]
                            )
                            / self.n_fhatm[fhat, m]
                        )
                        num_added += 1
                        if self.substitute_x_small:  # type: ignore
                            break
                    self.x_pmdt[p, m, d, t] /= num_added

        else:
            x_pmdt_keys = []
            for p in self.P:
                x_pmdt_keys.extend(product([p], self.M_p[p], self.A_p[p], self.T))

            self.x_pmdt = self.model.addVars(
                x_pmdt_keys, vtype=gp.GRB.BINARY, name="x_pmdt"
            )

        # Create self.y_mfdt
        y_mgfdt_keys = []
        for m in self.M:
            for fhat in self.Fhat_m[m]:
                y_mgfdt_keys.extend(
                    product([m], [fhat], self.fhat[fhat], self.D, self.T)
                )
        self.y_mgfdt = self.model.addVars(
            y_mgfdt_keys, vtype=gp.GRB.BINARY, name="y_mfdt"
        )

        # Create self.a_pd
        a_pd_keys = []
        for p in self.P:
            a_pd_keys.extend(product([p], self.D_p[p]))
        self.a_pd = self.model.addVars(a_pd_keys, vtype=gp.GRB.BINARY, name="a_pd")

        logger.debug("Variables created.")

    def _create_constraints(self):
        self.model.update()
        #####################################
        # Create constraints
        #####################################

        # Constraint: y_fmdt are set correctly
        for m, g, f, d, t in self.y_mgfdt.keys():
            for p in self.P:
                if d in self.A_p[p] and m in self.M_p[p]:
                    self.model.addConstr(
                        self.z_pmgfdt[p, m, g, f, d, t] <= self.y_mgfdt[m, g, f, d, t],
                        name=f"constraint_y_fmdt_m{m.id}_fhat{g.id}_f{f.id}_d{d}_t{t}_p{p.id}",
                    )

        # Constraint (p1): Sum over all scheduled treatments equals total repetitions left
        for p in self.P:
            for m in self.M_p[p]:
                self.model.addConstr(
                    gp.quicksum(
                        self.x_pmdt[p, m, d, t] for d in self.A_p[p] for t in self.T
                    )
                    <= self.lr_pm[p, m],
                    name=f"constraint_p1_p{p.id}_m{m.id}",
                )
        logger.debug("Constraint (p1) created.")
        # Constraint (p2): Patients not admitted have no treatments scheduled
        for p in self.P:
            for m in self.M_p[p]:
                for d in self.A_p[p]:
                    delta_set = [
                        delta for delta in self.D_p[p] if d - self.l_p[p] < delta <= d
                    ]
                    for t in self.T:
                        self.model.addConstr(
                            self.x_pmdt[p, m, d, t]
                            <= gp.quicksum(self.a_pd[p, delta] for delta in delta_set),
                            name=f"constraint_m_when_admitted_p{p.id}_m{m.id}_d{d}_t{t}",
                        )
        logger.debug("Constraint (when_admitted) created.")
        # Constraint (p3): Only one treatment at a time per patient
        for p in self.P:
            for d in self.A_p[p]:
                for t in self.T:
                    expr = gp.LinExpr()
                    for m in self.M_p[p]:
                        tau_set = [
                            tau
                            for tau in self.T
                            if t - self.du_m[m] * self.instance.time_slot_length.hours
                            < tau
                            <= t
                        ]
                        expr += gp.quicksum(
                            self.x_pmdt[p, m, d, tau] for tau in tau_set
                        )
                    self.model.addConstr(
                        expr <= 1, name=f"constraint_p3_p{p.id}_d{d}_t{t}"
                    )
        logger.debug("Constraint (p3) created.")
        # Constraint (p4): Total admitted patients cannot exceed total beds
        for d in range(
            max([p.admitted_before_date.day + p.length_of_stay for p in self.P]) + 1
        ):
            expr = gp.LinExpr()
            for p in self.P:
                delta_set = [
                    delta for delta in self.D_p[p] if d - self.l_p[p] < delta <= d
                ]
                expr += gp.quicksum(self.a_pd[p, delta] for delta in delta_set)
            self.model.addConstr(expr <= self.b, name=f"constraint_total_beds_d{d}")
        logger.debug("Constraint (p4) created.")
        # Constraint (p6): Patient admitted exactly once within the specified time
        for p in self.P:
            self.model.addConstr(
                gp.quicksum(self.a_pd[p, d] for d in self.D_p[p]) == 1,
                name=f"constraint_p6_p{p.id}",
            )

        logger.debug("Constraint (p7) created.")
        # Constraint (r2): Resource availability and utilization
        for f in self.F:
            for d in self.D:
                for t in self.T:

                    expr = gp.LinExpr()
                    for fhat in f.resource_groups:
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
                                self.y_mgfdt[m, fhat, f, d, tau]
                                for tau in tau_set
                                if (m, fhat, f, d, tau) in self.y_mgfdt
                            )
                    self.model.addConstr(
                        expr <= self.av_fdt[f, d, t],
                        name=f"constraint_r2_fhat{fhat.id}_f{f.id}_d{d}_t{t}",
                    )

        logger.debug("Constraint (r3) created.")
        # Constraint (r4): Assign required number of resources for each treatment
        for p in self.P:
            for m in self.M_p[p]:
                for fhat in self.Fhat_m[m]:
                    for d in self.A_p[p]:
                        for t in self.T:
                            lhs = gp.quicksum(
                                self.z_pmgfdt[p, m, fhat, f, d, t]
                                for f in self.fhat[fhat]
                            )
                            rhs = self.n_fhatm[fhat, m] * self.x_pmdt[p, m, d, t]
                            self.model.addConstr(
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
                                self.z_pmgfdt[p, m, fhat, f, d, t]
                                for p in self.P
                                if (p, m, fhat, f, d, t) in self.z_pmgfdt
                            )
                            self.model.addConstr(
                                lhs <= self.k_m[m],
                                name=f"constraint_max_patients_m{m.id}_f{f.id}_d{d}_t{t}",
                            )

        # Constraint (a1): Limit the number of patients per treatment
        for m in self.M:
            for fhat in self.Fhat_m[m]:
                for f in self.fhat[fhat]:
                    for d in self.D:
                        for t in self.T:
                            lhs = gp.quicksum(
                                self.z_pmgfdt[p, m, fhat, f, d, t]
                                for p in self.P
                                if (p, m, fhat, f, d, t) in self.z_pmgfdt
                            )
                            self.model.addConstr(
                                lhs >= self.j_m[m] * self.y_mgfdt[m, fhat, f, d, t],
                                name=f"constraint_min_patients_m{m.id}_f{f.id}_d{d}_t{t}",
                            )

        logger.debug("Constraint (a1) created.")

        # Constraint (a4): Max num of treatments over e_w days

        for p in self.P:
            # Skip patients that do not require even distribution as they stay fewer days than the
            # length of the rolling window
            if p.length_of_stay <= self.e_w:
                continue
            for d in self.A_p[p]:
                # check if the window is partially outside of patients stay => ignore
                self.model.addConstr(
                    gp.quicksum(
                        self.x_pmdt[p, m, d_prime, t]
                        for m in self.M_p[p]
                        for t in self.T
                        for d_prime in range(d, d + self.e_w)
                        if d_prime in self.A_p[p]
                    )
                    <= self.e_w_upper[p],
                    name=f"constraint_a4_ub_p{p.id}_d{d}",
                )

        logger.debug("Constraint (a4) created.")

        # Constraint (a5): Max num of treatments every day
        for p in self.P:
            for d in self.A_p[p]:
                delta_set = [
                    delta for delta in self.D_p[p] if d - self.l_p[p] < delta <= d
                ]

                self.model.addConstr(
                    gp.quicksum(
                        self.x_pmdt[p, m, d, t] for m in self.M_p[p] for t in self.T
                    )
                    <= self.daily_upper[p]
                    * gp.quicksum(self.a_pd[p, delta] for delta in delta_set),
                    name=f"constraint_a5_ub_p{p.id}_d{d}",
                )
        logger.debug("Constraint (a5) created.")

        # Constraint (a7): Min num of treatments every day
        for p in self.P:
            for d in self.A_p[p]:
                # Test if for day d the patient is always admitted then we can enforce the lower bound
                if (
                    p.admitted_before_date.day - 1
                    <= d
                    < p.earliest_admission_date.day + p.length_of_stay
                ):

                    self.model.addConstr(
                        gp.quicksum(
                            self.x_pmdt[p, m, d, t] for m in self.M_p[p] for t in self.T
                        )
                        >= self.daily_lower[p],
                        name=f"constraint_a6_lb_p{p.id}_d{d}",
                    )
                else:
                    # otherwise only enfore the lowerbound if admitted before
                    delta_set = [
                        delta for delta in self.D_p[p] if d - self.l_p[p] < delta <= d
                    ]

                    self.model.addConstr(
                        gp.quicksum(
                            self.x_pmdt[p, m, d, t] for m in self.M_p[p] for t in self.T
                        )
                        >= self.daily_lower[p]
                        * gp.quicksum(self.a_pd[p, delta] for delta in delta_set),
                        name=f"constraint_a6_lb_p{p.id}_d{d}",
                    )

            logger.debug("Constraint (a5) created.")
        # Constraint: Make sure that already admitted patients are admitted on the first day of the planning horizon
        for p in self.P:
            if p.already_admitted:
                self.model.addConstr(
                    self.a_pd[p, 0] == 1, name=f"constraint_already_admitted_p{p.id}"
                )
        logger.debug("Constraint (already_admitted) created.")

        if not self.use_lazy_constraints:  # type: ignore
            # Constraint: Group people into sessions, enforce transitive property amongs resource usage
            p1_index = 0
            for p1, p2 in product(self.P, self.P):
                if self.P.index(p1) > p1_index:
                    p1_index = self.P.index(p1)
                    logger.debug(f"New patient index seen: {p1_index}")
                if self.P.index(p1) >= self.P.index(p2):
                    continue
                # Find iterate over common treatments
                for m in sorted(set(self.M_p[p1]) & set(self.M_p[p2])):

                    for (fhat1, f1), (fhat2, f2) in product(
                        self.all_resources_possibly_used_by_m[m],
                        self.all_resources_possibly_used_by_m[m],
                    ):
                        D_p1p2 = sorted(set(self.A_p[p1]) & set(self.A_p[p2]))
                        for d, t in product(D_p1p2, self.T):
                            self.model.addConstr(
                                self.z_pmgfdt[p1, m, fhat1, f1, d, t]
                                >= self.z_pmgfdt[p2, m, fhat1, f1, d, t]
                                + self.z_pmgfdt[p1, m, fhat2, f2, d, t]
                                + self.z_pmgfdt[p2, m, fhat2, f2, d, t]
                                - 2,
                                name=f"constraint_transitive_p{p1.id}_p{p2.id}_m{m.id}_f{f.id}_{d}_{t}",
                            )
            logger.debug("Constraint (transitive) created.")

    def get_x_pmdt_value(self, p, m, d, t):
        if self.substitute_x_pmdt:  # type: ignore
            return self.x_pmdt[p, m, d, t].getValue()  # type: ignore
        else:
            return self.x_pmdt[p, m, d, t].X

    def _extract_solution(self):
        """
        Extracts the solution from the MIP self.model and constructs a Solution object.

        Returns:
            Solution: The constructed solution with appointments.
        """
        if not self.solution_found:
            return NO_SOLUTION_FOUND
        from collections import defaultdict

        # Print out all variables that are one
        if False:
            for key in self.x_pmdt:
                if self.get_x_pmdt_value(*key) > 0.5:  # type: ignore
                    logger.debug(f"self.x_pmdt{key} = {self.get_x_pmdt_value(*key)}")  # type: ignore

            for key in self.z_pmgfdt:
                if self.z_pmgfdt[key].X > 0.5:  # type: ignore
                    logger.debug(f"self.z_pmfdt{key} = {self.z_pmgfdt[key].X}")  # type: ignore

            for key in self.a_pd:
                if self.a_pd[key].X > 0.5:  # type: ignore
                    logger.debug(f"self.a_pd{key} = {self.a_pd[key].X}")  # type: ignore

            for key in self.y_mgfdt:
                if self.y_mgfdt[key].X > 0.5:
                    logger.debug(f"self.y_mgfdt{key} = {self.y_mgfdt[key].X}")

        # appointments_dict: key=(m, d, t, frozenset of resource IDs), value=list of patients
        appointments_dict = defaultdict(list)

        # Collect scheduled treatments and group patients based on resources used
        for (p, m, d, t), var in self.x_pmdt.items():  # type: ignore
            if self.get_x_pmdt_value(p, m, d, t) > 0.5:
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
                        if fhat not in f.resource_groups:
                            continue
                        z_key = (p, m, fhat, f, d, t)
                        if z_key in self.z_pmgfdt and self.z_pmgfdt[z_key].X > 0.5:  # type: ignore
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
                solver=self,
            )
            appointments.append(appointment)

        patients_arrival: dict[Patient, DayHour] = {}
        for p in self.P:
            for d in self.D_p[p]:
                if self.a_pd[p, d].X > 0.5:  # type: ignore
                    patients_arrival[p] = DayHour(
                        day=d, hour=self.instance.workday_start.hour
                    )

        # Create the solution
        solution = Solution(
            instance=self.instance,
            schedule=appointments,
            patients_arrival=patients_arrival,
            solver=self,
            solution_value=self.model.objVal,
        )

        return solution
