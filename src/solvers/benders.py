from src.solvers.solver import Solver
import gurobipy as gp
from gurobipy import Var
from datetime import timedelta

from collections import defaultdict
from src.logging import logger
from src.instance import Instance
from src.time import DayHour, Duration
from src.solution import Solution, Appointment
from src.solution import NO_SOLUTION_FOUND

from math import floor, gcd, ceil
from pprint import pformat
from .subsolvers import Subsolver, CPSubsolver
from itertools import product, combinations
from typing import Callable, Any


class LBBDSolver(Solver):

    SOLVER_OPTIONS = Solver.BASE_SOLVER_OPTIONS.copy()
    SOLVER_OPTIONS.update(
        {
            "break_symmetry": [True, False],
            "use_helper_constraints": [True, False],
        }
    )  # Add any additional options here

    SOLVER_DEFAULT_OPTIONS = {
        "break_symmetry": True,
        "subsolver_cls": CPSubsolver,
        "use_helper_constraints": True,
    }

    def __init__(self, instance: Instance, **kwargs):
        logger.debug(f"Setting options: {self.__class__.__name__}")

        for key in kwargs:
            assert (
                key in self.__class__.BASE_SOLVER_DEFAULT_OPTIONS
                or key in self.__class__.SOLVER_DEFAULT_OPTIONS
                or key.startswith("subsolver.")
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

        subsolver_kwargs = {
            key[len("subsolver.") :]: value
            for key, value in kwargs.items()
            if key.startswith("subsolver.")
        }

        super().__init__(instance, **kwargs)
        self._create_parameter_sets()

        # Add min_patients_per_treatment to subsolver kwargs
        subsolver_kwargs["enforce_min_patients_per_treatment"] = self.enforce_min_patients_per_treatment  # type: ignore

        self.subsolver: Subsolver = self.subsolver_cls(instance=instance, solver=self, **subsolver_kwargs)  # type: ignore

        self.count_good_cuts = 0
        self.count_bad_cuts = 0
        self.already_seen_forbidden_strs = set()
        self.count_unnecessary_comps = 0

    def build_patients_dict(self, f_is_true: Callable[[Var], bool]):
        patients = [{m: {} for m in self.M} for _ in self.D]
        index = 0
        for p in self.P:
            for m in self.M_p[p]:
                for d in self.A_p[p]:
                    found_false = False
                    for r in self.BD_L_pm[p, m]:
                        if not sum(
                            f_is_true(self.x_pmdri[p, m, d, r, i])
                            for i in self.max_repetitions[m]
                        ):
                            break
                    else:
                        found_false = True
                        patients[d][m][p] = r

                    if not found_false and r > 1:
                        patients[d][m][p] = r - 1

        return patients

    def _solve_model(self):
        # Define the callback for creating subproblems
        def _solution_callback(model, where):
            if where == gp.GRB.Callback.MIPSOL:
                try:
                    logger.debug("Solution callback")
                    # Retrieve the solution
                    # patients day assignments

                    patients = self.build_patients_dict(
                        lambda x: model.cbGetSolution(x) > 0.5
                    )

                    logger.debug("Subproblem created.")
                    logger.debug("\n" + pformat(patients))
                    count_days_infeasible = 0
                    for d in self.D:
                        logger.debug(f"Checking day {d}")
                        logger.debug(f"Patients: {pformat(patients[d])}")
                        result = self.subsolver.is_day_infeasible(d, patients[d])  # type: ignore
                        status_code = result["status_code"]
                        if result["status_code"] == Subsolver.COMPLETELY_INFEASIBLE:
                            self.count_good_cuts += 1
                            # not is Feasible, generate cut
                            logger.debug("Subproblem not feasible.")
                            count_days_infeasible += 1

                            forbidden_vars_str = []
                            forbidden_vars = []
                            for m in patients[d]:
                                for p in patients[d][m]:
                                    if patients[d][m][p] > 0:
                                        forbidden_vars.append(
                                            sum(
                                                self.x_pmdri[
                                                    p, m, d, patients[d][m][p], i
                                                ]
                                                for i in self.max_repetitions[m]
                                            )
                                        )
                                        forbidden_vars_str.append(
                                            f"{m.id},{patients[d][m][p]}"
                                        )
                            forbidden_vars_str = "|".join(sorted(forbidden_vars_str))

                            logger.info("forbidden_vars: " + forbidden_vars_str)
                            logger.info(
                                forbidden_vars_str in self.already_seen_forbidden_strs
                            )
                            self.count_unnecessary_comps += (
                                forbidden_vars_str in self.already_seen_forbidden_strs
                            )
                            self.already_seen_forbidden_strs.add(forbidden_vars_str)

                            model.cbLazy(
                                gp.quicksum(forbidden_vars) <= len(forbidden_vars) - 1
                            )
                        if (
                            result["status_code"]
                            == Subsolver.DIFFERENT_TREATMENTS_NEEDED
                        ):
                            self.count_bad_cuts += 1
                            # forbid this exact treatment combination
                            logger.info("Different treatments needed")
                            logger.info("patients[d]: " + str(patients[d]))

                            forbidden_vars = []

                            for m in patients[d]:
                                for p in patients[d][m]:
                                    if patients[d][m][p] > 0:
                                        forbidden_vars.append(
                                            sum(
                                                self.x_pmdri[
                                                    p, m, d, patients[d][m][p], i
                                                ]
                                                for i in self.max_repetitions[m]
                                            )
                                        )
                                    if (p, m, d, patients[d][m][p] + 1) in self.x_pmdri:
                                        forbidden_vars.append(
                                            1
                                            - sum(
                                                self.x_pmdri[
                                                    p, m, d, patients[d][m][p] + 1, i
                                                ]
                                                for i in self.max_repetitions[m]
                                            )
                                        )

                            count_days_infeasible += 1
                            model.cbLazy(
                                gp.quicksum(forbidden_vars) <= len(forbidden_vars) - 1
                            )

                    logger.info(
                        f"Number of cuts added: {self.count_bad_cuts + self.count_good_cuts} (bad={self.count_bad_cuts}, good={self.count_good_cuts})"
                    )
                    logger.debug(
                        "All days finished with {} days infeasible".format(
                            count_days_infeasible
                        )
                    )
                    logger.info(
                        f"Unnecessary comps: ({self.count_unnecessary_comps} / {self.count_good_cuts})"
                    )
                    logger.info(f"Calls to is_day_infeasible: {self.subsolver.calls_to_solve_subsystem}/ {self.subsolver.calls_to_is_day_infeasible}")  # type: ignore
                    logger.info("Calls")
                except Exception as e:
                    logger.debug(str(e))
                    logger.exception("Error in solution callback")
                    raise Exception("Error in solution callback")

        self.model.optimize(_solution_callback)

        if (
            self.model.status == gp.GRB.OPTIMAL
            or self.model.status == gp.GRB.TIME_LIMIT
            or self.model.status == gp.GRB.INTERRUPTED
        ):
            logger.info("Optimal solution found.")
            logger.debug("Sub Objectives:")
            # logger.debug(f"(SOLVER):Minimize Treatments: {self.mt.X}")
            logger.debug(f"(SOLVER):Minimize Delay: {self.md.X}")
            logger.debug(f"(SOLVER):Minimize Missing Treatments: {self.mmt.X}")

            solution = self._extract_solution()
            return solution
        else:
            logger.info("No optimal solution found.")
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

    def _create_model(self):

        # Create the self.model
        self.model = gp.Model("PatientAdmissionScheduling")
        self.model.setParam("LogToConsole", int(self.log_to_console))  # type: ignore
        self.model.setParam("Threads", self.number_of_threads)  # type: ignore
        self.model.setParam("Cuts", 0)
        self.model.setParam("LazyConstraints", 1)
        # self.model.setParam("CutPasses", 3)
        self.model.setParam("NoRelHeurTime", self.no_rel_heur_time)  # type: ignore
        self._create_variables()
        self._create_constraints()

        if self.break_symmetry:  # type: ignore
            self._break_symmetry()
            pass

        if self.use_helper_constraints:  # type: ignore
            self._helper_constraints()

        self._set_optimization_goal()

    def _create_variables(self):
        #####################################
        # Create variables
        #####################################
        # max_number of treatments possible needed divided by min participants per treatment
        self.max_repetitions = {
            m: range(
                ceil(
                    sum(self.lr_pm[p, m] for p in self.P if m in self.M_p[p])
                    / self.j_m[m]
                )
            )
            for m in self.M
        }

        y_mdi_keys = []
        for m in self.M:
            y_mdi_keys.extend(product([m], self.D, self.max_repetitions[m]))
        self.y_mdi = self.model.addVars(y_mdi_keys, vtype=gp.GRB.BINARY, name="y_mdi")

        # Create x_pmdr
        x_pmdri_keys = []
        for p in self.P:
            for m in self.M_p[p]:

                x_pmdri_keys.extend(
                    product(
                        [p],
                        [m],
                        self.A_p[p],
                        self.BD_L_pm[p, m],
                        self.max_repetitions[m],
                    )
                )

        self.x_pmdri = self.model.addVars(
            x_pmdri_keys, vtype=gp.GRB.BINARY, name="x_pmdri"
        )

        # Create a_pd
        a_pd_keys = []
        for p in self.P:
            a_pd_keys.extend(product([p], self.D_p[p]))
        self.a_pd = self.model.addVars(a_pd_keys, vtype=gp.GRB.BINARY, name="a_pd")
        logger.debug("Variables created.")

    def _create_parameter_sets(self):
        super()._create_parameter_sets()

    def _break_symmetry(self):
        for d in self.D:
            for m in self.M:
                for i in self.max_repetitions[m]:
                    if i == 0:
                        continue
                    self.model.addConstr(
                        gp.quicksum(
                            self.x_pmdri[p, m, d, r, i]
                            for p in self.P
                            if m in self.M_p[p] and d in self.A_p[p]
                            for r in self.BD_L_pm[p, m]
                        )
                        <= gp.quicksum(
                            self.x_pmdri[p, m, d, r, i - 1]
                            for p in self.P
                            if m in self.M_p[p] and d in self.A_p[p]
                            for r in self.BD_L_pm[p, m]
                        ),
                        name=f"constraint_break_symmetry_m{m.id}_d{d}_i{i}",
                    )

    def _set_optimization_goal(self):

        minimize_delay = gp.quicksum(
            (d - min(self.A_p[p])) * self.a_pd[p, d]
            for p in self.P
            for d in self.D_p[p]
        )

        minimize_missing_treatment = gp.LinExpr()
        for p in self.P:
            # Sum up the number of all treatments for patient p
            scheduled_treatments = gp.quicksum(
                value
                for (p_prime, _, _, _, _), value in self.x_pmdri.items()
                if p_prime == p
            )

            # Get requested number of treatments for patient p if total stay was in the planning horizon
            total_treatments = sum(self.lr_pm[p, m] for m in self.M_p[p])

            minimize_missing_treatment += total_treatments - scheduled_treatments

        objective = (
            self.delay_value * minimize_delay  # type: ignore
            + self.missing_treatment_value * minimize_missing_treatment  # type: ignore
        )
        # Create variables for easier extraction of individual values
        self.mt = self.model.addVar(name="minimize_treatments", vtype=gp.GRB.CONTINUOUS)
        self.md = self.model.addVar(name="minimize_delay", vtype=gp.GRB.CONTINUOUS)
        self.mmt = self.model.addVar(
            name="minimize_missing_treatment", vtype=gp.GRB.CONTINUOUS
        )
        self.model.addConstr(self.md == self.delay_value * minimize_delay)
        self.model.addConstr(
            self.mmt == self.missing_treatment_value * minimize_missing_treatment
        )
        self.model.setObjective(objective, gp.GRB.MINIMIZE)

    def _create_constraints(self):
        self.model.update()
        for p, m, d, r, i in self.x_pmdri:
            if i >= 1:
                continue
            if r == 1:
                continue
            self.model.addConstr(
                gp.quicksum(
                    self.x_pmdri[p, m, d, r, i] for i in self.max_repetitions[m]
                )
                <= gp.quicksum(
                    self.x_pmdri[p, m, d, r - 1, i] for i in self.max_repetitions[m]
                ),
                name=f"constraint_x_pmdri_work_p{p.id}_m{m.id}_d{d}_r{r}",
            )

        # Ensure the i part of x_pmdri works
        for p in self.P:
            for m in self.M_p[p]:
                for d in self.A_p[p]:
                    for r in self.BD_L_pm[p, m]:
                        self.model.addConstr(
                            gp.quicksum(
                                self.x_pmdri[p, m, d, r, i]
                                for i in self.max_repetitions[m]
                            )
                            <= 1,
                            name=f"constraint_one_repetition_p{p.id}_m{m.id}_d{d}_r{r}",
                        )
        for p in self.P:
            for m in self.M_p[p]:
                for d in self.A_p[p]:
                    for i in self.max_repetitions[m]:
                        self.model.addConstr(
                            gp.quicksum(
                                self.x_pmdri[p, m, d, r, i] for r in self.BD_L_pm[p, m]
                            )
                            <= 1,
                            name=f"constraint_one_treatment_p{p.id}_m{m.id}_d{d}",
                        )

        # Ensure that y_mdi is only 1 if x_pmdri is 1 for some pr
        for m, d, i in self.y_mdi:
            for p in self.P:
                if d in self.A_p[p] and m in self.M_p[p]:

                    for r in self.BD_L_pm[p, m]:
                        self.model.addConstr(
                            self.x_pmdri[p, m, d, r, i] <= self.y_mdi[m, d, i],
                            name=f"constraint_y_mdi_p{p.id}_m{m.id}_d{d}_i{i}",
                        )

        # Every repetition has to have at most k_m patients
        for m in self.M:
            for i in self.max_repetitions[m]:
                for d in self.D:
                    num_participants = gp.quicksum(
                        self.x_pmdri[p, m, d, r, i]
                        for p in self.P
                        if m in self.M_p[p] and d in self.A_p[p]
                        for r in self.BD_L_pm[p, m]
                    )
                    self.model.addConstr(
                        num_participants <= self.k_m[m],
                        name=f"constraint_max_participants_m{m.id}_d{d}_i{i}",
                    )
                    if self.enforce_min_patients_per_treatment:  # type: ignore
                        self.model.addConstr(
                            num_participants >= self.j_m[m] * self.y_mdi[m, d, i],
                            name=f"constraint_min_participants_m{m.id}_d{d}_i{i}",
                        )

        # Constraint: Every patient is assigned to at most the required number of treatments
        for p in self.P:
            for m in self.M_p[p]:
                self.model.addConstr(
                    gp.quicksum(
                        self.x_pmdri[p, m, d, r, i]
                        for d in self.A_p[p]
                        for r in self.BD_L_pm[p, m]
                        for i in self.max_repetitions[m]
                    )
                    <= self.lr_pm[p, m],
                    name=f"constraint_not_too_many_p{p.id}_m{m.id}",
                )
        logger.debug("Constraint (not too many treatments) created.")

        # Constraint (p2): Patients not admitted have no treatments scheduled
        for p in self.P:
            for m in self.M_p[p]:
                for d in self.A_p[p]:
                    delta_set = [
                        delta for delta in self.D_p[p] if d - self.l_p[p] < delta <= d
                    ]
                    for r in self.BD_L_pm[p, m]:
                        for i in self.max_repetitions[m]:
                            self.model.addConstr(
                                self.x_pmdri[p, m, d, r, i]
                                <= gp.quicksum(
                                    self.a_pd[p, delta] for delta in delta_set
                                ),
                                name=f"constraint_outside_a_pd_p{p.id}_m{m.id}_d{d}_r{r}_i{i}",
                            )
        logger.debug("Constraint (no_treatments_outside admittance) created.")

        # Constraint: Bed usage constraint
        for d in self.D:
            term = gp.LinExpr()
            for p in self.P:
                delta_set = [
                    delta for delta in self.D_p[p] if d - self.l_p[p] < delta <= d
                ]
                term += gp.quicksum(self.a_pd[p, delta] for delta in delta_set)
            self.model.addConstr(term <= self.b, name=f"constraint_bed_usage_d{d}")
        logger.debug("Constraint (bed_usage) created.")

        # Every patient must be admitted
        for p in self.P:
            self.model.addConstr(
                gp.quicksum(self.a_pd[p, d] for d in self.D_p[p]) == 1,
                name=f"constraint_r2_b_p{p.id}",
            )
        logger.debug("Constraint (must be admitted) created.")
        if self.enforce_max_treatments_per_e_w:  # type: ignore
            # Constraint (a4): Max num of treatments over e_w days
            for p in self.P:
                # Skip patients that do not require even distribution as they stay fewer days than the
                # length of the rolling window
                if p.length_of_stay <= self.e_w:
                    continue
                for d in self.A_p[p]:
                    # check if the window is partially outside of patients stay => ignore
                    if (
                        d + self.e_w >= p.admitted_before_date.day + p.length_of_stay
                        or d + self.e_w > max(self.D) + 1
                    ):
                        continue

                    self.model.addConstr(
                        gp.quicksum(
                            self.x_pmdri[p, m, d_prime, r, i]
                            for m in self.M_p[p]
                            for d_prime in range(d, d + self.e_w)
                            for r in self.BD_L_pm[p, m]
                            for i in self.max_repetitions[m]
                        )
                        <= self.e_w_upper[p],
                        name=f"constraint_e_w_ub_p{p.id}_d{d}",
                    )
            logger.debug("Constraint (e_w_ub) created.")
        # Constraint (a4): Max num of treatments every day
        for p in self.P:
            for d in self.A_p[p]:
                self.model.addConstr(
                    gp.quicksum(
                        self.x_pmdri[p, m, d, r, i]
                        for m in self.M_p[p]
                        for r in self.BD_L_pm[p, m]
                        for i in self.max_repetitions[m]
                    )
                    <= self.daily_upper[p],
                    name=f"constraint_day_ub_p{p.id}_d{d}",
                )
        logger.debug("Constraint (day_ub) created.")
        # Constraint (a5): Min num of treatments every day
        if self.enforce_min_treatments_per_day:  # type: ignore
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
                                self.x_pmdri[p, m, d, r, i]
                                for m in self.M_p[p]
                                for r in self.BD_L_pm[p, m]
                                for i in self.max_repetitions[m]
                            )
                            >= self.daily_lower[p],
                            name=f"constraint_day_always_lb_p{p.id}_d{d}",
                        )
                    else:
                        # otherwise only enfore the lowerbound if admitted before
                        delta_set = [
                            delta
                            for delta in self.D_p[p]
                            if d - self.l_p[p] < delta <= d
                        ]

                        self.model.addConstr(
                            gp.quicksum(
                                self.x_pmdri[p, m, d, r, i]
                                for m in self.M_p[p]
                                for r in self.BD_L_pm[p, m]
                                for i in self.max_repetitions[m]
                            )
                            >= self.daily_lower[p]
                            * gp.quicksum(self.a_pd[p, delta] for delta in delta_set),
                            name=f"constraint_day_lb_p{p.id}_d{d}",
                        )
            logger.debug("Constraint (Day always) created.")
        # Constraint: Make sure that already admitted patients are admitted on the first day of the planning horizon
        for p in self.P:
            if p.already_admitted:
                self.model.addConstr(
                    self.a_pd[p, 0] == 1, name=f"constraint_already_admitted_p{p.id}"
                )
        logger.debug("Constraint (already_admitted) created.")

    def _helper_constraints(self):
        """Adds constraints to make sure that obvious constraints are met."""
        resource_groups = set()
        for f in self.F:
            for r in range(len(f.resource_groups)):
                resource_groups = resource_groups.union(
                    set(combinations(f.resource_groups, r + 1))
                )

        for d in self.D:
            # Count resource usage
            for resource_group in resource_groups:
                # count max number of available resources time slots
                avail_resources = 0
                for fhat in resource_group:
                    for f in self.fhat[fhat]:
                        avail_resources += sum(self.av_fdt[f, d, t] for t in self.T)

                all_treatments_require_rg = gp.LinExpr()
                for m in self.M:
                    for fhat in resource_group:
                        if fhat not in m.resources:
                            continue
                        all_treatments_require_rg += (
                            gp.quicksum(
                                self.y_mdi[m, d, i] for i in self.max_repetitions[m]
                            )
                            * self.n_fhatm[fhat, m]
                            * self.du_m[m]
                        )
                self.model.addConstr(
                    all_treatments_require_rg <= avail_resources,
                    name=f"constraint_enough_resource_group_{resource_group}_d{d}",
                )
        # Add combined resources groups when resources are in multiple groups
        pass

    def _extract_solution(self):
        # Print all variables if they are 1
        if True:
            for key in self.x_pmdri:
                if self.x_pmdri[key].X > 0.5:
                    logger.debug(f"x_pmdr{key} = {self.x_pmdri[key].X}")
            for key in self.a_pd:
                if self.a_pd[key].X > 0.5:
                    logger.debug(f"a_pd{key} = {self.a_pd[key].X}")

        # patients arrival
        patients_arrival = {}
        for p in self.P:
            for d in self.D_p[p]:
                if self.a_pd[p, d].X > 0.5:
                    patients_arrival[p] = DayHour(d, 0)
                    break

        # get dailys schedule
        patients = self.build_patients_dict(lambda x: x.X > 0.5)
        appointments = []
        patients = self.build_patients_dict(lambda x: x.X > 0.5)
        for d in self.D:
            day_solution = self.subsolver.get_day_solution(d, patients[d])
            appointments.extend(day_solution)

        solution = Solution(
            instance=self.instance,
            schedule=appointments,
            patients_arrival=patients_arrival,
            solver=self,
            solution_value=self.model.objVal,
        )
        return solution

    def create_subproblem(self):
        pass
