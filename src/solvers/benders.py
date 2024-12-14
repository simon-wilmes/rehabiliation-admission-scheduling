from src.solvers.solver import Solver
import gurobipy as gp

from datetime import timedelta

from collections import defaultdict
from src.logging import logger
from src.instance import Instance

from src.solution import Solution, Appointment
from src.solution import NO_SOLUTION_FOUND

from copy import copy
from math import floor, gcd
from pprint import pformat
from .subsolvers import Subsolver, CPSubsolver
from itertools import product


class LBBDSolver(Solver):

    SOLVER_OPTIONS = Solver.BASE_SOLVER_OPTIONS.copy()
    SOLVER_OPTIONS.update(
        {
            "break_symmetry": [True, False],
        }
    )  # Add any additional options here

    SOLVER_DEFAULT_OPTIONS = {
        "break_symmetry": True,
        "subsolver_cls": CPSubsolver,
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

        # Add min_patients_per_treatment to subsolver kwargs
        subsolver_kwargs["enforce_min_patients_per_treatment"] = self.enforce_min_patients_per_treatment  # type: ignore

        self.subsolver: Subsolver = self.subsolver_cls(instance=instance, solver=self, **subsolver_kwargs)  # type: ignore

    def _solve_model(self):
        # Define the callback for creating subproblems
        def _solution_callback(model, where):
            if where == gp.GRB.Callback.MIPSOL:
                try:
                    logger.debug("Solution callback")
                    # Retrieve the solution
                    # patients day assignments
                    patients = [{m: {} for m in self.M} for _ in self.D]
                    index = 0
                    for p in self.P:
                        for m in self.M_p[p]:
                            for d in self.A_p[p]:
                                for r in self.L_pm[p, m]:
                                    if (
                                        not model.cbGetSolution(self.x_pmdr[p, m, d, r])
                                        > 0.5
                                    ):
                                        break
                                if r > 0:
                                    patients[d][m][p] = r

                    # Day treatment assignments
                    appointments = [{} for _ in self.D]
                    for m in self.M:
                        for d in self.D:
                            for i in self.I_m[m]:
                                if not model.cbGetSolution(self.y_mdi[m, d, i]) > 0.5:
                                    break
                            if i > 0:
                                appointments[d][m] = i

                    logger.debug("Subproblem created.")
                    logger.debug("\n" + pformat(patients))
                    logger.debug("\n" + pformat(appointments))
                    for d in self.D:
                        # d = 1
                        result = self.subsolver.is_day_infeasible(d, appointments[d], patients[d])  # type: ignore
                        status_code = result["status_code"]
                        if result["status_code"] in [Subsolver.COMPLETELY_INFEASIBLE]:
                            # not is Feasible, generate cut
                            logger.debug("Subproblem not feasible.")
                            logger.debug("Add cut")
                            forbidden_vars = [
                                self.x_pmdr[p, m, d, patients[d][m][p]]
                                for m in patients[d]
                                for p in patients[d][m]
                                if patients[d][m][p] > 0
                            ]
                            model.cbLazy(
                                gp.quicksum(forbidden_vars) <= len(forbidden_vars) - 1
                            )
                except:
                    logger.exception("Error in solution callback")

        self.model.optimize(_solution_callback)
        if (
            self.model.status == gp.GRB.OPTIMAL
            or self.model.status == gp.GRB.TIME_LIMIT
            or self.model.status == gp.GRB.INTERRUPTED
        ):
            logger.info("Optimal solution found.")
            logger.debug("Sub Objectives:")
            logger.debug(f"(SOLVER):Minimize Treatments: {self.mt.X}")
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
        self._create_parameter_sets()
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

        self._helper_constraints()

        self._set_optimization_goal()

    def _create_variables(self):
        #####################################
        # Create variables
        #####################################
        # Create x_pmdr
        x_pmdr_keys = []
        for p in self.P:
            for m in self.M_p[p]:
                x_pmdr_keys.extend(product([p], [m], self.A_p[p], self.L_pm[p, m]))

        self.x_pmdr = self.model.addVars(
            x_pmdr_keys, vtype=gp.GRB.BINARY, name="x_pmdr"
        )

        # Create y_mdi
        y_mdi_keys = []
        for m in self.M:
            y_mdi_keys.extend(product([m], self.D, self.I_m[m]))

        self.y_mdi = self.model.addVars(y_mdi_keys, vtype=gp.GRB.BINARY, name="y_mdi")

        # Create a_pd
        a_pd_keys = []
        for p in self.P:
            a_pd_keys.extend(product([p], self.D_p[p]))
        self.a_pd = self.model.addVars(a_pd_keys, vtype=gp.GRB.BINARY, name="a_pd")
        logger.debug("Variables created.")

    def _create_parameter_sets(self):
        super()._create_parameter_sets()

    def _break_symmetry(self):
        for m, d, i in self.y_mdi:
            if i == 0:
                continue
            self.model.addConstr(
                self.y_mdi[m, d, i] <= self.y_mdi[m, d, i - 1],
                name=f"symmetry_breaking_y_mdi_{m}_{d}_{i}",
            )

        for p, m, d, r in self.x_pmdr:
            if r == 0:
                continue
            self.model.addConstr(
                self.x_pmdr[p, m, d, r] <= self.x_pmdr[p, m, d, r - 1],
                name=f"symmetry_breaking_x_pmdr_{p}_{m}_{d}_{r}",
            )

    def _set_optimization_goal(self):

        minimize_treatments = gp.quicksum(
            self.y_mdi[m, d, i] for m in self.M for i in self.I_m[m] for d in self.D
        )

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
                for (p_prime, _, _, _), value in self.x_pmdr.items()
                if p_prime == p
            )

            # Get requested number of treatments for patient p if total stay was in the planning horizon
            total_treatments = sum(self.lr_pm[p, m] for m in self.M_p[p])

            minimize_missing_treatment += total_treatments - scheduled_treatments

        objective = (
            self.treatment_value * minimize_treatments  # type: ignore
            + self.delay_value * minimize_delay  # type: ignore
            + self.missing_treatment_value * minimize_missing_treatment  # type: ignore
        )
        # Create variables for easier extraction of individual values
        self.mt = self.model.addVar(name="minimize_treatments", vtype=gp.GRB.CONTINUOUS)
        self.md = self.model.addVar(name="minimize_delay", vtype=gp.GRB.CONTINUOUS)
        self.mmt = self.model.addVar(
            name="minimize_missing_treatment", vtype=gp.GRB.CONTINUOUS
        )

        self.model.addConstr(self.mt == self.treatment_value * minimize_treatments)
        self.model.addConstr(self.md == self.delay_value * minimize_delay)
        self.model.addConstr(
            self.mmt == self.missing_treatment_value * minimize_missing_treatment
        )
        self.model.setObjective(objective, gp.GRB.MINIMIZE)

    def _create_constraints(self):
        self.model.update()
        # Constraint: Every patient is assigned to at most the required number of treatments
        for p in self.P:
            for m in self.M_p[p]:
                self.model.addConstr(
                    gp.quicksum(
                        self.x_pmdr[p, m, d, r]
                        for d in self.A_p[p]
                        for r in self.L_pm[p, m]
                    )
                    == self.lr_pm[p, m],
                    name=f"constraint_p1_b_p{p.id}_m{m.id}",
                )
        logger.debug("Constraint (p1) created.")

        # Constraint: Bed usage constraint
        for d in self.D:
            term = gp.LinExpr()
            for p in self.P:
                delta_set = [
                    delta for delta in self.D_p[p] if d - self.l_p[p] < delta <= d
                ]
                term += gp.quicksum(self.a_pd[p, delta] for delta in delta_set)
            self.model.addConstr(term <= self.b, name=f"constraint_r1_b_d{d}")
        logger.debug("Constraint (r1) created.")

        # Every patient must be admitted
        for p in self.P:
            self.model.addConstr(
                gp.quicksum(self.a_pd[p, d] for d in self.D_p[p]) == 1,
                name=f"constraint_r2_b_p{p.id}",
            )
        logger.debug("Constraint (r2) created.")
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
                            self.x_pmdr[p, m, d_prime, r]
                            for m in self.M_p[p]
                            for d_prime in range(d, d + self.e_w)
                            for r in self.L_pm[p, m]
                        )
                        <= self.e_w_upper[p],
                        name=f"constraint_a3_ub_p{p.id}_d{d}",
                    )
            logger.debug("Constraint (a3) created.")
        # Constraint (a4): Max num of treatments every day
        for p in self.P:
            for d in self.A_p[p]:
                self.model.addConstr(
                    gp.quicksum(
                        self.x_pmdr[p, m, d, r]
                        for m in self.M_p[p]
                        for r in self.L_pm[p, m]
                    )
                    <= self.daily_upper[p],
                    name=f"constraint_a4_ub_p{p.id}_d{d}",
                )

        # Constraint (a5): Min num of treatments every day
        if self.enforce_min_treatments_per_day:  # type: ignore
            for p in self.P:
                for d in self.A_p[p]:
                    self.model.addConstr(
                        gp.quicksum(
                            self.x_pmdr[p, m, d, r]
                            for m in self.M_p[p]
                            for r in self.L_pm[p, m]
                        )
                        >= self.daily_lower[p],
                        name=f"constraint_a5_lb_p{p.id}_d{d}",
                    )
            logger.debug("Constraint (a5) created.")

        # Constraint: Make sure that already admitted patients are admitted on the first day of the planning horizon
        for p in self.P:
            if p.already_admitted:
                self.model.addConstr(
                    self.a_pd[p, 0] == 1, name=f"constraint_already_admitted_p{p.id}"
                )
        logger.debug("Constraint (already_admitted) created.")

        # Constraint: Total treatments for patient p is less than or equal to the number of treatments requested
        for p in self.P:
            for m in self.M_p[p]:

                self.model.addConstr(
                    gp.quicksum(
                        self.x_pmdr[p, m, d, r]
                        for d in self.A_p[p]
                        for r in self.L_pm[p, m]
                    )
                    == self.lr_pm[p, m],
                    name=f"constraint_m1_b_m{m.id}_d{d}",
                )
                pass
        logger.debug("Constraint (m1) created.")

        # Constraint: Total number of treatment repetitions is less than |I_m|
        for m in self.M:
            self.model.addConstr(
                gp.quicksum(self.y_mdi[m, d, i] for d in self.D for i in self.I_m[m])
                <= len(self.I_m[m]),
                name=f"constraint_m2_b_m{m.id}_d{d}",
            )
            pass
        logger.debug("Constraint (m2) created.")

    def _helper_constraints(self):
        """Adds constraints to make sure that obvious constraints are met."""

        # Constraint: Treatment is scheduled if any patient is assigned to it
        # sum of patients that day at most number of treatments divided by capacity of treatment
        for m in self.M:
            for d in self.D:

                self.model.addConstr(
                    self.k_m[m] * gp.quicksum(self.y_mdi[m, d, i] for i in self.I_m[m])
                    >= gp.quicksum(
                        self.x_pmdr[p, m, d, r]
                        for p in self.P
                        if m in self.M_p[p]
                        for r in self.L_pm[p, m]
                    ),
                    name=f"constraint_m2_b_m{m.id}_d{d}",
                )

    def _extract_solution(self):
        # Print all variables if they are 1
        if True:
            for key in self.y_mdi:
                if self.y_mdi[key].X > 0.5:
                    logger.debug(f"y_mdi{key} = {self.y_mdi[key].X}")
            for key in self.x_pmdr:
                if self.x_pmdr[key].X > 0.5:
                    logger.debug(f"x_pmdr{key} = {self.x_pmdr[key].X}")
            for key in self.a_pd:
                if self.a_pd[key].X > 0.5:
                    logger.debug(f"a_pd{key} = {self.a_pd[key].X}")

        solution = Solution(
            instance=self.instance,
            schedule=appointments,
            patients_arrival=patients_arrival,
            solver=self,
            solution_value=solver.objective_value,
        )
        return solution

    def create_subproblem(self):
        pass
