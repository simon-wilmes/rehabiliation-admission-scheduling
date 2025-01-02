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

from math import floor, gcd, ceil, prod, log
from pprint import pformat
from .subsolvers import Subsolver, CPSubsolver, CPSubsolver2
from itertools import product, combinations
from typing import Callable, Any
from time import time
from random import random
from copy import copy, deepcopy


class LBBDSolver(Solver):

    SOLVER_OPTIONS = Solver.BASE_SOLVER_OPTIONS.copy()
    SOLVER_OPTIONS.update(
        {
            "break_symmetry": [True, False],
            "use_helper_constraints": [True, False],
            "add_constraints_to_symmetric_days": [True, False],
        }
    )  # Add any additional options here

    SOLVER_DEFAULT_OPTIONS = {
        "break_symmetry": True,
        "subsolver_cls": CPSubsolver,
        "use_helper_constraints": True,
        "add_constraints_to_symmetric_days": True,
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

        if isinstance(self.subsolver_cls, str):  # type: ignore
            dict_subsolvers = {"CPSubsolver": CPSubsolver, "CPSubsolver2": CPSubsolver2}
            self.subsolver_cls = dict_subsolvers[self.subsolver_cls]

        self.subsolver: Subsolver = self.subsolver_cls(instance=instance, solver=self, **subsolver_kwargs)  # type: ignore

        if not self.add_constraints_to_symmetric_days:  # type: ignore
            self.subsolver.get_days_symmetric_to = lambda d: [d]  # type: ignore
        else:
            for d in self.D:
                assert d in self.subsolver.get_days_symmetric_to(
                    d
                ), f"Day {d} not in symmetric days"

        self.count_good_cuts = 0
        self.count_bad_cuts = 0
        self.already_seen_forbidden_strs = set()
        self.count_unnecessary_comps = 0
        self.time_is_feasible = 0
        self.time_total = 0
        self.num_constraints_added = 0
        self.all_cuts = []
        self.all_cuts_str = []

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
                            for i in self.J_md[m, d]
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
        self.num_solution_callbacks = 0

        def _solution_callback(model, where):

            if where == gp.GRB.Callback.MIPSOL:
                self.time_total -= time()
                self.num_solution_callbacks += 1
                # logger.debug("Solution callback")
                # Retrieve the solution
                # patients day assignments

                patients = self.build_patients_dict(
                    lambda x: model.cbGetSolution(x) > 0.5
                )

                # logger.debug("\n" + pformat(patients))
                count_days_infeasible = 0

                for d in self.D:
                    # logger.debug(f"Checking day {d}")
                    # logger.debug(f"Patients: {pformat(patients[d])}")
                    self.time_is_feasible -= time()
                    result = self.subsolver.is_day_infeasible(d, patients[d])  # type: ignore
                    self.time_is_feasible += time()

                    status_code = result["status_code"]

                    if status_code == Subsolver.TOO_MANY_TREATMENTS:
                        self.count_good_cuts += 1
                        # not is Feasible, generate cut
                        count_days_infeasible += 1
                        # Dict that maps (p, m) to the number of treatments
                        self._add_too_many_treatments(model, d, patients, result)

                    if status_code == Subsolver.MIN_PATIENTS_PROBLEM:
                        self.count_bad_cuts += 1
                        count_days_infeasible += 1
                        # forbid this exact treatment combination
                        # logger.info("Different treatments needed")
                        # logger.info("patients[d]: " + str(patients[d]))
                        self._add_min_patients_violated(model, d, patients)

                self.time_total += time()

                if self.num_solution_callbacks % 100 == 0:
                    self._print_subsolver_stats()

        self.model.optimize(_solution_callback)

        logger.debug("Finished optimization")
        logger.debug("Subsolver stats:")
        self._print_subsolver_stats()

        if (
            self.model.status == gp.GRB.OPTIMAL
            or self.model.status == gp.GRB.TIME_LIMIT
            or self.model.status == gp.GRB.INTERRUPTED
        ):
            logger.info("Optimal solution found.")
            logger.debug("Sub Objectives:")
            logger.debug(f"(SOLVER):Minimize Delay: {self.md.X}")
            logger.debug(f"(SOLVER):Minimize Missing Treatments: {self.mmt.X}")

            solution = self._extract_solution()
            return solution
        else:
            logger.info("No optimal solution found.")
            return NO_SOLUTION_FOUND

    def _add_min_patients_violated(self, model, d: int, patients: list[dict]):
        forbidden_vars = []
        for m in patients[d]:
            for p in patients[d][m]:
                if patients[d][m][p] > 0:
                    forbidden_vars.append(
                        sum(
                            self.x_pmdri[p, m, d, patients[d][m][p], i]
                            for i in self.J_md[m, d]
                        )
                    )
                if (p, m, d, patients[d][m][p] + 1) in self.x_pmdri:
                    forbidden_vars.append(
                        1
                        - sum(
                            self.x_pmdri[p, m, d, patients[d][m][p] + 1, i]
                            for i in self.J_md[m, d]
                        )
                    )

        model.cbLazy(gp.quicksum(forbidden_vars) <= len(forbidden_vars) - 1)

    def _print_subsolver_stats(self):
        try:
            logger.info(
                f"Number of cuts added: {self.count_bad_cuts + self.count_good_cuts} (bad={self.count_bad_cuts}, good={self.count_good_cuts})"
            )
            logger.info(
                f"Unnecessary comps: ({self.count_unnecessary_comps} / {self.count_good_cuts})"
            )
            logger.info(
                f"Calls to is_day_infeasible: cp_solver={self.subsolver.calls_to_solve_subsystem}/ total={self.subsolver.calls_to_is_day_infeasible} stored={self.subsolver.calls_to_is_day_infeasible - self.subsolver.calls_to_solve_subsystem}"
            )  # type: ignor
            logger.info(
                f"Is feasible: {self.time_is_feasible} {self.time_is_feasible / self.time_total}"
            )
            logger.info(
                "Subsolver: Solve Model: {}/{}".format(
                    self.subsolver.time_solve_model,  # type: ignore
                    self.subsolver.time_solve_model  # type: ignore
                    / (  # type: ignore
                        self.subsolver.time_create_model  # type: ignore
                        + self.subsolver.time_solve_model  # type: ignore
                    ),  # type: ignore
                )  # type: ignore
            )  # type: ignore
            logger.info(  # type: ignore
                "Subsolver: Create Model: {}/{}".format(  # type: ignore
                    self.subsolver.time_create_model,  # type: ignore
                    self.subsolver.time_create_model  # type: ignore
                    / (  # type: ignore
                        self.subsolver.time_create_model  # type: ignore
                        + self.subsolver.time_solve_model  # type: ignore
                    ),
                )
            )
            logger.info(f"Num_constraints-added: {self.num_constraints_added}")
        except Exception as e:
            logger.info("Debug Error in print_subsolver_stats")
            logger.error(e)

    def _add_too_many_treatments(
        self, model, d: int, patients: list[dict], result: dict
    ):
        patients_mem = defaultdict(dict)
        for m in patients[d]:
            for p in patients[d][m]:
                if patients[d][m][p] > 0:
                    patients_mem[p][m] = patients[d][m][p]

        if "late_scheduled" in result:
            late_scheduled = result["late_scheduled"]
            min_cut = patients_mem

            for (m, p), num in late_scheduled.items():
                min_cut[p][m] -= num

            # min_cut is the minimal infeasible set if any of the late scheduled treatments are added
            for m_added, p_added in late_scheduled:
                for d_prime in self.subsolver.get_days_symmetric_to(d):
                    forbidden_vars = []
                    abort = False
                    for m in patients[d]:
                        for p in patients[d][m]:
                            if min_cut[p][m] > 0:
                                if d_prime not in self.A_p[p]:
                                    abort = True
                                    break

                                # logger.info(
                                #    f"{p},{m},{d_prime},{(min_cut[p][m] if p != p_added or m != m_added else min_cut[p][m] + 1)}"
                                # )
                                forbidden_vars.append(
                                    sum(
                                        self.x_pmdri[
                                            p,
                                            m,
                                            d_prime,
                                            (
                                                min_cut[p][m]
                                                if p != p_added or m != m_added
                                                else min_cut[p][m] + 1
                                            ),
                                            i,
                                        ]
                                        for i in self.J_md[m, d_prime]
                                    )
                                )

                        if abort:
                            break
                    if not abort:
                        logger.info(
                            f"d_prime: {d_prime} d:{d} m: {m_added} p: {p_added}"
                        )
                        model.cbLazy(
                            gp.quicksum(forbidden_vars) <= len(forbidden_vars) - 1
                        )
                        self.num_constraints_added += 1
        else:
            for d_prime in self.subsolver.get_days_symmetric_to(d):
                forbidden_vars = []
                abort = False
                for m in patients[d]:
                    for p in patients[d][m]:
                        if patients[d][m][p] > 0:
                            if d_prime not in self.A_p[p]:
                                abort = True
                                break
                            forbidden_vars.append(
                                sum(
                                    self.x_pmdri[
                                        p,
                                        m,
                                        d_prime,
                                        patients[d][m][p],
                                        i,
                                    ]
                                    for i in self.J_md[m, d_prime]
                                )
                            )

                    if abort:
                        break
                if not abort:
                    logger.info("d_prime: " + str(d_prime) + " d: " + str(d))
                    model.cbLazy(gp.quicksum(forbidden_vars) <= len(forbidden_vars) - 1)
                    self.num_constraints_added += 1

            self.all_cuts.append(deepcopy(patients_mem))
            patients_copy = list(
                map(
                    lambda x: x[0],
                    sorted(
                        [
                            (
                                {m.id: patients_mem[p][m] for m in patients_mem[p]},
                                sum(patients_mem[p].values()),
                            )
                            for p in patients_mem
                        ],
                        key=lambda x: x[1],
                    ),
                )
            )

            self.all_cuts_str.append(f"({''.join(str(patients_copy).split())})")
            forbidden_vars_str = f"({''.join(str(patients_copy).split())})"

            logger.info("forbidden_vars: " + forbidden_vars_str)
            logger.info(forbidden_vars_str in self.already_seen_forbidden_strs)
            self.count_unnecessary_comps += (
                forbidden_vars_str in self.already_seen_forbidden_strs
            )
            self.already_seen_forbidden_strs.add(forbidden_vars_str)

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

        y_mdi_keys = []
        for m in self.M:
            for d in self.D:
                y_mdi_keys.extend(product([m], [d], self.J_md[m, d]))
        self.y_mdi = self.model.addVars(y_mdi_keys, vtype=gp.GRB.BINARY, name="y_mdi")

        # Create x_pmdr
        x_pmdri_keys = []
        for p in self.P:
            for m in self.M_p[p]:
                for d in self.A_p[p]:
                    x_pmdri_keys.extend(
                        product(
                            [p],
                            [m],
                            [d],
                            self.BD_L_pm[p, m],
                            self.J_md[m, d],
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
                for i in self.J_md[m, d]:
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
            (d - min(self.D_p[p])) * self.a_pd[p, d]
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
                gp.quicksum(self.x_pmdri[p, m, d, r, i] for i in self.J_md[m, d])
                <= gp.quicksum(
                    self.x_pmdri[p, m, d, r - 1, i] for i in self.J_md[m, d]
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
                                self.x_pmdri[p, m, d, r, i] for i in self.J_md[m, d]
                            )
                            <= 1,
                            name=f"constraint_one_repetition_p{p.id}_m{m.id}_d{d}_r{r}",
                        )
        for p in self.P:
            for m in self.M_p[p]:
                for d in self.A_p[p]:
                    for i in self.J_md[m, d]:
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
            for d in self.D:
                for i in self.J_md[m, d]:
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
                        for i in self.J_md[m, d]
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
                        for i in self.J_md[m, d]:
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

                    self.model.addConstr(
                        gp.quicksum(
                            self.x_pmdri[p, m, d_prime, r, i]
                            for m in self.M_p[p]
                            for d_prime in range(d, d + self.e_w)
                            if d_prime in self.A_p[p]
                            for r in self.BD_L_pm[p, m]
                            for i in self.J_md[m, d_prime]
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
                        for i in self.J_md[m, d]
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
                                for i in self.J_md[m, d]
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
                                for i in self.J_md[m, d]
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
        # Add combined resources groups when resources are in multiple groups
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
                            gp.quicksum(self.y_mdi[m, d, i] for i in self.J_md[m, d])
                            * self.n_fhatm[fhat, m]
                            * self.du_m[m]
                        )
                self.model.addConstr(
                    all_treatments_require_rg <= avail_resources,
                    name=f"constraint_enough_resource_group_{resource_group}_d{d}",
                )

        # make sure that the daily number of treaments do not exceed the daily time upper bound for every patient may or may not be irrelevent
        max_treatment_length_p = {
            p: max(self.du_m[m] for m in self.M_p[p]) for p in self.P
        }
        for p in self.P:
            if len(self.T) / self.daily_upper[p] > max_treatment_length_p[p]:
                continue

            for d in self.A_p[p]:
                expr = gp.LinExpr()
                for m in self.M_p[p]:

                    expr += self.du_m[m] * gp.quicksum(
                        self.x_pmdri[p, m, d, r, i]
                        for r in self.BD_L_pm[p, m]
                        for i in self.J_md[m, d]
                    )
                self.model.addConstr(
                    expr <= len(self.T),
                    name=f"constraint_daily_time_upper_bound_p{p.id}_d{d}",
                )

        pass

    def _extract_solution(self):
        # Print all variables if they are 1
        if False:
            for key in self.x_pmdri:
                if self.x_pmdri[key].X > 0.5:
                    logger.info(f"x_pmdr{key} = {self.x_pmdri[key].X}")
            for key in self.a_pd:
                if self.a_pd[key].X > 0.5:
                    logger.info(f"a_pd{key} = {self.a_pd[key].X}")

        logger.info(f"All cuts {self.all_cuts}")
        logger.info(f"All cuts {self.all_cuts_str}")
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

    def generate_unique_tuples(self, lists):
        num_solutions = 1000
        worst_case = prod(len(sublist) for sublist in lists)
        prop = (num_solutions / worst_case) ** (1 / len(lists))

        def backtrack(current_tuple, used_elements, depth):
            # Base case: If the tuple is complete, yield it
            if depth == len(lists):
                yield tuple(current_tuple)
                return

            # Iterate through the current list at 'depth'
            for num in lists[depth]:
                if (
                    num not in used_elements and random() < prop
                ):  # Check if num causes duplicates
                    # Include num in the current tuple and mark it as used
                    current_tuple.append(num)
                    used_elements.add(num)
                    # Recur to the next depth
                    if random() < 0.1:
                        yield from backtrack(current_tuple, used_elements, depth + 1)

                    # Backtrack: remove num and unmark it
                    current_tuple.pop()
                    used_elements.remove(num)

        # Start backtracking from depth 0
        return backtrack([], set(), 0)
