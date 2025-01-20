from src.instance import Instance
import gurobipy as gp
from src.logging import logger, print
from math import ceil
from itertools import product

from src.solution import Solution, NO_SOLUTION_FOUND, Appointment
from src.solvers.solver import Solver
from src.time import DayHour
from src.patients import Patient
from src.utils import slice_dict
from collections import defaultdict
from math import floor, ceil


class MIPSolver4(Solver):
    SOLVER_OPTIONS = Solver.BASE_SOLVER_OPTIONS.copy()
    SOLVER_OPTIONS.update(
        {"break_symmetry": [True, False]}
    )  # Add any additional options here

    SOLVER_DEFAULT_OPTIONS = {
        "break_symmetry": False,
    }

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
        self._create_parameter_sets()

    def _solve_model(self):
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

            # return NO_SOLUTION_FOUND
            # Compute IIS
            self.model.computeIIS()

            # Print IIS
            logger.debug("\nIIS Constraints:")
            for c in self.model.getConstrs():
                if c.IISConstr:
                    print(f"\t{c.constrname}: {self.model.getRow(c)} {c.Sense} {c.RHS}")  # type: ignore

            for v in self.model.getVars():
                if v.IISLB:
                    print(f"\t{v.varname} ≥ {v.LB}")  # type: ignore
                if v.IISUB:
                    print(f"\t{v.varname} ≤ {v.UB}")  # type: ignore
            return NO_SOLUTION_FOUND

    def _create_model(self):

        self.model = gp.Model("UpdatedMIP")
        self.model.setParam("LogToConsole", int(self.log_to_console))  # type: ignore
        self.model.setParam("Threads", self.number_of_threads)  # type: ignore
        # self.model.setParam("NoRelHeurTime", self.no_rel_heur_time)  # type: ignore
        self._create_variables()
        self._create_constraints()
        if self.break_symmetry:  # type: ignore
            self._break_symmetry()

        self._set_optimization_goal()
        self.model = self.model

    def _break_symmetry(self):
        return
        if self.break_symmetry_strong:  # type: ignore
            for m in self.M:
                for i in self.J_m[m]:
                    if i == 0:
                        continue
                    for d in self.D:
                        prev = gp.quicksum(
                            self.x_midt[m, i - 1, d_prime, t]
                            for d_prime in range(d + 1)
                            for t in self.T
                        )
                        succ = gp.quicksum(self.x_midt[m, i, d, t] for t in self.T)
                        self.model.addConstr(prev >= succ)

            logger.debug("Constraint (strong symmetry) created.")

        for m in self.M:
            for i in self.J_m[m]:
                if i == 0:
                    continue
                self.model.addConstr(
                    gp.quicksum(
                        self.x_midt[m, i, d, t] for d, t in product(self.D, self.T)
                    )
                    <= gp.quicksum(
                        self.x_midt[m, i - 1, d, t] for d, t in product(self.D, self.T)
                    )
                )
        logger.debug("Constraint (symmetry) created.")

    def _create_variables(self):
        self.x_midt = self.model.addVars(
            (
                (m, i, d, t)
                for m in self.M
                for d in self.D
                for i in self.J_md[m, d]
                for t in self.T
            ),  # type: ignore
            vtype=gp.GRB.BINARY,
            name="x_midt",
        )

        self.a_pd = self.model.addVars(
            ((p, d) for p in self.P for d in self.D_p[p]),  # type: ignore
            vtype=gp.GRB.BINARY,
            name="a_pd",
        )

        self.y_pmidt = self.model.addVars(
            ((p, m, i, d, t) for p in self.P for m in self.M_p[p] for d in self.A_p[p] for i in self.J_md[m, d] for t in self.T),  # type: ignore
            vtype=gp.GRB.BINARY,
            name="y_pmidt",
        )

        self.z_gfmidt = self.model.addVars(
            ((fhat, f, m, i, d, t) for m in self.M for fhat in self.Fhat_m[m] for f in self.fhat[fhat] for d in self.D for i in self.J_md[m, d] for t in self.T),  # type: ignore
            vtype=gp.GRB.BINARY,
            name="z_fmidt",
        )
        logger.debug("Variables created.")

    def _create_constraints(self):
        self.model.update()

        # Constraint: Every patient is assigned to at most the required number of treatments
        for p in self.P:
            for m in self.M_p[p]:
                self.model.addConstr(
                    gp.quicksum(
                        self.y_pmidt[p, m, i, d, t]
                        for d in self.A_p[p]
                        for i in self.J_md[m, d]
                        for t in self.T
                    )
                    <= self.lr_pm[p, m],
                    name=f"constraint_p1_b_p{p.id}_m{m.id}",
                )
        logger.debug("Constraint (p1) created.")

        # Constraint: Only treatments scheduled if addmitted
        for p in self.P:
            for m in self.M_p[p]:
                for d in self.A_p[p]:
                    delta_set = [
                        delta for delta in self.D_p[p] if d - self.l_p[p] < delta <= d
                    ]
                    for i, t in product(self.J_md[m, d], self.T):
                        self.model.addConstr(
                            self.y_pmidt[p, m, i, d, t]
                            <= gp.quicksum(self.a_pd[p, delta] for delta in delta_set),
                            name=f"constraint_m_when_admitted_p{p.id}_m{m.id}_i{i}_d{d}_t{t}",
                        )
        logger.debug("Constraint (when_admitted) created.")

        # Constraint: Only one treatment every timeslot per patient
        for p in self.P:
            for d, t in product(self.A_p[p], self.T):
                constr = gp.LinExpr()
                for m in self.M_p[p]:
                    tau_set = [
                        tau
                        for tau in self.T
                        if t - self.du_m[m] * self.instance.time_slot_length.hours
                        < tau
                        <= t
                    ]

                    constr += gp.quicksum(
                        self.y_pmidt[p, m, i, d, tau]
                        for i in self.J_md[m, d]
                        for tau in tau_set
                    )

                self.model.addConstr(
                    constr <= 1,
                    name=f"constraint_p3_b_p{p.id}_d{d}_t{t}",
                )
        logger.debug("Constraint (p3) created.")
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

        # Constraint: Resource usage constraint

        for f in self.F:
            for d, t in product(self.D, self.T):
                self.model.addConstr(
                    gp.quicksum(
                        self.z_gfmidt[fhat, f, m, i, d, tau]
                        for fhat in f.resource_groups
                        for m in self.M_fhat[fhat]
                        for i in self.J_md[m, d]
                        for tau in self.T
                        if t - self.du_m[m] * self.instance.time_slot_length.hours
                        < tau
                        <= t
                    )
                    <= int(f.is_available(DayHour(d, t))),
                    name=f"constraint_r2_b_f{f.id}_d{d}_t{t}",
                )
        logger.debug("Constraint (r2) created.")

        # Constraint: Treatment has resources
        for m in self.M:
            for d, t, fhat in product(self.D, self.T, self.Fhat_m[m]):
                for i in self.J_md[m, d]:

                    self.model.addConstr(
                        gp.quicksum(
                            self.z_gfmidt[fhat, f, m, i, d, t] for f in self.fhat[fhat]
                        )
                        == self.n_fhatm[fhat, m] * self.x_midt[m, i, d, t],
                        name=f"constraint_r3_b_m{m.id}_i{i}_fhat{fhat.id}",
                    )
        logger.debug("Constraint (r3) created.")

        # Constraint: Treatment must be scheduled if patient is assigned
        for p in self.P:
            for m in self.M_p[p]:
                for d, t in product(self.A_p[p], self.T):
                    for i in self.J_md[m, d]:
                        self.model.addConstr(
                            self.y_pmidt[p, m, i, d, t] <= self.x_midt[m, i, d, t],
                            name=f"constraint_r4_b_m{m.id}_i{i}",
                        )
        logger.debug("Constraint (r4) created.")

        # Constraint: every repetition is only scheduled once
        for m in self.M:
            for d in self.D:
                for i in self.J_md[m, d]:
                    self.model.addConstr(
                        gp.quicksum(self.x_midt[m, i, d, t] for t in self.T) <= 1,
                        name=f"constraint_r5_b_m{m.id}_i{i}",
                    )
        logger.debug("Constraint (r5) created.")
        # Constraint every session has at most k_m patients assigned
        for m in self.M:
            for d, t in product(self.D, self.T):
                for i in self.J_md[m, d]:

                    self.model.addConstr(
                        gp.quicksum(
                            self.y_pmidt[p, m, i, d, t]
                            for p in self.P
                            if m in self.M_p[p] and d in self.A_p[p]
                        )
                        <= self.k_m[m] * self.x_midt[m, i, d, t],
                        name=f"constraint_r6_b_m{m.id}_i{i}",
                    )
                # Constraint every session has at most k_m patients assigned
        for m in self.M:
            for d, t in product(self.D, self.T):
                for i in self.J_md[m, d]:
                    self.model.addConstr(
                        gp.quicksum(
                            self.y_pmidt[p, m, i, d, t]
                            for p in self.P
                            if m in self.M_p[p] and d in self.A_p[p]
                        )
                        >= self.j_m[m] * self.x_midt[m, i, d, t],
                        name=f"constraint_r6_b_m{m.id}_i{i}",
                    )
        logger.debug("Constraint (r6) created.")
        # Every patient must be admitted
        for p in self.P:
            self.model.addConstr(
                gp.quicksum(self.a_pd[p, d] for d in self.D_p[p]) == 1,
                name=f"constraint_r7_b_p{p.id}",
            )
        logger.debug("Constraint (r7) created.")

        # Constraint (a4): Max num of treatments over e_w days
        for p in self.P:
            # Skip patients that do not require even distribution as they stay fewer days than the
            # length of the rolling window
            if p.length_of_stay <= self.e_w:
                continue
            for d in self.A_p[p]:
                # check if the window is partially outside of patients stay => ignore
                viable_days_for_treatment = set(range(d, d + self.e_w)) & set(
                    self.A_p[p]
                )
                self.model.addConstr(
                    gp.quicksum(
                        self.y_pmidt[p, m, i, d_prime, t]
                        for m in self.M_p[p]
                        for d_prime in viable_days_for_treatment
                        for i in self.J_md[m, d_prime]
                        for t in self.T
                    )
                    <= self.e_w_upper[p],
                    name=f"constraint_a4_ub_p{p.id}_d{d}",
                )
        logger.debug("Constraint (a4) created.")

        # Constraint (a5): Max num of treatments every day
        for p in self.P:
            for d in self.A_p[p]:
                self.model.addConstr(
                    gp.quicksum(
                        self.y_pmidt[p, m, i, d, t]
                        for m in self.M_p[p]
                        for i in self.J_md[m, d]
                        for t in self.T
                    )
                    <= self.daily_upper[p],
                    name=f"constraint_a5_ub_p{p.id}_d{d}",
                )
        logger.debug("Constraint (a5) created.")

        for p in self.P:
            for d in self.A_p[p]:
                # otherwise only enfore the lowerbound if admitted before
                delta_set = [
                    delta for delta in self.D_p[p] if d - self.l_p[p] < delta <= d
                ]
                self.model.addConstr(
                    gp.quicksum(
                        self.y_pmidt[p, m, i, d, t]
                        for m in self.M_p[p]
                        for i in self.J_md[m, d]
                        for t in self.T
                    )
                    >= self.daily_lower[p]
                    * gp.quicksum(self.a_pd[p, delta] for delta in delta_set),
                    name=f"constraint_a5_ub_p{p.id}_d{d}",
                )
        logger.debug("Constraint (a5) created.")

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
                for (p_prime, _, _, _, _), value in self.y_pmidt.items()
                if p == p_prime
            )
            # Get requested number of treatments for patient p if total stay was in the planning horizon
            total_treatments = sum(self.lr_pm[p, m] for m in self.M_p[p])

            minimize_missing_treatment += total_treatments - scheduled_treatments

        objective = (
            self.delay_value * minimize_delay  # type: ignore
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

    def _extract_solution(self):
        if not self.solution_found:
            return NO_SOLUTION_FOUND
        print_vars = False
        # Simplified for clarity, adjust if required
        appointments = []

        x_midt_values = self.model.getAttr("X", self.x_midt)
        y_midt_values = self.model.getAttr("X", self.y_pmidt)
        z_gfmidt_values = self.model.getAttr("X", self.z_gfmidt)
        a_pd_values = self.model.getAttr("X", self.a_pd)
        for m in self.M:
            for d in self.D:
                for i in self.J_md[m, d]:
                    for t in self.T:
                        if x_midt_values[m, i, d, t] > 0.5:
                            if print_vars:
                                logger.debug(
                                    f"x_{m.id}_{i}_{d}_{t} = {x_midt_values[m, i, d, t]}"
                                )
                            # Get patients
                            patients = []
                            for p in self.P:
                                if m not in self.M_p[p]:
                                    continue
                                if d not in self.A_p[p]:
                                    continue
                                if print_vars:
                                    logger.debug(
                                        f"y_{p.id}_{m.id}_{i}_{d}_{t} = {y_midt_values[p, m, i,d,t]}"
                                    )
                                if y_midt_values[p, m, i, d, t] > 0.5:
                                    patients.append(p)

                            # Get resources
                            resources = defaultdict(list)

                            for fhat in self.Fhat_m[m]:
                                for f in self.fhat[fhat]:
                                    if z_gfmidt_values[fhat, f, m, i, d, t] > 0.5:
                                        resources[fhat].append(f)
                                    if print_vars:
                                        logger.debug(
                                            f"z_{fhat.id}_f{f.id}_{m.id}_{i}_{d}_{t} = {z_gfmidt_values[fhat, f, m, i, d, t]}"
                                        )
                            app = Appointment(
                                start_date=DayHour(d, t),
                                treatment=m,
                                patients=patients,
                                resources=resources,
                            )
                            appointments.append(app)
                            pass
                        else:
                            pass

        patients_arrival: dict[Patient, DayHour] = {}
        for p in self.P:
            for d in self.D_p[p]:
                if a_pd_values[p, d] > 0.5:  # type: ignore
                    # logger.debug(f"a_{p.id}_{d} = {self.a_pd[p,d].X}")
                    patients_arrival[p] = DayHour(
                        day=d, hour=self.instance.workday_start.hour
                    )

        return Solution(
            instance=self.instance,
            schedule=appointments,
            patients_arrival=patients_arrival,
            solver=self,
            solution_value=self.model.objVal,
        )
