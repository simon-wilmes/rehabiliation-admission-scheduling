from src.instance import Instance
import gurobipy as gp
from src.logging import logger, print

from itertools import product
from src.solution import Solution, NO_SOLUTION_FOUND, Appointment
from src.solvers.solver import Solver
from src.time import DayHour
from src.patients import Patient
from src.utils import slice_dict
from collections import defaultdict


class MIPSolver2(Solver):
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

    def _solve_model(self) -> Solution | int:
        self.model.optimize()
        if self.model.status == gp.GRB.OPTIMAL:
            logger.debug("Optimal solution found.")
            solution = self._extract_solution()
            return solution
        else:
            logger.error("No optimal solution found.")
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

        self._create_parameter_sets()

        self.model = gp.Model("UpdatedMIP")
        self.model.setParam("LogToConsole", int(self.log_to_console))  # type: ignore
        self.model.setParam("Threads", self.number_of_threads)  # type: ignore
        self._create_variables()
        self._create_constraints()
        self._set_optimization_goal()
        self.model = self.model

    def _create_variables(self):
        self.x_midt = self.model.addVars(
            (
                (m, i, d, t)
                for m in self.M
                for i in self.I_m[m]
                for d in self.D
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

        self.y_pmi = self.model.addVars(
            ((p, m, i) for p in self.P for m in self.M_p[p] for i in self.I_m[m]),  # type: ignore
            vtype=gp.GRB.BINARY,
            name="y_pmi",
        )

        self.z_fmi = self.model.addVars(
            ((f, m, i) for f in self.F for m in self.M for i in self.I_m[m]),  # type: ignore
            vtype=gp.GRB.BINARY,
            name="z_fmi",
        )

        self.v_pmf = self.model.addVars(
            ((p, m, f) for p in self.P for m in self.M_p[p] for f in self.F),  # type: ignore
            vtype=gp.GRB.BINARY,
            name="v_pmf",
        )

        self.u_mi = self.model.addVars(
            ((m, i) for m in self.M for i in self.I_m[m]),  # type: ignore
            vtype=gp.GRB.BINARY,
            name="u_mi",
        )

    def _create_constraints(self):
        self.model.update()
        # Constraint: Every patient is assigned to exactly the required number of treatments
        for p in self.P:
            for m in self.M_p[p]:
                self.model.addConstr(
                    gp.quicksum(self.y_pmi[p, m, i] for i in self.I_m[m])
                    == self.lr_pm[p, m],
                    name=f"constraint_p1_b_p{p.id}_m{m.id}",
                )

        # Constraint: Only treatments scheduled if addmitted
        for p in self.P:
            for m in self.M_p[p]:
                for i, d, t in product(self.I_m[m], self.D, self.T):
                    delta_set = [
                        delta for delta in self.D_p[p] if d - self.l_p[p] < delta <= d
                    ]

                    self.model.addConstr(
                        self.x_midt[m, i, d, t] * self.y_pmi[p, m, i]
                        <= gp.quicksum(self.a_pd[p, delta] for delta in delta_set),
                        name=f"constraint_p2_b_p{p.id}_m{m.id}_i{i}_d{d}_t{t}",
                    )

        # Constraint: Only one treatment every timeslot per patient
        for p in self.P:
            for d, t in product(self.A_p[p], self.T):
                tau_set = [tau for tau in self.T if t - self.du_m[m] < tau <= t]

                self.model.addConstr(
                    gp.quicksum(
                        self.x_midt[m, i, d, tau] * self.y_pmi[p, m, i]
                        for m in self.M_p[p]
                        for i in self.I_m[m]
                        for tau in tau_set
                    )
                    <= 1,
                    name=f"constraint_p3_b_p{p.id}_d{d}_t{t}",
                )

        # Constraint: Bed usage constraint
        for d in self.D:
            term = gp.LinExpr()
            for p in self.P:
                delta_set = [
                    delta for delta in self.D_p[p] if d - self.l_p[p] < delta <= d
                ]
                term += gp.quicksum(self.a_pd[p, delta] for delta in delta_set)
            self.model.addConstr(term <= self.b, name=f"constraint_r1_b_d{d}")

        # Constraint: Resource usage constraint
        for fhat in self.Fhat:
            for f in self.fhat[fhat]:
                for d, t in product(self.D, self.T):
                    self.model.addConstr(
                        gp.quicksum(
                            self.z_fmi[f, m, i] * self.x_midt[m, i, d, tau]
                            for m in self.M_fhat[fhat]
                            for i in self.I_m[m]
                            for tau in self.T
                            if t - self.du_m[m] < tau <= t
                        )
                        <= int(f.is_available(DayHour(d, t))),
                        name=f"constraint_r2_b_f{f.id}_d{d}_t{t}",
                    )
        # Constraint: Treatment has resources
        for m in self.M:
            for i in self.I_m[m]:
                for fhat in self.Fhat_m[m]:
                    self.model.addConstr(
                        gp.quicksum(self.z_fmi[f, m, i] for f in self.fhat[fhat])
                        == self.n_fhatm[fhat, m],
                        name=f"constraint_r3_b_m{m.id}_i{i}_fhat{fhat.id}",
                    )

        # Constraint: Treamtnet must be scheduled if patient is assigned
        for p in self.P:
            for m in self.M_p[p]:
                for i in self.I_m[m]:
                    self.model.addConstr(
                        self.y_pmi[p, m, i]
                        <= gp.quicksum(
                            self.x_midt[m, i, d, t] for d in self.A_p[p] for t in self.T
                        ),
                        name=f"constraint_r4_b_m{m.id}_i{i}",
                    )

        # Constraint: every repetition is only scheduled once
        for m in self.M:
            for i in self.I_m[m]:
                self.model.addConstr(
                    gp.quicksum(self.x_midt[m, i, d, t] for d in self.D for t in self.T)
                    <= 2,
                    name=f"constraint_r5_b_m{m.id}_i{i}",
                )

        if self.add_resource_loyal():
            logger.warning("Resource loyalty not implemented for MIP2")

        if self.add_even_distribution():
            logger.warning("Even distribution not implemented for MIP2")

        if self.add_conflict_groups():
            logger.warning("Conflict groups not implemented for MIP2")

    def _set_optimization_goal(self):

        minimize_treatments = gp.quicksum(
            self.x_midt[m, i, d, t]
            for m in self.M
            for i in self.I_m[m]
            for d in self.D
            for t in self.T
        )

        minimize_delay = gp.quicksum(
            (d - min(self.A_p[p])) * self.a_pd[p, d]
            for p in self.P
            for d in self.D_p[p]
        )

        objective = minimize_treatments + minimize_delay

        self.model.setObjective(objective, gp.GRB.MINIMIZE)

    def _extract_solution(self):

        # Simplified for clarity, adjust if required
        appointments = []
        for m in self.M:
            for i in self.I_m[m]:
                for d in self.D:
                    for t in self.T:
                        if self.x_midt[m, i, d, t].X > 0.5:
                            logger.debug(
                                f"x_{m}_{i}_{d}_{t} = {self.x_midt[m, i, d, t].X}"
                            )
                            # Get patients
                            patients = []
                            for p, _, _ in slice_dict(self.y_pmi, None, m, i):
                                logger.debug(f"y_{p}_{m}_{i} = {self.y_pmi[p, m, i].X}")
                                if self.y_pmi[p, m, i].X > 0.5:
                                    patients.append(p)
                                    break
                            # Get resources
                            resources = defaultdict(list)

                            for f, _, _ in slice_dict(self.z_fmi, None, m, i):
                                if self.z_fmi[f, m, i].X > 0.5:
                                    resources[f.resource_group].append(f)
                                logger.debug(f"z_{f}_{m}_{i} = {self.z_fmi[f, m, i].X}")
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
                if self.a_pd[p, d].X > 0.5:  # type: ignore
                    patients_arrival[p] = DayHour(
                        day=d, hour=self.instance.workday_start.hour
                    )

        return Solution(
            instance=self.instance,
            schedule=appointments,
            patients_arrival=patients_arrival,
            solver=self,
            test_even_distribution=self.add_even_distribution(),
            test_conflict_groups=self.add_conflict_groups(),
            test_resource_loyalty=self.add_resource_loyal(),
            solution_value=self.model.objVal,
        )
