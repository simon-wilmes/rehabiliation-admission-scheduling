from src.instance import create_instance_from_file
from src.solvers.mip import MIPSolver
from src.solvers.cp_or import CPSolver
from src.logging import logger
from itertools import combinations

OPTIONS = {"conflict-groups", "resource-loyalty", "even-distribution"}
SOLVER_OPTIONS = {"mip": MIPSolver, "cp1": CPSolver}


def main():
    for r in range(len(OPTIONS) + 1):
        for comb in combinations(OPTIONS, r):
            for solver in SOLVER_OPTIONS:
                # logger.debug(f"Running with options: {comb} and solver: {solver}")
                # inst = create_instance_from_file("data/inst001/inst001.txt")
                # solver = SOLVER_OPTIONS[solver](inst, ignored_constraints=comb)
                # solver.create_model()
                # solver.solve_model()
                pass
    inst = create_instance_from_file("data/inst001/inst001.txt")
    solver_mip = MIPSolver(inst)
    solver_mip.create_model()
    solver_mip.solve_model()

    solver_cp = CPSolver(inst, use_resource_loyalty=False)
    solver_cp.create_model()
    solver_cp.solve_model()
    pass
    logger.debug("Successfully created instance from file.")


if __name__ == "__main__":
    main()
