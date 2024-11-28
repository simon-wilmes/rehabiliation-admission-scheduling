from src.instance import create_instance_from_file
from src.solvers import MIPSolver, MIPSolver2
from src.solvers import CPSolver, CPSolver2
from src.logging import logger
from itertools import combinations
import os

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
    data_path = "data"
    folders = [
        f for f in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, f))
    ]
    inst_folders = [f for f in folders if f.startswith("inst") and f[4:].isdigit()]

    largest_folder = max(inst_folders, key=lambda x: int(x[4:]))

    # Assert Custom Instance
    largest_folder = "comp_study_001"
    file = "instance_1.txt"
    logger.setLevel("DEBUG")

    logger.info(f"Running with instance folder: {largest_folder}/{file}")
    inst = create_instance_from_file("data/" + str(largest_folder) +"/" +file)
    logger.info("Successfully created instance from file.")
    solvers = [
        # (CPSolver2, {"break_symetry": False}),
        # (MIPSolver2, {}),
        # (MIPSolver, {}),
        (MIPSolver, {}),
        (CPSolver, {"break_symetry": True}),
        (CPSolver2, {"break_symetry": True}),
        # (CPSolver, {"break_symetry": False}),
    ]

    for solver, kwargs in solvers:
        logger.info(f"Running with solver: {solver.__name__} and kwargs: {kwargs}")
        solver_cp = solver(
            inst,
            use_resource_loyalty=False,
            use_even_distribution=False,
            use_conflict_groups=False,
            **kwargs,
        )
        solver_cp.create_model()
        solver_cp.solve_model()

    pass


if __name__ == "__main__":
    main()
