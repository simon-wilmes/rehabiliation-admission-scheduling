from src.instance import create_instance_from_file
from src.solvers import MIPSolver, MIPSolver2
from src.solvers import CPSolver, CPSolver2
from src.logging import logger
from itertools import combinations
import os


def main():
    data_path = "data"
    folders = [
        f for f in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, f))
    ]
    inst_folders = [f for f in folders if f.startswith("inst") and f[4:].isdigit()]

    largest_folder = max(inst_folders, key=lambda x: int(x[4:]))

    # Assert Custom Instance
    if False:
        largest_folder = "test_inst"
        file = "instance_2.txt"
    else:
        largest_folder = "comp_study_001"
        file = "instance_2.txt"
    logger.setLevel("DEBUG")

    logger.info(f"Running with instance folder: {largest_folder}/{file}")
    inst = create_instance_from_file("data/" + str(largest_folder) + "/" + file)
    logger.info("Successfully created instance from file.")
    solvers = [
        (CPSolver, {"break_symetry": "True"}),
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
        solution = solver_cp.solve_model()
        continue

        solver_cp1 = CPSolver(
            inst,
            use_resource_loyalty=False,
            use_even_distribution=False,
            use_conflict_groups=False,
            break_symetry="False",
            **kwargs,
        )
        solver_cp1.create_model()
        solver_cp1.assert_solution(solution)
        solution2 = solver_cp1.solve_model()
        solver3 = solver_cp1.solver


if __name__ == "__main__":
    main()
