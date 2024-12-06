from src.instance import create_instance_from_file
from src.solvers import MIPSolver, MIPSolver2, MIPSolver3
from src.solvers import CPSolver, CPSolver2
from src.logging import logger
from itertools import combinations
import os
from src.utils import get_file_writer_context
import contextlib


def main():
    #####################################
    # Select instance
    #####################################

    # Get largest test instance
    data_path = "data"
    folders = [
        f for f in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, f))
    ]
    inst_folders = [f for f in folders if f.startswith("inst") and f[4:].isdigit()]
    largest_folder = max(inst_folders, key=lambda x: int(x[4:]))

    # Assert Custom Instance
    if True:
        largest_folder = "test_inst"
        file = "instance_1.txt"
    else:
        largest_folder = "comp_study_001"
        file = "instance_1.txt"

    #####################################
    # Set Settings
    #####################################

    # Solver settings
    settings_dict = {
        MIPSolver: {},
        MIPSolver2: {},
        MIPSolver3: {
            "break_symmetry": False,
            "break_symmetry_strong": False,
        },
        CPSolver: {
            "break_symmetry": True,
        },
        CPSolver2: {
            "break_symmetry": True,
        },
    }

    # Debug Settings
    if True:
        debug_settings = {
            "log_to_console": True,
            "log_to_file": True,
        }
        logger.setLevel("DEBUG")
    else:
        debug_settings = {
            "log_to_console": False,
            "log_to_file": False,
        }
        logger.setLevel("INFO")

    #####################################
    # Run Solver
    #####################################

    logger.info(f"Runningwith instance folder: {largest_folder}/{file}")
    inst = create_instance_from_file("data/" + str(largest_folder) + "/" + file)
    logger.info("Successfully created instance from file.")

    solvers = [
        MIPSolver3,
    ]

    for solver in solvers:
        context = get_file_writer_context(solver, inst, **debug_settings)

        with contextlib.redirect_stdout(context):  # type: ignore
            # Build kwargs
            kwargs = settings_dict[solver]
            kwargs.update(debug_settings)

            logger.info(f"Running with solver: {solver.__name__} and kwargs: {kwargs}")
            solver_cp = solver(
                inst,
                **kwargs,
            )
            solver_cp.create_model()
            solution = solver_cp.solve_model()


if __name__ == "__main__":
    main()
