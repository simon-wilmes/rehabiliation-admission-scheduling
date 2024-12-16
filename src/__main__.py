from src.instance import create_instance_from_file
from src.solvers import (
    MIPSolver,
    MIPSolver2,
    MIPSolver3,
    LBBDSolver,
    CPSolver,
    CPSolver2,
)
from src.solvers.subsolvers import CPSubsolver

from src.logging import logger


from itertools import combinations
import os
from src.utils import get_file_writer_context, generate_combis
import contextlib
from typing import Type
from src.solvers.solver import Solver
from pprint import pformat


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
    if len(inst_folders) != 0:
        largest_folder = max(inst_folders, key=lambda x: int(x[4:]))

    # Assert Custom Instance
    if True:
        largest_folder = "test_inst"
        file = "instance_1.txt"
    elif False:
        largest_folder = "comp_study_001"
        file = "instance_1.txt"
    else:
        largest_folder = "testinstance_files"
        file = "resource_only_in_groups.txt"

    #####################################
    # Set Settings
    #####################################
    default_settings = {
        "enforce_min_treatments_per_day": True,
        "enforce_max_treatments_per_e_w": True,
        "enforce_min_patients_per_treatment": True,
    }

    # Solver settings
    settings_dict = {
        MIPSolver: {"use_lazy_constraints": False},
        MIPSolver2: {"break_symmetry": True},
        MIPSolver3: {
            "break_symmetry": True,
            "break_symmetry_strong": True,
        },
        CPSolver: {
            "break_symmetry": True,
            "max_repr": "cumulative",
            "min_repr": "cumulative",
        },
        CPSolver2: {
            "break_symmetry": False,
            "treatments_in_adm_period": "cumulative",
        },
        LBBDSolver: {
            "break_symmetry": True,
            "subsolver_cls": CPSubsolver,
            "subsolver.store_results": True,
            "subsolver.store_results_method": "hash",
        },
    }

    # Debug Settings
    print_detailed_debug = True
    if print_detailed_debug:
        debug_settings = {
            "log_to_console": True,
            "log_to_file": True,
        }
        logger.setLevel("INFO")
    else:
        debug_settings = {
            "log_to_console": False,
            "log_to_file": False,
        }
        logger.setLevel("INFO")

    #####################################
    # Run Solver
    #####################################

    logger.info(f"Running with instance: {largest_folder}/{file}")
    inst = create_instance_from_file("data/" + str(largest_folder) + "/" + file)
    logger.info("Successfully created instance from file.")

    solver_cls = LBBDSolver
    # Create kwargs for solver
    kwargs = settings_dict[solver_cls]
    kwargs.update(default_settings)
    kwargs.update(debug_settings)

    test_parameter_combinations = False
    if test_parameter_combinations:
        test_run(
            solver_cls,
            inst,
            debug_settings,
            kwargs,
            ["add_knowledge", "break_symmetry", "break_symmetry_strong"],
        )
    else:
        logger.info("Running with: " + str(kwargs))
        context = get_file_writer_context(solver_cls, inst, **debug_settings)
        with contextlib.redirect_stdout(context):  # type: ignore
            logger.info(
                f"Running with solver: {solver_cls.__name__} and kwargs: {kwargs}"
            )
            solver = solver_cls(
                inst,
                **kwargs,
            )
            solver.create_model()
            solution = solver.solve_model()


def test_run(solver_cls: Type[Solver], inst, debug_settings, kwargs, testing_keys):
    # Build kwargs

    testing_keys = sorted(testing_keys)
    settings_comb = generate_combis(solver_cls, testing_keys)
    time_values = {}
    solution_values = {}
    logger.info(
        f"{len(settings_comb)} Combinations to test: \n"
        + pformat(settings_comb, indent=4)
    )

    for settings in settings_comb:
        logger.info("Running with Testing settings Combination: " + str(settings))
        kwargs.update(settings)
        context = get_file_writer_context(solver_cls, inst, **debug_settings)
        with contextlib.redirect_stdout(context):  # type: ignore
            # Build kwargs

            logger.info(
                f"Running with solver: {solver_cls.__name__} and kwargs: {kwargs}"
            )
            solver = solver_cls(
                inst,
                **kwargs,
            )
            solver.create_model()
            solution = solver.solve_model()
            sorted_comb = tuple(
                x[1] for x in sorted(list(settings.items()), key=lambda x: x[0])
            )
            time_values[sorted_comb] = solver.total_time if solution else "N/A"
            solution_values[sorted_comb] = solution.value if solution else "N/A"

    logger.info(f"Keys Tested: {testing_keys}")
    for key, value in time_values.items():
        logger.info(f"Settings: {key} - Time: {value}")


if __name__ == "__main__":
    main()
