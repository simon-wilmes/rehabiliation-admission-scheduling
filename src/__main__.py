from src.instance import create_instance_from_file
from src.solvers import (
    MIPSolver,
    MIPSolver3,
    LBBDSolver,
    CPSolver,
    CPSolver2,
)
from src.solvers.subsolvers import CPSubsolver, CPSubsolver2


from src.logging import logger


from itertools import combinations
import os
from src.utils import get_file_writer_context, generate_combis
import contextlib
from typing import Type
from src.solvers.solver import Solver
from pprint import pformat
import sys
import ast


def read_solver_cls():
    available_solvers: dict[str, Type[Solver]] = {
        "MIPSolver": MIPSolver,
        "MIPSolver3": MIPSolver3,
        "CPSolver": CPSolver,
        "LBBDSolver": LBBDSolver,
    }

    try:
        arg_str = sys.argv[1]
        assert arg_str in available_solvers, "1st argument is not a valid solver"
        return available_solvers[arg_str]
    except (ValueError, SyntaxError) as e:
        logger.error(f"Invalid argument: {e}")

    raise AssertionError("Invalid argument 1")


def read_arg_dict():
    try:
        arg_dict = ast.literal_eval(sys.argv[2])
        if isinstance(arg_dict, dict):
            return arg_dict

    except (ValueError, SyntaxError) as e:
        logger.error(f"Invalid argument: {e}")

    raise AssertionError("Invalid argument 2")


def read_file_path():
    try:
        arg_str = sys.argv[3]
        assert os.path.isfile(arg_str), "3rd Argument is not a path"

        return arg_str
    except (ValueError, SyntaxError) as e:
        logger.error(f"Invalid argument: {e}")

    assert False, "Invalid argument 3"


def main():
    #####################################
    # Select instance
    #####################################
    solver_cls = read_solver_cls()
    arg_dict = read_arg_dict()
    file_path = read_file_path()
    # Get largest test instance

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
        MIPSolver: {"use_lazy_constraints": True},
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
            "break_symmetry": False,
            "subsolver_cls": CPSubsolver,
            "subsolver.store_results": True,
            "subsolver.store_results_method": "hash",
            "use_helper_constraints": True,
        },
    }

    # Debug Settings
    debug_settings = {
        "log_to_console": True,
    }
    logger.setLevel("DEBUG")

    #####################################
    # Run Solver
    #####################################

    inst = create_instance_from_file(file_path)

    # Create kwargs for solver
    kwargs = settings_dict[solver_cls]  # type: ignore
    kwargs.update(default_settings)
    kwargs.update(debug_settings)

    test_parameter_combinations = False

    context = get_file_writer_context(**debug_settings)

    print_dict = {"solver": solver_cls.__name__, "args": kwargs}
    logger.info(print_dict)
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
