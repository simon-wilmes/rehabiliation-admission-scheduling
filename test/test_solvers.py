import pytest
from src.solvers.solver import Solver
from src.solvers.mip import MIPSolver
from src.solvers.cp_or import CPSolver
from src.solvers.mip3 import MIPSolver3
from src.solvers import LBBDSolver
from src.instance import create_instance_from_file
from src.time import DayHour, Duration
from src.solution import NO_SOLUTION_FOUND
from typing import Type, Any
from src.logging import logger


@pytest.mark.parametrize("solver", [MIPSolver, LBBDSolver, CPSolver])
@pytest.mark.parametrize(
    "file_path_results",
    [
        ("data/testinstance_files/resource_only_in_groups.txt", None),
        ("data/testinstance_files/too_many_treatments.txt", None),
        ("data/testinstance_files/barely_fit.txt", 0),
        ("data/testinstance_files/no_avail_test.txt", None),
        ("data/testinstance_files/double_resources.txt", 0),
    ],
)
def test_small_solution(
    solver: Type[Solver], file_path_results: tuple[str, int | None]
):
    file_path, result = file_path_results
    instance = create_instance_from_file(file_path)
    logger.setLevel("DEBUG")

    solver_instance = solver(instance)
    solver_instance.create_model()
    solution = solver_instance.solve_model()
    if result is None:
        assert solution is NO_SOLUTION_FOUND
    else:
        assert solution.calc_objective() == result
