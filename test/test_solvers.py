import pytest
from src.solvers.solver import Solver
from src.solvers.mip import MIPSolver
from src.solvers.cp_or import CPSolver
from src.solvers import LBBDSolver
from src.instance import create_instance_from_file
from src.time import DayHour, Duration
from src.solution import NO_SOLUTION_FOUND
from typing import Type
from src.logging import logger


@pytest.mark.parametrize("solver", [MIPSolver, CPSolver, LBBDSolver])
def test_infeasible_solution(solver: Type[Solver]):
    instance = create_instance_from_file(
        "data/testinstance_files/resource_only_in_groups.txt"
    )
    logger.setLevel("DEBUG")
    solver_instance = solver(instance)
    solver_instance._create_model()
    solution = solver_instance.solve_model()
    assert solution is NO_SOLUTION_FOUND
