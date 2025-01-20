from docplex.cp.model import CpoModel
from sys import stdout
from src.utils import CP_PATH

mdl = CpoModel(name="buses")

from collections import namedtuple


import sys
import time
from ortools.sat.python import cp_model


# -----------------------------------------------------------------------------
# Build the model
# -----------------------------------------------------------------------------
def cp_main(queens):
    # Set model parameters
    NB_QUEEN = queens
    # Create model
    mdl = CpoModel()

    # Create column index of each queen
    x = mdl.integer_var_list(NB_QUEEN, 0, NB_QUEEN - 1, "X")

    # One queen per raw
    mdl.add(mdl.all_diff(x))

    # One queen per diagonal xi - xj != i - j
    mdl.add(mdl.all_diff(x[i] + i for i in range(NB_QUEEN)))

    # One queen per diagonal xi - xj != j - i
    mdl.add(mdl.all_diff(x[i] - i for i in range(NB_QUEEN)))

    # ----------------------------------------------------f-------------------------
    # Solve the model and display the result
    # -----------------------------------------------------------------------------

    msol = mdl.solve(
        # trace_log=False,
        execfile="/home/simon/ibm/ILOG/CPLEX_Studio2211/cpoptimizer/bin/x86-64_linux/cpoptimizer",
    )
    return [msol[l] for l in x]


def main(board_size: int):
    # Creates the solver.
    model = cp_model.CpModel()

    # Creates the variables.
    # There are `board_size` number of variables, one for a queen in each column
    # of the board. The value of each variable is the row that the queen is in.
    queens = [model.new_int_var(0, board_size - 1, f"x_{i}") for i in range(board_size)]

    # Creates the constraints.
    # All rows must be different.
    model.add_all_different(queens)

    # No two queens can be on the same diagonal.
    model.add_all_different(queens[i] + i for i in range(board_size))
    model.add_all_different(queens[i] - i for i in range(board_size))

    # Solve the model.
    solver = cp_model.CpSolver()
    solver.solve(model)
    return [solver.value(q) for q in queens]


if __name__ == "__main__":
    mdl = CpoModel()
    interval_1 = mdl.interval_var(
        start=(0, 13), size=12, optional=True, name="interval_treatment"
    )

    interval_2 = mdl.interval_var(
        start=(0, 16), size=5, optional=True, name="interval_2"
    )
    mdl.add(mdl.presence_of(interval_1) == 1)
    mdl.add(mdl.alternative(interval_1, [interval_2]))
    result = mdl.solve(execfile=CP_PATH)
    pass
else:
    test_runs = 100
    sols_a = [0] * test_runs
    from time import time

    timea = time()
    for i in range(test_runs):
        sols_a[i] = cp_main(30)
    timea = time() - timea
    sols_b = [0] * test_runs
    timeb = time()
    for i in range(test_runs):
        sols_b[i] = main(30)
    timeb = time() - timeb
    print(timea, timeb)
    pass
    print(sols_a)
    print(sols_b)
