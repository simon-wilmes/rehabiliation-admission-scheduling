from src.instance import create_instance_from_file
from src.solvers.mip import MIPSolver
from src.logging import logger


def main():
    inst = create_instance_from_file("data/inst001/inst001.txt")
    solver = MIPSolver(inst)
    solver.create_model()
    solver.solve_model()
    pass
    logger.debug("Successfully created instance from file.")


if __name__ == "__main__":
    main()
