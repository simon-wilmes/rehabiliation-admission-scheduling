from src.logging import logger
import random
import time
from src.instance_generation.create_instance import generate_instance_to_file
import os
from src.instance import create_instance_from_file
from src.solvers import MIPSolver
from src.solvers import CPSolver

goal_time = 2 * 60  # 2 minutes


if __name__ == "__main__":
    logger.setLevel("DEBUG")
    random.seed(1)
    while True:
        params = {
            "num_patients": 5,
            "num_resource_groups": 4,
            "num_treatments": 8,
            "workday_start": 8,
            "workday_end": 18,
            "length_of_stay_range": (3, 6),
            "treatments_per_patient_range": (1, 5),
            "treatment_duration_distribution": {"mean": 1, "std_dev": 0.25},
            "resource_usage_target": 0.1,
            "admitted_patient_percentage": 0.3,
            "resource_group_loyalty_probability_range": (0, 0.9),
            "time_granularity_minutes": 15,
            "num_beds": 5,
            "max_admission_day": 30,
        }
        logger.info("Generating instance")

        params["num_patients"] = random.randint(4, 20)
        params["num_beds"] = random.randint(3, 7)
        params["resource_usage_target"] = random.random() * 0.8 + 0.1
        logger.info(f"num_patients: {params['num_patients']}")
        logger.info(f"num_beds: {params['num_beds']}")
        logger.info(f"resource_usage_target: {params['resource_usage_target']}")

        instance = generate_instance_to_file(params)

        start_time = time.time()
        data_path = "data"
        folders = [
            f
            for f in os.listdir(data_path)
            if os.path.isdir(os.path.join(data_path, f))
        ]
        inst_folders = [f for f in folders if f.startswith("inst") and f[4:].isdigit()]

        largest_folder = max(inst_folders, key=lambda x: int(x[4:]))

        logger.info(f"Running with instance folder: {largest_folder}/instance_1.txt")
        inst = create_instance_from_file(
            "data/" + str(largest_folder) + "/instance_1.txt"
        )
        logger.info("Successfully created instance from file.")
        solver_mip = MIPSolver(
            inst,
            use_resource_loyalty=False,
            use_even_distribution=False,
            use_conflict_groups=False,
        )

        solver_mip.create_model()
        solver_mip.solve_model()

        solver_cp = CPSolver(
            inst,
            use_resource_loyalty=False,
            use_even_distribution=False,
            use_conflict_groups=False,
        )
        solver_cp.create_model()
        solver_cp.solve_model()

        end_time = time.time()
        logger.info(f"Total Time: {end_time - start_time:.2f} seconds")
