from src.instance_generation.create_instance import generate_instance
from src.instance_generation.write_instance import write_instance_to_file
import os

if __name__ == "__main__":
    params = {
        "num_patients": 15,
        "num_resource_groups": 6,
        "num_treatments": 15,
        "workday_start": 8,
        "workday_end": 18,
        "length_of_stay_range": (15, 25),
        "treatments_per_patient_range": (1, 5),
        "treatment_duration_distribution": {"mean": 1, "std_dev": 0.25},
        "resource_usage_target": 1,
        "admitted_patient_percentage": 0,
        "resource_group_loyalty_probability_range": (0.1, 0.9),
        "time_granularity_minutes": 15,
        "num_beds": 9,
        "max_admission_day": 15,
        "resource_group_unavailability_percentages": {
            0: 0,  # 10% unavailability for Resource Group 0
            1: 0,  # 20% unavailability for Resource Group 1
            2: 0.05,  # 5% unavailability for Resource Group 2
        },
        "unavailability_duration_range": (
            30,
            120,
        ),  # Durations between 30 and 120 minutes
    }

    # Find the biggest number of folder of form inst(number)
    base_dir = "data"
    folders = [
        f for f in os.listdir(base_dir) if f.startswith("inst") and f[4:].isdigit()
    ]
    max_number = max([int(f[4:]) for f in folders], default=0)
    new_folder_number = max_number + 1
    new_folder_name = f"inst{new_folder_number:03}"
    new_folder_path = os.path.join(base_dir, new_folder_name)

    # Create the new folder
    os.makedirs(new_folder_path, exist_ok=True)

    # Generate (variable) many instances
    num_instances = 1  # Set the number of instances you want to generate
    for i in range(num_instances):
        instance = generate_instance(params)
        instance_filename = os.path.join(new_folder_path, f"instance_{i+1}.txt")
        write_instance_to_file(instance_filename, instance)
