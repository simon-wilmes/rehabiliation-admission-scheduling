import os
import re
from collections import defaultdict

data_all = defaultdict(dict)


def process_file(file_path):
    with open(file_path, "r") as file:
        # Process the file content here
        print(f"Processing file: {file_path}")
        content = file.read()
        add_values = True
        # Define the regex pattern to capture the runtime
        runtime_pattern = re.compile(r"Time to find solution: (([0-9]*)\.[0-9]*)s")

        # Search for the pattern in the file content
        match = runtime_pattern.search(content)
        if match:
            runtime = match.group(1)
            print(f"Runtime found: {runtime}")
        else:
            print("Runtime not found in the file.")
            add_values = False
        cpu_cores_pattern = r"5.CPU_CORES: ([0-9]*)"
        match = re.search(cpu_cores_pattern, content)
        if match:
            cpu_cores = match.group(1)
            print(f"CPU cores found: {cpu_cores}")
        else:
            print("CPU cores not found in the file.")
            add_values = False
        threads_pattern = r"'number_of_threads': ([0-9]*)"
        match = re.search(threads_pattern, content)
        if match:
            num_threads = match.group(1)
            print(f"Number of threads found: {num_threads}")
        else:
            print("Number of threads not found in the file.")
            add_values = False

        if "instance_1_30.txt" in content:
            instance = "instance_1_30.txt"
        elif "instance_p10" in content:
            instance = "instance_p10_t15_low.txt"
        else:
            print("Unknown instance.")
            add_values = False
        if "MIPSolver" in content:
            solver = "MIPSolver"
        elif "LBBDSolver" in content:
            solver = "LBBDSolver"
        else:
            print("Unknown solver.")
            add_values = False
        if add_values:
            data_all[(solver, instance)][(int(num_threads), int(cpu_cores))] = float(
                runtime
            )


# Specify the folder path
folder_path = "/work/wx350715/Kobra/output_core_test/"
for filename in os.listdir(folder_path):
    if filename.endswith(".out"):
        file_path = os.path.join(folder_path, filename)
        process_file(file_path)
        # Process the file content here

print(data_all)
data_keys = sorted(data_all.keys())
for data_key in data_keys:
    print("Solver:", data_key[0])
    print("Instance:", data_key[1])
    data = data_all[data_key]
    # Step 1: Extract unique and sorted values for a and b
    a_values = sorted({key[0] for key in data.keys()})
    b_values = sorted({key[1] for key in data.keys()})

    # Step 2: Create the table
    table = []
    header = ["a\\b"] + [str(b) for b in b_values]
    table.append(header)

    for a in a_values:
        row = [str(a)]  # First column (row label)
        for b in b_values:
            value = data.get((a, b), "")  # Use empty string for missing pairs
            row.append(str(value))
        table.append(row)

    # Step 3: Print the table
    print("a = threads")
    print("b = CPU cores")
    col_widths = [max(len(item) for item in col) for col in zip(*table)]
    for row in table:
        print(" | ".join(item.ljust(width) for item, width in zip(row, col_widths)))
    pass
