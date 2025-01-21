import os
import re


def process_file(file_path):
    with open(file_path, "r") as file:
        # Process the file content here
        print(f"Processing file: {file_path}")
        content = file.read()   
        # Define the regex pattern to capture the runtime
        runtime_pattern = re.compile(r"Runtime:\s*(\d+\.\d+)")

        # Search for the pattern in the file content
        match = runtime_pattern.search(content)
        if match:
            runtime = match.group(1)
            print(f"Runtime found: {runtime}")
        else:
            print("Runtime not found in the file.")
    


# Specify the folder path
folder_path = "/work/wx350715/Kobra/output_core_test/"
for filename in os.listdir(folder_path):
    if filename.endswith(".out"):
        file_path = os.path.join(folder_path, filename)
        process_file(file_path)
        # Process the file content here
