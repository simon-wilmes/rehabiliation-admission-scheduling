import os
import shutil
from copy import copy
import math
import re


def replace_dict_fact(match, factor):
    # Extract all key-value pairs
    kv_pairs = match.group(1)

    # Process each key-value pair
    modified_pairs = []
    for kv_pair in kv_pairs.split(","):
        key, value = map(int, kv_pair.split(":"))
        new_value = math.ceil(value * factor)
        modified_pairs.append(f"{key}: {new_value}")

    # Return the modified dictionary as a string
    return "{" + ", ".join(modified_pairs) + "}"


input_dir = "data/comp_study_003"
output_dir = "data/comp_study_004"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)


for file in os.listdir(output_dir):
    if file.endswith(".txt"):
        os.remove(os.path.join(output_dir, file))


pattern = re.compile(r"\{((?:\d+:\s*\d+,\s*)*\d+:\s*\d+)\}")


def create_time_slot_copy(filename):
    for key, time_slot_length in {"t30": 30, "t05": 5}.items():
        modified_data = copy(data)
        mod_file_name = filename.replace("t15", key)
        modified_data = modified_data.replace(
            "time_slot_length: Duration(hours=0, minutes=15)",
            f"time_slot_length: Duration(hours=0, minutes={time_slot_length})",
        )

        new_file_path = os.path.join(output_dir, mod_file_name)
        with open(new_file_path, "w") as new_file:
            new_file.write(modified_data)
            pass


for filename in os.listdir(input_dir):
    if filename.endswith(".txt"):
        file_path = os.path.join(input_dir, filename)
        with open(file_path, "r") as file:
            data = file.read()

        # Copy original file to new directory
        shutil.copy(file_path, os.path.join(output_dir, filename))
        create_time_slot_copy(filename)

        # Modify the number of treatments
        for key, factor in {"medium": 1.5, "high": 2.0, "vigh": 3.0}.items():
            modified_data = copy(data)
            mod_file_name = filename.replace("low", key)
            replace_dict = lambda match: replace_dict_fact(match, factor)
            parts = modified_data.split("[DATA:")
            new_parts = []
            for part in parts:
                if part.startswith(" PATIENTS]"):
                    new_parts.append(pattern.sub(replace_dict, part))
                else:
                    new_parts.append(part)

            # Join the data back together
            modified_data = "[DATA:".join(new_parts)

            new_file_path = os.path.join(output_dir, mod_file_name)
            with open(new_file_path, "w") as new_file:
                new_file.write(modified_data)

            if "p50" not in filename and "p60" not in filename:
                create_time_slot_copy(mod_file_name)


# Count the number of .txt files in the output directory
txt_files = [f for f in os.listdir(output_dir) if f.endswith(".txt")]
num_files = len(txt_files)
print(f"Number of instances generated: {num_files}")
