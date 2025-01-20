import re
import random

value = 1.2


def double_second_number_in_file(file_path):
    # Read the contents of the file
    with open(file_path, "r") as file:
        content = file.read()

    # Define the pattern for finding occurrences of "[0-9]:[0-9]*"
    pattern = r"(\d+):(\d+)"

    # Function to double the second number

    def replace_match(match):
        first_number = match.group(1)
        second_number = int(match.group(2))
        if random.random() < 0.5:
            modified_second = second_number * value
        else:
            modified_second = second_number * value - 1
        return f"{first_number}:{max(round(modified_second), 0)}"

    # Replace occurrences in the content
    updated_content = re.sub(pattern, replace_match, content)

    # Write the updated content back to the file
    with open(file_path, "w") as file:
        file.write(updated_content)


# Example usage
file_path = (
    "data/comp_study_002/instance_10_30.txt"  # Replace with the path to your file
)
double_second_number_in_file(file_path)
