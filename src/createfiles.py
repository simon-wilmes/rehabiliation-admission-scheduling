import os


def create_file(file_path, file_name):
    full_path = os.path.join(file_path, file_name)
    if not os.path.exist(file_path):
        return "file path does not exist"

    with open(full_path, "w") as file:
        file.write("")

    print(f"File created at: {full_path}")


print("hallo")
