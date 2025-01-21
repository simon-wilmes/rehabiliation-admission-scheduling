import os
import re
import subprocess


def get_slurm_stats(job_name):
    """
    Retrieves SLURM resource usage statistics for a given job name.
    Replace 'sacct' with the appropriate SLURM command as needed.
    """
    try:
        result = subprocess.run(
            [
                "sacct",
                "-j",
                job_name,
                "--format=JobID,MaxRSS,Elapsed,State",
                "--noheader",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if result.returncode != 0:
            raise Exception(result.stderr)
        return result.stdout.strip()
    except Exception as e:
        print(f"Error retrieving SLURM stats for job {job_name}: {e}")
        return None


def process_output_files(output_folder):
    """
    Processes all .out files in the given folder.
    """
    for file_name in os.listdir(output_folder):
        if file_name.endswith(".out"):
            file_path = os.path.join(output_folder, file_name)

            # Read the file content
            with open(file_path, "r") as file:
                content = file.read()

            # Check if stats have already been appended
            if "SLURM STATS:" in content:
                print(f"Stats already appended for {file_name}. Skipping.")
                continue

            # Find the job name from the content
            match = re.search(r"^0\.JOB_NAME: (.+)$", content, re.MULTILINE)
            if match:
                job_name = match.group(1).strip()
                print(f"Processing job: {job_name} in file: {file_name}")

                # Get SLURM stats
                slurm_stats = get_slurm_stats(job_name)
                if slurm_stats:
                    # Append stats to the file
                    with open(file_path, "a") as file:
                        file.write("\nSLURM STATS:\n")
                        file.write(slurm_stats + "\n")
                    print(f"Appended stats to {file_name}.")
                else:
                    print(f"Failed to retrieve stats for {file_name}.")
            else:
                print(f"No job name found in {file_name}. Skipping.")


output_folder = "/work/wx350715/Kobra/output_core_test/"
if os.path.isdir(output_folder):
    process_output_files(output_folder)
