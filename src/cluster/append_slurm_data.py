import os
import re
import subprocess


def get_slurm_stats(job_id):
    """
    Retrieves SLURM resource usage statistics for a given job name.
    Replace 'sacct' with the appropriate SLURM command as needed.
    """
    try:
        result = subprocess.run(
            [
                "sacct",
                "-j",
                job_id,
                "--format=JobID%60,MaxRSS,Elapsed,State",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if result.returncode != 0:
            raise Exception(result.stderr)
        return result.stdout.strip()
    except Exception as e:
        print(f"Error retrieving SLURM stats for job {job_id}: {e}")
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

            job_id = file_path.split("_")[-1][:-4]
            print(f"Processing job: {job_id} in file: {file_name}")

            # Get SLURM stats
            slurm_stats = get_slurm_stats(job_id)
            if slurm_stats:
                # Append stats to the file
                with open(file_path, "a") as file:
                    file.write("\nSLURM STATS:\n")
                    file.write(slurm_stats + "\n")
                print(f"Appended stats to {file_name}.")
            else:
                print(f"Failed to retrieve stats for {file_name}.")


output_folder = "/work/ir803925/Kobra/output/"
if os.path.isdir(output_folder):
    process_output_files(output_folder)
else:
    print("Output folder doesnt exist!")
