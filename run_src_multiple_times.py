import subprocess
import time

command1 = 'pipenv run python -m src \'LBBDSolver\' \'{"subsolver_cls":"MIPSubsolver"}\' "data/comp_study_002/instance_1_short_timeframe_30.txt"'

command2 = "pipenv run python -m src 'MIPSolver' '{}' \"data/comp_study_002/instance_1_short_timeframe_5.txt\""


import subprocess
import time
import os
import glob


def run_script_with_runtime(i):

    # Build the command to activate pipenv and run the script

    try:
        # Start measuring time
        start_time = time.time()
        with open(f"output_src_{i}.txt", "w") as f:
            with open(f"error_src_{i}.txt", "w") as f_err:
                # Run the command
                result = subprocess.run(
                    command1,
                    cwd="/home/simon/Documents/uni/Kopra/rehabiliation-admission-scheduling",
                    shell=True,  # Use shell to allow `pipenv` commands
                    text=True,  # Return output as string
                    stdout=f,
                    stderr=f_err,  # Send stderr to a file
                )

        # Stop measuring time
        end_time = time.time()

        runtime = end_time - start_time
        return runtime

    except Exception as e:
        raise RuntimeError(f"An error occurred: {str(e)}")


# Example usage:
if __name__ == "__main__":
    # Delete all files starting with output_*
    files = glob.glob("output_src*")
    for file in files:
        os.remove(file)
    files = glob.glob("error_src*")
    for file in files:
        os.remove(file)
    try:
        runtimes = []
        for i in range(8):
            print("Running", i)
            runtime = run_script_with_runtime(i)
            runtimes.append(runtime)
            print("Finished", i)
            print(f"Runtimes: {runtimes}")
        # print(f"Runtime: {runtime:.2f} seconds")
        print(runtimes)

    except Exception as e:
        print(f"Error: {e}")
