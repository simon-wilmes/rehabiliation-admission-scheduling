import subprocess
import time

# how often to call script?
repetition_time = 300  # 1h

start_time = -1000000

while True:
    if time.time() - start_time > repetition_time:
        # Execute run_scripts.py
        result = subprocess.run(
            ["python", "cluster/run_scripts.py", "-y"], stdout=subprocess.PIPE
        )  # -y flag to skip the confirmation
        output = result.stdout.decode("utf-8")
        print(time.strftime("%Y-%m-%d %H:%M:%S"))
        print(output.split("\n")[-2:])
        start_time = time.time()
    time.sleep(60)
