import subprocess
import time
import sys

def launch():
    # The command to run your script
    run_params = ['gpt2', 'low_rank', 3e-5, 0.05, 128, 0.7, 0.2]
    cmd_str = (
                f"main.py --run_name {run_params[1]} --lr {run_params[2]} "
                f"--mask {run_params[3]} --lora_r {run_params[4]} "
                f"--lambda_retain {run_params[5]} --lambda_gk {run_params[6]} "
                f"--model {run_params[0]} --epochs 6"
            )
    cmd = [sys.executable] + cmd_str.split() 
    while True:
        print(f"\n>>> Launching {cmd_str}...")
        process = subprocess.Popen(cmd)
        process.wait() # Wait for the script to finish or crash

        if process.returncode == 0:
            print(">>> Script finished successfully.")
            break
        else:
            # You can't specifically catch OOM here, but usually, 
            # a crash in a GPU script is OOM-related.
            print(f"\n[!] Script crashed with exit code {process.returncode}.")
            print(">>> Waiting 10 seconds before restarting...")
            time.sleep(10)

if __name__ == "__main__":
    launch()    