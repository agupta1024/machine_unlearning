import subprocess
import time
import sys

def launch():
    # The command to run your script
    cmd = [sys.executable, "main_gpt.py"] 

    while True:
        print(f"\n>>> Launching {cmd[1]}...")
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