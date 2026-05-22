import subprocess
import time
import sys
import argparse


def _detect_process_count():
    """Detect a safe process count for accelerate launch."""
    try:
        import torch
        gpu_count = torch.cuda.device_count()
        return max(1, int(gpu_count))
    except Exception:
        return 1


def build_cmd(run_params, run_eval, train_with_ga_loss, use_accelerate=False, num_processes=None):
    cmd = [sys.executable]
    if use_accelerate:
        if num_processes is None:
            num_processes = _detect_process_count()
        cmd.extend(["-m", "accelerate.commands.launch"])
        cmd.extend(["--num_processes", str(num_processes)])
    cmd.extend([
        "main.py",
        "--unlearn_run", str(run_params[13]),
        "--epochs", str(run_params[12]),
        "--lambda_gk", str(run_params[11]),
        "--lambda_retain", str(run_params[10]),
        "--lambda_forget", str(run_params[9]),
        "--ulr", str(run_params[8]),

        "--oversample", str(run_params[7]),
        "--num_forget_ex", str(run_params[6]),
        "--lora_r", str(run_params[5]),
        "--num_epochs", str(run_params[4]),
        "--lr", str(run_params[3]),
        "--run_name", str(run_params[2]),
        "--model", str(run_params[1]),
        "--proj_name", str(run_params[0]),
    ])
    if run_eval:
        cmd.append("--run_eval")
    if train_with_ga_loss:
        cmd.append("--train_with_ga_loss")
    return cmd

def launch():
    parser = argparse.ArgumentParser(description="Launch evaluation runs")
    parser.add_argument("--accelerate", action="store_true", help="Run main.py via accelerate launch")
    parser.add_argument("--num_processes", type=int, default=None, help="Accelerate process count")
    args = parser.parse_args()

    # Format: "Proj Model RunName LR saliency_pc lora_r forget_lambda retain_lambda GK_lambda epoch"
    ext = 'eval'
    run_eval = True
    train_with_ga_loss = False
    epochs = 4
    
    run_params_list = [
        # [f'tofu-llama_{ext}', 'llama1b', 'r64_3e-5_20ex', 3e-5, 10, 64, 20, 20,
        #  5e-6, 1.0, 0.7, 1.2, 5, 'com_r64_3e-5_20ex_u_r64-5e-6_3e'],
         [f'tofu-llama_{ext}', 'llama1b-bnb-4bit', 'r64_3e-5_20ex', 3e-5, 10, 64, 20, 20,
         5e-6, 1.0, 0.7, 1.2, 5, 'u_r64-5e-6_8e'],
        # [f"tofu-run_gemma_{ext}", 'gemma',  'com_r128_3e-5_20ex', 3e-5, 10, 128, 20, 50,
        #  5e-6, 0.05, 2.0, 0.7, 1.2, 3, 'com_r128_3e-5_20ex_u_r128-5e-6_3e'],
        # [f"tofu-run_gemma_{ext}", 'gemma',  'com_r64_3e-5_20ex', 3e-5, 10, 64, 20, 50,
        #  5e-6, 0.05, 2.0, 0.7, 1.2, 3, 'com_r64_3e-5_20ex_u_r64-5e-6_3e'],
        # [f"tofu-run_gemma_{ext}", 'gemma',  'com_r64_3e-5_50ex', 3e-5, 10, 64, 50, 20,
        #  5e-6, 0.05, 2.0, 0.7, 1.2, 3, 'com_r64_3e-5_50ex_u_r64-5e-6_3e'],
        # [f"tofu-run_gemma_{ext}", 'gemma',  'com_r64_3e-5_Allex', 3e-5, 10, 64, -1, 1,
        #  5e-6, 0.05, 2.0, 0.7, 1.2, 3, 'com_r64_3e-5_Allex_u_r64-5e-6_3e'],
                    ]

    for run_params in run_params_list:
        retry_num = 2
        cmd = build_cmd(
            run_params=run_params,
            run_eval=run_eval,
            train_with_ga_loss=train_with_ga_loss,
            use_accelerate=args.accelerate,
            num_processes=args.num_processes,
        )
        cmd_str = " ".join(cmd)

        while retry_num:
            retry_num -= 1
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