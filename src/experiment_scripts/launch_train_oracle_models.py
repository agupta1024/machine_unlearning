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

# Format: "Proj Model RunName LR epochs lora_r num_forget_ex oversample"
def build_cmd(run_params, train_oracle, use_accelerate=False, num_processes=None):
    cmd = [sys.executable]
    if use_accelerate:
        if num_processes is None:
            num_processes = _detect_process_count()
        cmd.extend(["-m", "accelerate.commands.launch"])
        cmd.extend(["--num_processes", str(num_processes)])
    cmd.extend([
        "main.py",
        "--oversample", str(run_params[7]),
        "--num_forget_ex", str(run_params[6]),
        "--lora_r", str(run_params[5]),
        "--num_epochs", str(run_params[4]),
        "--lr", str(run_params[3]),
        "--run_name", str(run_params[2]),
        "--model", str(run_params[1]),
        "--proj_name", str(run_params[0]),
    ])
    if train_oracle:
        cmd.append("--train_oracle")
    return cmd

def launch():
    parser = argparse.ArgumentParser(description="Launch oracle training runs")
    parser.add_argument("--accelerate", action="store_true", help="Run main.py via accelerate launch")
    parser.add_argument("--num_processes", type=int, default=None, help="Accelerate process count")
    args = parser.parse_args()

    # The command to run your script
    ext = 'training_oracle'
    train_oracle = True
    retry_num = 2
    epochs = 10
    run_params_list = [
    #     [f"tofu-unlearning_gemma_{ext}", 'gemma',  'r128_3e-5_10ex', 3e-5, epochs, 128, 10, 50],
    #     [f"tofu-unlearning_gemma_{ext}", 'gemma',  'r64_3e-5_10ex', 3e-5, epochs, 64, 10, 50],

        [f'tofu-llama_{ext}', 'llama1b', 'r64_3e-5_20ex', 3e-5, epochs, 64, 20, 20],
        # [f'tofu-llama_{ext}', 'llama1b', 'r128_3e-5_20ex', 3e-5, epochs, 128, 20, 20],
                    # [f"tofu-unlearning_qwen_{ext}", 'qwen',  'r128', 3e-5, 0.05, 128, 1.0, 0.7, 1.2, epochs],
                    # [f"tofu-unlearning_qwen_{ext}", 'qwen',  'r256_2e-6', 2e-6, 15, 256, 10, 50],
                    # [f"tofu-unlearning_qwen_{ext}", 'qwen',  'r256_5e-6', 5e-6, 15, 256, 10, 50],
                    # [f"tofu-unlearning_qwen_{ext}", 'qwen',  'r256_1e-5', 1e-5, 15, 256, 10, 50],
                    # [f"tofu-unlearning_qwen_{ext}", 'qwen',  'r64_1e-6_10ex', 1e-6, 17, 64, 10, 50],
                    # [f"tofu-unlearning_qwen_{ext}", 'qwen',  'r256_1e-5_30ex', 1e-5, 15, 256, 30, 50],
                    # [f"tofu-unlearning_qwen_{ext}", 'qwen',  'r64', 3e-5, 0.05, 64, 1.0, 0.7, 1.2, epochs],

                    # [f"tofu-unlearning_gemma_{ext}", 'gemma',  'r128', 3e-5, 0.05, 128, 1.0, 0.7, 1.2, epochs],
                    # [f"tofu-unlearning_gemma_{ext}", 'gemma',  'r256', 3e-5, 0.05, 256, 1.0, 0.7, 1.2, epochs],
                    # [f"tofu-unlearning_gemma_{ext}", 'gemma',  'r64', 3e-5, 0.05, 64, 1.0, 0.7, 1.2, epochs],

                    # [f"tofu-unlearning_phi_{ext}", 'phi', 'r128', 3e-5, 0.05, 128, 1.0, 0.7, 1.2, epochs],
                    # [f"tofu-unlearning_phi_{ext}", 'phi',  'r256', 3e-5, 0.05, 256, 1.0, 0.7, 1.2, epochs],
                    # [f"tofu-unlearning_phi_{ext}", 'phi',  'r64', 3e-5, 0.05, 64, 1.0, 0.7, 1.2, epochs],

                    # [f'tofu-unlearning_llama_{ext}', 'llama1b', 'r128', 3e-5, 0.05, 128, 1.0, 0.7, 1.2, epochs],
                    # [f"tofu-unlearning_llama_{ext}", 'llama1b',  'r256', 3e-5, 0.05, 256, 1.0, 0.7, 1.2, epochs],
                    # [f"tofu-unlearning_llama_{ext}", 'llama1b',  'r64', 3e-5, 0.05, 64, 1.0, 0.7, 1.2, epochs],
                    ]

    for run_params in run_params_list:
        retry_num = 2
        cmd = build_cmd(
            run_params=run_params,
            train_oracle=train_oracle,
            use_accelerate=args.accelerate,
            num_processes=args.num_processes,
        )
        cmd_str = " ".join(cmd)

        while retry_num:
            # retry_num = retry_num -1
            print(f"\n>>> Launching {cmd_str}...{retry_num}")
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