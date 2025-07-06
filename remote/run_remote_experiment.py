import argparse
import os
import pty
import re
import subprocess
import sys
import yaml
from datetime import datetime

# --- Helpers ---

def print_color(text, color):
    """Prints text in a given color."""
    colors = {
        "red": "\033[91m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "purple": "\033[95m",
        "cyan": "\033[96m",
        "end": "\033[0m",
    }
    sys.stdout.write(colors.get(color, "") + text + colors["end"] + "\n")
    sys.stdout.flush()

def run_command(command, step_name, interactive=False, prefix_output=True):
    """Runs a command, streaming output. For interactive commands, uses a PTY."""
    print_color(f"\n[STEP {step_name}] Running: {' '.join(command)}", "yellow")
    try:
        if interactive:
            # Use pty.spawn for commands that require a real TTY (like tqdm).
            # This directly connects the subprocess to our terminal, which is the
            # key to making the progress bar render correctly in-place.
            return_code = pty.spawn(command)
            if return_code != 0:
                print_color(f"\n[✖] Command failed with exit code {return_code}.", "red")
                return False
        else:
            # Use subprocess.Popen for standard, non-interactive commands.
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            if prefix_output:
                prefix = f"[{step_name}] | "
                for line in iter(process.stdout.readline, ''):
                    sys.stdout.write(f"{prefix}{line.rstrip()}\n")
                    sys.stdout.flush()
            else: # Stream raw output, for scripts that have their own formatting
                for line in iter(process.stdout.readline, ''):
                    sys.stdout.write(line)
                    sys.stdout.flush()
            
            process.stdout.close()
            rc = process.wait()
            if rc != 0:
                print_color(f"[✖] Command failed with exit code {rc}.", "red")
                return False

        print_color(f"[✔] Step completed successfully.", "green")
        return True
    except Exception as e:
        print_color(f"[✖] Failed to execute command: {e}", "red")
        return False

def get_user_input(args):
    """Gets required inputs from the user interactively if not provided as args."""
    inputs = {}
    if args.ssh_string:
        inputs['ssh_string'] = args.ssh_string
    else:
        inputs['ssh_string'] = input(print_color("Please paste your full SSH connection string from RunPod (e.g., root@1.2.3.4 -p 12345): ", "cyan"))

    if args.key_path:
        inputs['key_path'] = args.key_path
    else:
        default_key = os.path.expanduser("~/.ssh/id_ed25519")
        inputs['key_path'] = input(print_color(f"Enter the full path to your SSH private key [default: {default_key}]: ", "cyan")) or default_key

    if args.config:
        inputs['config_path'] = args.config
    else:
        inputs['config_path'] = input(print_color("Enter the path to your grid search config file (e.g., configs/lorenz_search.yaml): ", "cyan"))

    return inputs

def parse_ssh_string(ssh_string):
    """Parses the user@host and port from an SSH connection string."""
    match_uri = re.search(r'([\w-]+@(?:[\d]{1,3}\.){3}[\d]{1,3})', ssh_string)
    match_port = re.search(r'-p\s+(\d+)', ssh_string)
    
    if not match_uri or not match_port:
        print_color("[✖] Invalid SSH string. Expected format: 'user@ip.ad.dr.ess -p 12345'", "red")
        return None

    return {'uri': match_uri.group(1), 'port': match_port.group(1)}

def update_pods_yaml(uri, port, key_path):
    """Updates the remote/pods.yml file with the user's connection details."""
    pods_file = 'remote/pods.yml'
    print_color(f"\n[INFO] Updating {pods_file} with your connection details...", "blue")
    try:
        with open(pods_file, 'r') as f:
            data = yaml.safe_load(f)
        
        data['pods']['gpu1']['uri'] = uri
        data['pods']['gpu1']['port'] = int(port)
        data['pods']['gpu1']['key'] = os.path.expanduser(key_path)

        with open(pods_file, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)
        
        print_color("[✔] pods.yml updated successfully.", "green")
        return True
    except Exception as e:
        print_color(f"[✖] Failed to update {pods_file}: {e}", "red")
        return False

# --- Main Orchestrator ---

def main():
    parser = argparse.ArgumentParser(description="Fully Automated Remote Grid Search Orchestrator.")
    parser.add_argument('--ssh-string', type=str, help="Full SSH connection string from RunPod.")
    parser.add_argument('--key-path', type=str, help="Full path to your SSH private key.")
    parser.add_argument('--config', type=str, help="Path to the grid search YAML config file.")
    parser.add_argument('--no-sync', action='store_true', help="Skip the automatic git add, commit, and push.")
    args = parser.parse_args()

    # --- Step 0: Git Sync ---
    if not args.no_sync:
        commit_message = f"Auto-sync before experiment run at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        if not run_command(['git', 'add', '.'], "Staging Changes"):
            sys.exit(1)
        # Only commit if there are staged changes
        if subprocess.run(['git', 'diff', '--staged', '--quiet']).returncode != 0:
            if not run_command(['git', 'commit', '-m', commit_message], "Committing Changes"):
                sys.exit(1)
            if not run_command(['git', 'push'], "Pushing to Remote"):
                sys.exit(1)
        else:
            print_color("\n[INFO] No changes to commit. Skipping sync.", "blue")
    else:
        print_color("\n[INFO] Skipping Git sync as requested.", "blue")

    # --- Step 1: Get User Input & Configure ---
    user_input = get_user_input(args)
    if not all(user_input.values()):
        print_color("[✖] All inputs are required. Exiting.", "red")
        sys.exit(1)

    ssh_details = parse_ssh_string(user_input['ssh_string'])
    if not ssh_details:
        sys.exit(1)
    
    if not update_pods_yaml(ssh_details['uri'], ssh_details['port'], user_input['key_path']):
        sys.exit(1)

    # --- Step 2: Kill Lingering Remote Processes ---
    kill_command = [
        "ssh", "-T",  # Disable pseudo-tty allocation for non-interactive command
        "-i", os.path.expanduser(user_input['key_path']),
        "-p", ssh_details['port'], ssh_details['uri'],
        # Wrap in bash -c for robustness
        "bash -c \"pkill -f 'grid_search.py' || true\""
    ]
    if not run_command(kill_command, "Cleaning Remote Processes"):
        sys.exit(1)

    # --- Step 3: Provision the Remote Server ---
    if not run_command(['python', 'remote/setup.py'], "Provisioning Server", prefix_output=False):
        print_color("[✖] Failed to provision the remote server. Please check the logs.", "red")
        sys.exit(1)

    # --- Step 4: Launch the Grid Search (Interactive) ---
    ssh_command = [
        "ssh", "-t",  # Force TTY allocation for interactive tqdm rendering
        "-i", os.path.expanduser(user_input['key_path']),
        "-p", ssh_details['port'],
        ssh_details['uri'],
        (
            f"cd /root/MemoryCapacitorTopologies && "
            f"OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 "
            f"/root/miniconda/envs/rc/bin/python -u experiments/grid_search.py {user_input['config_path']}"
        )
    ]
    if not run_command(ssh_command, "Launching Grid Search", interactive=True):
        print_color("[✖] Grid search failed. Please check the logs.", "red")
        sys.exit(1)

    # --- Step 5: Retrieve Results ---
    if not run_command(['python', 'remote/pull_outputs.py'], "Retrieving Results", prefix_output=False):
        print_color("[✖] Failed to retrieve results. You may need to run 'python remote/pull_outputs.py' manually.", "red")
        sys.exit(1)
    
    print_color("\n--- All Steps Complete! ---", "green")
    print_color("Your experiment results have been downloaded to the 'training/outputs' directory.", "blue")
    print_color("You can now analyze the plots and the best model found.", "blue")

if __name__ == "__main__":
    main()
