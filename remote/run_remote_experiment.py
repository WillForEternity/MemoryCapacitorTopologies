import argparse
import os
import pty
import re
import select
import subprocess
import sys
import yaml

# --- Helpers ---

def print_color(text, color, end='\n'):
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
    sys.stdout.write(colors.get(color, "") + text + colors["end"] + end)
    sys.stdout.flush()

def run_command(command, step_name):
    """Runs a command, streaming its output. For the grid search, uses a PTY."""
    print_color(f"\n[STEP {step_name}] Running: {' '.join(command)}", "yellow")
    try:
        if step_name == "Launching Grid Search":
            # Use a manually managed PTY for full, unbuffered control over I/O.
            # This is the key to making remote tqdm work correctly.
            master_fd, slave_fd = pty.openpty()

            process = subprocess.Popen(
                command,
                preexec_fn=os.setsid,
                stdin=slave_fd,
                stdout=slave_fd,
                stderr=slave_fd,
            )
            os.close(slave_fd)

            try:
                while process.poll() is None:
                    r, _, _ = select.select([master_fd], [], [], 0.1)
                    if r:
                        try:
                            # Read raw bytes and write directly to stdout buffer
                            data = os.read(master_fd, 1024)
                            if data:
                                sys.stdout.buffer.write(data)
                                sys.stdout.buffer.flush()
                            else:
                                break  # EOF
                        except OSError:
                            break # PTY closed
            finally:
                os.close(master_fd)

            return_code = process.wait()
            if return_code != 0:
                # Add a newline to not overwrite the last line of failed output
                print_color(f"\n[✖] Command failed with exit code {return_code}.", "red")
                return False
        else:
            # Standard non-interactive command execution
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            prefix = f"[{step_name}] | "
            for line in iter(process.stdout.readline, ''):
                sys.stdout.write(f"{prefix}{line.rstrip()}\n")
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
    """Gets required inputs from the user, falling back to interactive prompts."""
    # SSH String
    ssh_string = args.ssh_string
    if not ssh_string:
        print_color("Please paste your full SSH connection string from RunPod (e.g., root@1.2.3.4 -p 12345): ", "cyan", end="")
        ssh_string = sys.stdin.readline().strip()

    # Key Path
    key_path = args.key_path
    if not key_path:
        default_key = os.path.expanduser("~/.ssh/id_ed25519")
        print_color(f"Enter the full path to your SSH private key [default: {default_key}]: ", "cyan", end="")
        key_path_input = sys.stdin.readline().strip()
        key_path = key_path_input or default_key
    
    # Config Path
    config_path = args.config
    if not config_path:
        default_config = "configs/lorenz_random_search.yaml"
        print_color(f"Enter the path to your grid search config file [default: {default_config}]: ", "cyan", end="")
        config_input = sys.stdin.readline().strip()
        config_path = config_input or default_config
        
    return {
        'ssh_string': ssh_string,
        'key_path': os.path.expanduser(key_path),
        'config_path': config_path
    }

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
    parser = argparse.ArgumentParser(description="Automated Remote Grid Search Orchestrator.")
    parser.add_argument('--ssh-string', type=str, default=None, help="Full SSH connection string from RunPod (e.g., 'root@1.2.3.4 -p 12345').")
    parser.add_argument('--key-path', type=str, default=None, help="Full path to your SSH private key.")
    parser.add_argument('--config', type=str, default=None, help="Path to the grid search YAML config file.")
    args = parser.parse_args()

    # --- Step 1: Get User Input ---
    user_input = get_user_input(args)
    if not all(user_input.values()):
        print_color("[✖] All inputs are required. Exiting.", "red")
        return

    # --- Step 2: Configure pods.yml ---
    ssh_details = parse_ssh_string(user_input['ssh_string'])
    if not ssh_details:
        return
    
    if not update_pods_yaml(ssh_details['uri'], ssh_details['port'], user_input['key_path']):
        return

    # --- Step 3: Provision the Remote Server ---
    if not run_command(['python', 'remote/setup.py'], "Provisioning Server"):
        print_color("[✖] Failed to provision the remote server. Please check the logs.", "red")
        return

    # --- Step 4: Launch the Grid Search ---
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
    if not run_command(ssh_command, "Launching Grid Search"):
        print_color("[✖] Grid search failed. Please check the logs.", "red")
        return

    # --- Step 5: Retrieve Results ---
    if not run_command(['python', 'remote/pull_outputs.py'], "Retrieving Results"):
        print_color("[✖] Failed to retrieve results. You may need to run 'python remote/pull_outputs.py' manually.", "red")
        return
    
    print_color("\n--- All Steps Complete! ---", "green")
    print_color("Your experiment results have been downloaded to the 'training/outputs' directory.", "blue")
    print_color("You can now analyze the plots and the best model found.", "blue")

if __name__ == "__main__":
    main()
