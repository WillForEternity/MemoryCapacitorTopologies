# Remote GPU Grid Search Workflow

This guide explains **exactly** how to spin up a fresh GPU (or CPU) pod on a cloud provider such as RunPod, clone the repository, install dependencies, execute a YAML-driven grid search, and synchronise the results back to your local machine.  The process is fully automated – once the pod is running you can kick off the search with a single command.

> **TL;DR** – Already have a pod?  Jump to **Quick Start**.

---

## 0  Prerequisites

| Item | Notes |
|------|-------|
| SSH key on your **local** machine | The pod will be accessed via key-based auth (no passwords).  The default key is `~/.ssh/id_ed25519`. |
| Python ≥ 3.10 + `conda` (recommended) **on the pod** | The `remote/setup.py` script will install Miniforge automatically if Conda is missing. |
| Internet egress from the pod | Required to `git clone` the repo and `pip install` dependencies. |
| An entry in `remote/pods.yml` | Describes the pod’s SSH endpoint(s) so the helper scripts know where to connect. |

Example `pods.yml`:
```yaml
runpod_gpu1:
  host: 157.157.221.29
  port: 23202
  user: root              # RunPod default
  key: ~/.ssh/id_ed25519  # local path to *private* key
```

---

## 1  Provision a RunPod Instance

1. Log into the RunPod dashboard.
2. Launch a **Secure Cloud GPU** (Ubuntu 22.04, CUDA 11 or later).
3. Enable `SSH` and note the `IP`, `Port`, and **root** username.
4. Add your **public** SSH key in the *SSH Keys* panel.
5. Wait for the pod status to turn **Running**.

---

## 2  Automated One-Time Setup (`remote/setup.py`)

From your *local* project root:

```bash
python remote/setup.py --pod runpod_gpu1
```

What this does:

1. **SSH → pod**.
2. Installs **Miniforge + conda** (if absent) and creates env `rc`.
3. `git clone https://github.com/WillForEternity/MemoryCapacitorTopologies.git` into `~/MemoryCapacitorTopologies` on the pod.
4. Installs Python requirements inside the env.
5. Prints the exact command to start a grid search.

You can rerun the script at any time – it is idempotent and will simply `git pull` if the repo already exists.

---

## 3  Launch the Grid Search

SSH into the pod (or let `setup.py` do it for you) and run:

```bash
conda activate rc
cd ~/MemoryCapacitorTopologies
python experiments/grid_search.py \

Once logged in, update the package manager and install `git` and `wget`:

```bash
apt-get update && apt-get install -y git wget
```

### Step 3: Install Miniconda

1.  Download the Miniconda installer. We'll install it to `/root/miniconda` to match the target command.
    ```bash
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /root/miniconda.sh
    ```
2.  Run the installer. The `-b` flag runs in batch mode (no prompts) and `-p` specifies the installation path.
    ```bash
    bash /root/miniconda.sh -b -p /root/miniconda
    ```
3.  Initialize your shell to use Conda. This is crucial.
    ```bash
    /root/miniconda/bin/conda init bash
    ```
4.  **IMPORTANT:** You must now **exit and log back in** for the `conda` command to be available in your path.

### Step 4: Clone the Project Repository

Clone the project code into the `/root` directory:

```bash
cd /root
git clone https://github.com/WillForEternity/MemoryCapacitorTopologies.git
```

### Step 5: Create the Conda Environment

1.  Navigate into the cloned repository:
    ```bash
    cd /root/MemoryCapacitorTopologies
    ```
2.  Create the project's Conda environment (`rc`) using the provided `environment.yml` file. This installs all required Python packages (PyTorch, NumPy, etc.).
    ```bash
    /root/miniconda/bin/conda env create -f environment.yml
    ```
    This step may take several minutes.

### Step 6: Run the Grid Search

You are now ready. The remote environment is fully configured.

From your **local machine**, you can now execute the target command. This single line connects to the pod and launches the workload.

-   **-t**: Allocates a pseudo-terminal, which helps ensure the script runs correctly.
-   **-i**: Specifies your private key.
-   **-p**: Specifies the port.
-   **"..."**: The command to run on the remote machine.
    -   `cd /root/MemoryCapacitorTopologies`: Navigates to the project directory.
    -   `/root/miniconda/envs/rc/bin/python`: Uses the **exact** Python executable from our `rc` environment. This is the most reliable way to ensure the correct dependencies are used, bypassing any shell path issues.
    -   `-u`: Unbuffered output, so you see results in real-time.
    -   `experiments/grid_search.py configs/lorenz_search.yaml`: The script to run and its configuration file.

```bash
# Replace with your pod's details
SSH_USER="root"
SSH_HOST="157.157.221.29"
SSH_PORT="23202"
SSH_KEY="~/.ssh/id_ed25519"

# The final, working command
ssh -t -p $SSH_PORT $SSH_USER@$SSH_HOST -i $SSH_KEY \
    "cd /root/MemoryCapacitorTopologies && \
     /root/miniconda/envs/rc/bin/python -u experiments/grid_search.py configs/lorenz_search.yaml"
```

This command will now work perfectly, and you will see the grid search progress streaming in your local terminal.

---

## Quick Start

This is the most robust way to launch a remote grid search in a single command.

```bash
# From your LOCAL machine:

# 1. (One-time) Set up the pod:
python remote/setup.py --pod runpod_gpu1

# 2. Launch the grid search using a login shell wrapper:
ssh -i ~/.ssh/id_ed25519 -p 23202 root@157.157.221.29 'bash -l -c "\
  cd ~/MemoryCapacitorTopologies && \
  git pull --quiet && \
  conda activate rc && \
  python experiments/grid_search.py --config configs/lorenz_random_search.yaml --workers 32"'

# 3. (After it finishes) Pull the results:
python remote/pull_outputs.py --pod runpod_gpu1
```

**Why `bash -l -c`?** The `-l` flag makes Bash act as a **login shell**, which forces it to load startup files like `~/.profile` or `~/.bash_profile`. This is where `conda init` places its configuration, making this command work automatically without needing to know the exact path to `conda.sh`.

Enjoy your accelerated hyper-parameter sweeps!  🎉
