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
       --config configs/lorenz_search.yaml \
       --workers 8
```

Flags explained:

- `--config` – YAML describing search space and default values.
- `--workers` – number of parallel Python processes.  For a single GPU choose the number of **physical** CPU cores to keep utilisation high without context-switch overhead.

**Tip:** add `--name my_run` to override the timestamped output directory.

Real-time logs stream to the terminal; tracebacks from worker processes are forwarded to the main process for easy debugging.

---

## 4  Retrieve Results (`remote/pull_outputs.py`)

Back on **local** machine:

```bash
python remote/pull_outputs.py --pod runpod_gpu1
```

This performs an `rsync` over SSH:

* `~/MemoryCapacitorTopologies/training/outputs/` ➜ `training/outputs/` (local)

The best network (`*.npz`) and any figures (`*.png`, `*.html`) will appear inside a sub-folder such as `training/outputs/lorenz_search_results/`.

---

## 5  Run Final Validation Locally

```bash
python -m training.train --config configs/best_lorenz_config.yaml --plot
```

The command loads the saved `.npz`, re-trains (if desired), and writes fresh figures to the same output directory.

---

## 6  Common Issues & Fixes

| Symptom | Cause | Fix |
|---------|-------|-----|
| `ModuleNotFoundError: yaml` on local pull | `pyyaml` not in your *local* env | `pip install pyyaml` or use the project’s conda env |
| `conda: command not found` or `conda.sh: No such file or directory` on remote | The remote shell is non-interactive and didn't load Conda's path. | Use the `bash -l -c "..."` wrapper from the Quick Start guide. If that fails, find your `conda.sh` with `find ~/ -name "conda.sh"` and source it manually. |
| CUDA OOM on pod | Too many workers or large `n_nodes` | Reduce `--workers` or search space size |
| Numerical instability (singular matrix) | ill-conditioned data | The code uses `torch.linalg.lstsq`; ensure `ridge_lam` includes non-zero values |

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
