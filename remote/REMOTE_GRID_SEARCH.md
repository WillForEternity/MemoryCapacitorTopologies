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
| CUDA OOM on pod | Too many workers or large `n_nodes` | Reduce `--workers` or search space size |
| Numerical instability (singular matrix) | ill-conditioned data | The code uses `torch.linalg.lstsq`; ensure `ridge_lam` includes non-zero values |

---

## Quick Start

```bash
# local machine
python remote/setup.py --pod runpod_gpu1  # one-time
ssh -i ~/.ssh/id_ed25519 -p 23202 root@157.157.221.29 "\
  source ~/miniforge3/etc/profile.d/conda.sh && \
  conda activate rc && \
  cd ~/MemoryCapacitorTopologies && \
  python experiments/grid_search.py --config configs/lorenz_search.yaml --workers 8"

# after it finishes
python remote/pull_outputs.py --pod runpod_gpu1
```

Enjoy your accelerated hyper-parameter sweeps!  🎉

---

## The Road to a Perfect Grid Search: A Step-by-Step History

The final, perfected command you ran is the result of a detailed, iterative process of debugging and optimization. This section documents that journey, explaining the key problems we solved to get from a failing script to a fast, robust, and fully automated workflow.

### The Perfected Command

```bash
ssh -t -i ~/.ssh/id_ed25519 -p 23202 root@157.157.221.29 "\
  cd /root/MemoryCapacitorTopologies && \
  /root/miniconda/envs/rc/bin/python -u experiments/grid_search.py --config configs/lorenz_search.yaml --workers 8"
```

Here’s how we made every part of that command work perfectly:

### Step 1: Solving Instability and Crashes

*   **Problem:** The initial grid search was unstable. It frequently crashed due to `TypeError` and `singular matrix` errors in PyTorch.
*   **Investigation:** We added detailed logging to the `_ridge_regression` function. The logs revealed two root causes:
    1.  The regularization parameter, `lam`, was being read from the YAML config as a string (e.g., `'1e-5'`) instead of a `float`, causing a `TypeError`.
    2.  For some hyper-parameters, `torch.linalg.solve` was failing on ill-conditioned matrices.
*   **Solution:**
    1.  We added an explicit `float()` cast to the `lam` parameter inside the training script.
    2.  We replaced the sensitive `torch.linalg.solve` with the more numerically stable `torch.linalg.lstsq`, which gracefully handles these edge cases.

### Step 2: Achieving True Parallelism

*   **Problem:** Even when the script ran, it wasn't significantly faster with more workers. The CPU was thrashing, with processes competing for resources instead of running in parallel.
*   **Investigation:** This is a classic PyTorch multiprocessing issue. By default, PyTorch tries to use multiple threads per process, which leads to massive overhead when you're also using multiple processes.
*   **Solution:** We added `torch.set_num_threads(1)` at the beginning of each worker's execution. This forces each of the 8 worker processes onto a single CPU core, eliminating thread contention and enabling true, efficient parallelism.

### Step 3: Enabling Real-Time Debugging

*   **Problem:** It was impossible to debug on the remote pod because `print` statements and logs would only appear after the entire script finished. This was caused by Python's default output buffering.
*   **Investigation:** We found that using `conda run` was a primary cause of the buffering.
*   **Solution:** The perfected command bypasses `conda run` and invokes the Python interpreter directly from the conda environment's `bin` path. We also added the `-u` flag to ensure unbuffered output.
    *   **This is the key to the command:** `/root/miniconda/envs/rc/bin/python -u ...` ensures every `print` statement and log message appears on your local terminal in real-time.

### Step 4: Adding Early Stopping (Your Contribution!)

*   **Problem:** The grid search would always run through all 432 combinations, even if a great result was found early on.
*   **Solution:** You implemented an elegant early-stopping mechanism. By adding a `target_mse` to the config, the script now monitors the best validation score and automatically cancels all remaining jobs once the target is met, saving significant time and compute cost.

By systematically identifying and fixing these issues, we transformed the grid search from a broken, slow, and opaque script into the highly optimized, robust, and transparent workflow you have today.
