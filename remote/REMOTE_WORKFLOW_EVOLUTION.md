# Evolution of the Remote Grid Search

This document tells the story of how we transformed a simple, buggy script into a robust, parallel, and fully-automated remote grid search pipeline. The final, "perfected" command is the result of solving several subtle but critical challenges. Understanding this journey is key to appreciating *why* the final workflow is so effective.

---

## From Serial & Buggy to Parallel & Robust

Our initial attempts at running a grid search were plagued by silent failures, numerical instability, and a lack of true parallelism. Here is the step-by-step evolution of the fixes that led to our successful workflow.

### 1. Achieving True Parallelism

- **Problem:** The initial script was serial. A large grid search would take days.
- **Solution:** We introduced Python's `concurrent.futures.ProcessPoolExecutor` to run multiple training jobs in parallel, controlled by a `--workers` command-line argument.

- **Problem:** When running on a GPU with PyTorch, using the default `fork` process start method is incompatible with CUDA. This led to crashes.
- **Solution:** We explicitly set the multiprocessing start method to `spawn` at the beginning of the script: `torch.multiprocessing.set_start_method("spawn")`.

- **Problem:** Even with multiple processes, we weren't seeing a linear speedup. The CPU was thrashing.
- **Solution:** We discovered that PyTorch, by default, tries to use all available CPU cores for its own internal threading. When running multiple Python processes, this leads to massive over-subscription and context-switching overhead. The critical fix was to force each worker process to use only a single thread: `torch.set_num_threads(1)`.

### 2. Ensuring Numerical Stability

- **Problem:** Certain hyperparameter combinations (especially a `ridge_lam` of 0) produced singular or ill-conditioned matrices, causing `torch.linalg.solve` to crash the entire worker process.
- **Solution:** We replaced the sensitive `torch.linalg.solve` with the more robust `torch.linalg.lstsq` (Least Squares). This function can handle these cases gracefully, allowing the grid search to continue even if some combinations are mathematically problematic.

### 3. Solving Remote Execution Bugs

- **Problem:** When running the script on a remote pod via SSH, output was heavily buffered. We wouldn't see any logs until the entire script finished, making it impossible to monitor progress or debug failures in real-time.
- **Solution:** We added `flush=True` to all `print()` statements in the grid search script. More importantly, we added the `-u` flag to the Python command (`python -u ...`), which forces unbuffered standard output.

- **Problem:** A persistent `TypeError` was crashing workers. After adding robust traceback reporting, we found that the `lam` parameter, read from the YAML config, was being interpreted as a string (e.g., `'1e-5'`) instead of a float.
- **Solution:** We added an explicit `float()` cast where the `lam` parameter was used, making the code resilient to the string-based output of the YAML parser.

### 4. Adding Automation & Early Stopping

- **Problem:** The process of setting up a remote pod, cloning the repo, installing dependencies, and pulling results was manual and error-prone.
- **Solution:** We created the `remote/` directory with helper scripts:
    - `setup.py`: Automates the entire remote pod configuration.
    - `pull_outputs.py`: Uses `rsync` to efficiently sync results back to the local machine.

- **Problem:** For some searches, we might find a "good enough" result long before all combinations have been tested, wasting expensive GPU time.
- **Solution:** You introduced a `target_mse` option in the config. If a worker achieves a validation MSE below this target, it signals the main process to cancel all other pending jobs, saving significant time and resources.

---

## The "Perfected" Remote Execution Command

This brings us to the final, battle-tested command you used:

```bash
ssh -t -i ~/.ssh/id_ed25519 -p 23202 root@157.157.221.29 "\
  cd /root/MemoryCapacitorTopologies && \
  /root/miniconda/envs/rc/bin/python -u experiments/grid_search.py --config configs/lorenz_search.yaml --workers 8"
```

**Deconstruction:**

- `ssh -t ...`: The `-t` flag allocates a pseudo-terminal, which can improve the behavior of interactive scripts and ensure clean termination.
- `-i ~/.ssh/id_ed25519 -p 23202 root@157.157.221.29`: Standard SSH connection details.
- `"..."`: The entire remote command is wrapped in quotes.
- `cd /root/MemoryCapacitorTopologies && ...`: Ensures the command runs from the correct project directory.
- `/root/miniconda/envs/rc/bin/python`: This is a **highly robust** way to run the script. Instead of relying on the shell's `PATH` or an activated conda environment, it calls the Python executable directly from its absolute path within the target environment. This eliminates any ambiguity about which Python is being used.
- `-u experiments/grid_search.py ...`: As discussed, the `-u` flag is critical for unbuffered, real-time logging.
- `--config ... --workers 8`: The standard arguments to control the grid search.

This single line encapsulates all our hard-won lessons, providing a reliable and efficient way to launch a massive, parallel grid search on a remote machine.
