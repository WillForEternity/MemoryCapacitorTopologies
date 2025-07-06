# Fully Automated Remote Experiment Workflow

This guide provides the definitive, end-to-end workflow for running a hyperparameter search on a remote GPU server. The `run_remote_experiment.py` script automates every step, from provisioning the server to launching the grid search and retrieving the results.

---

## The 3-Step Workflow

Each time you want to run a new experiment with updated code, follow these three simple steps.

### Step 1: Commit Your Local Changes

Before launching an experiment, ensure all your local code changes are saved and pushed to the GitHub repository. The remote server will pull this latest version, so this step is critical for your changes to take effect.

From your project's root directory, run:

```bash
# Stage all your changes
git add .

# Commit them with a descriptive message
git commit -m "Your descriptive message about the changes"

# Push the changes to the main branch on GitHub
git push
```

### Step 2: Kill Any Lingering Remote Processes (Optional but Recommended)

To guarantee a clean slate and prevent conflicts from previous runs, it's good practice to terminate any old processes that might still be running on the remote server. 

You can do this with a single SSH command. **Replace the SSH connection string with your own.**

```bash
ssh -i ~/.ssh/id_ed25519 -p YOUR_PORT user@YOUR_IP 'pkill -f grid_search.py || true; pkill -f setup.py || true'
```

*   `YOUR_PORT`: Your RunPod SSH port.
*   `user@YOUR_IP`: Your RunPod SSH user and IP address.
*   The `|| true` part ensures the command doesn't fail if no processes are found.

### Step 3: Run the Automated Experiment Script

This is the final step. The script will handle everything else: provisioning the server, running the grid search, and downloading the results. It will prompt you for your SSH details and the path to your experiment configuration file.

```bash
python remote/run_remote_experiment.py
```

Alternatively, you can provide the arguments directly on the command line to skip the interactive prompts:

```bash
python remote/run_remote_experiment.py \
    --ssh-string 'user@YOUR_IP -p YOUR_PORT' \
    --key-path '~/.ssh/id_ed25519' \
    --config 'configs/your_experiment_config.yaml'
```

And that's it! The script will now execute, showing you a beautifully formatted, real-time log of the entire process, complete with a static, single-line progress bar for the grid search.

---

## One-Time Server Setup

If this is your first time using this system, you will need to perform a one-time setup to configure your remote GPU server. Please follow the detailed instructions in the `README.md` in the project root to:

1.  Rent a GPU server from a provider like RunPod.
2.  Generate and add an SSH key for secure, passwordless access.
3.  Ensure you have the necessary local tools like `git` and `conda` installed.

Once your server is running and accessible via SSH, the automated workflow above is all you need.


```bash
python remote/run_remote_experiment.py --ssh-string "root@157.157.221.29 -p 23202" --key-path "/Users/willnorden/.ssh/id_ed25519" --config "configs/mackey_glass_random_search.yaml"
```
