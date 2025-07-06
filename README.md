# Memcapacitive Reservoir Topologies

This project provides a comprehensive, extensible, and user-friendly framework for creating, testing, and deploying memcapacitor-based neuromorphic computing models. It is designed for both local experimentation and powerful, automated grid searches on remote GPU servers.

---

## Table of Contents

1.  [Getting Started](#getting-started)
2.  [Project Philosophy](#design-philosophy-human-and-ai-readability)
3.  [Running Experiments](#running-experiments)
    *   [Local Training](#local-training-example)
    *   [Automated Remote Grid Search](#automated-remote-gpu-grid-search)
4.  [Project Structure](#project-structure)
5.  [Extensibility](#extensibility)
6.  [Scientific Best Practices](#reproducibility--scientific-best-practices)

---

## Getting Started

Follow these steps to get the project up and running on your local machine.

### 1. Clone the Repository

```bash
git clone https://github.com/WillForEternity/MemoryCapacitorTopologies.git
cd MemoryCapacitorTopologies
```

### 2. Set Up the Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run Local Tests

Verify your setup by running the verbose, self-documenting tests. These tests check core functionalities and produce visual outputs for inspection.

```bash
python tests/test_memcapacitor_verbose.py
python tests/test_topology_small_world_verbose.py
python tests/test_dataset_mackey_glass_verbose.py
python tests/test_reservoir_forward_verbose.py
```

---

## Design Philosophy: Human *and* AI Readability

This project is written so that **any human researcher or AI agent** can reason about the codebase with minimal context switching.

*   **Verbose, Natural-Language Tests**: Every behavior test prints explanatory PASS/FAIL commentary and saves plots, allowing correctness to be judged directly from the output.
*   **Self-Documenting Modules**: Each folder has a single, clear responsibility (`models/`, `datasets/`, `networks/`).
*   **Plug-and-Play Extensibility**: New topologies and datasets are discovered automatically via decorators, requiring no central registration.
*   **Clear Naming & Typing**: Functions and variables use descriptive names and type hints to make the code easy to understand and analyze.

---

## Running Experiments

### Local Training Example

This project includes a full example of training a reservoir computer to predict the Lorenz attractor. The optimal hyperparameters, found via a remote grid search, are stored in `configs/best_lorenz_config.yaml`.

To run this pre-configured example locally:

```bash
python -m training.train --config configs/best_lorenz_config.yaml --plot
```

This command trains the model and saves an interactive 3D plot of the predicted vs. actual Lorenz attractor to `training/outputs/lorenz_predictions_3d.html`.

### Automated Remote GPU Grid Search

For comprehensive hyperparameter sweeps, the repository includes a powerful orchestration script that fully automates the process on a remote GPU server (e.g., from RunPod, Vast.ai, etc.).

**For a complete guide on the remote workflow, see [`remote/run_remote_experiment.md`](remote/run_remote_experiment.md).**

#### The 3-Step Remote Workflow

1.  **Commit Your Changes**: Ensure all local code changes are pushed to GitHub.
    ```bash
    git add . && git commit -m "Your changes" && git push
    ```
2.  **Kill Old Processes (Recommended)**: Free up resources on the remote server.
    ```bash
    ssh -i ~/.ssh/id_ed25519 -p YOUR_PORT user@YOUR_IP 'pkill -f grid_search.py || true'
    ```
3.  **Launch the Orchestrator**: This script handles server provisioning, setup, execution, and results retrieval.
    ```bash
    python remote/run_remote_experiment.py
    ```

This script provides a beautifully formatted, real-time log, including a static, single-line progress bar for the grid search.

---

## Project Structure

-   `configs/`: YAML configuration files for experiments and grid searches.
-   `datasets/`: Data loading modules. Each subdirectory is a self-contained dataset.
-   `experiments/`: The main grid search driver (`grid_search.py`).
-   `models/`: Core device models, such as the `Memcapacitor`.
-   `networks/`: Reservoir computer logic and network topology generators.
-   `remote/`: Scripts for automating remote server setup and experiment execution.
-   `tests/`: Verbose, behavioral tests that produce plots and clear pass/fail messages.
-   `training/`: The core training pipeline (`train.py`) and default output directory for artifacts.

---

## Extensibility

Adding your own components is designed to be simple:

*   **Add a Topology**: Create `networks/topologies/your_topology.py`. Inside, define a function that returns a NumPy adjacency matrix and decorate it with `@register("your_name")`.
*   **Add a Dataset**: Create a `datasets/your_dataset/` directory containing a `generator.py` file that exposes a `load()` function.

The framework will automatically discover and register these new components.

---

## Reproducibility & Scientific Best-Practices

This repository follows these guidelines:

1.  **Deterministic Seeds**: All random operations accept a `random_seed` to ensure reproducibility.
2.  **Explicit Device Placement**: Tensors are explicitly placed on the correct device (`cuda` or `cpu`).
3.  **Separation of Concerns**: Data generation, model definition, training, and evaluation are in distinct, modular components.
4.  **Version-Controlled Results**: Grid search scripts save the winning network and diagnostic plots, which are small enough to be committed to version control.
5.  **Verbose Logging**: Tests and training scripts print detailed, natural-language commentary for easy auditing.
