# Memcapacitive Reservoir Topologies

This repository provides a comprehensive framework for designing, testing, and deploying neuromorphic computing models based on memcapacitive devices. It features a modular architecture, a powerful automated remote training system, and a strong emphasis on reproducibility and extensibility.


*^Example of a Lorenz attractor predicted by a model trained with this framework.*^

---

## Key Features

- **Modular & Extensible:** Easily add new network topologies, datasets, and device models. The framework automatically discovers and registers new components.
- **Powerful Automation:** A single script (`remote/run_remote_experiment.py`) orchestrates the entire remote workflow: provisioning a GPU server, running a parallelized grid search, and downloading the results.
- **Beautiful & Informative CLI:** Get real-time, beautifully formatted output for all steps, including a clean, static progress bar for remote training.
- **Reproducibility Focused:** Experiments are driven by version-controlled YAML configuration files, and all sources of randomness are controlled by explicit seeds.
- **Rich Diagnostics:** Generate interactive 3D plots and detailed performance metrics for easy analysis.
- **AI-Friendly Design:** The codebase is written to be highly readable for both humans and AI agents, with verbose logging and a clear, single-responsibility structure.

---

## Getting Started (Local)

Follow these steps to set up the project on your local machine.

### 1. Clone the Repository

```bash
git clone https://github.com/WillForEternity/MemoryCapacitorTopologies.git
cd MemoryCapacitorTopologies
```

### 2. Set Up the Environment

We recommend using a Python virtual environment.

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Run a Local Training Example

Run a pre-configured experiment to train a model on the Lorenz attractor and generate a plot of the results.

```bash
python -m training.train --config configs/best_lorenz_config.yaml --plot
```

This will save an interactive 3D plot to `training/outputs/lorenz_predictions_3d.html`.

---

## Automated Remote Grid Search

For computationally intensive hyperparameter searches, this repository provides a fully automated system for running experiments on a remote GPU server (e.g., from RunPod, Vast.ai, etc.).

**The entire workflow is handled by a single script.** For a complete guide on the simple, 3-step process (commit, kill, run), please see:

➡️ **[Full Guide: Automated Remote Experiment Workflow](remote/run_remote_experiment.md)**

---

## Project Structure

```
├── configs/              # YAML configuration files for experiments and searches.
├── datasets/             # Data loading and generation modules (e.g., Lorenz, Mackey-Glass).
├── experiments/          # The core grid search driver script.
├── models/               # Core device models (e.g., Memcapacitor).
├── networks/             # Reservoir implementation and network topology generators.
├── remote/               # Scripts for automating remote server setup and execution.
├── tests/                # Verbose, human-readable tests for all components.
├── training/             # The main training pipeline and output directory.
└── README.md             # You are here!
```

---

## Extensibility

This framework is designed to be easily extended.

### Adding a New Topology

1.  Create a new file in `networks/topologies/`, e.g., `my_topology.py`.
2.  Inside, define a function that returns a NumPy adjacency matrix.
3.  Decorate it with `@register("my_topology_name")`.

That's it! The framework will automatically discover it, and you can now use `"my_topology_name"` in your configuration files.

### Adding a New Dataset

1.  Create a new directory in `datasets/`, e.g., `my_dataset/`.
2.  Inside, add a `generator.py` file that exposes a `load()` function returning your data.
3.  Add a minimal `__init__.py` in the same directory that re-exports the `load` function (`from .generator import load`).

Your new dataset is now available to the framework.

---

## Citation

If you use this framework in your research, please consider citing it. (Citation format to be added).
