#!/usr/bin/env python3
"""Bootstrap one or more RunPod instances given an inventory file.

Usage
-----
    python remote/setup.py [pods.yml]

The script runs LOCALLY.  It reads the YAML inventory (default: ``remote/pods.yml``)
and then for each host runs a small shell bootstrap via ``ssh``:

1. Ensure git and (mini)conda are present.
2. Create / update the specified conda environment.
3. Clone (or ``git pull``) the project repository at the requested branch.
4. Install Python dependencies from the repository's requirements.txt file.

After finishing it prints a short success / failure table.

It relies only on standard Python + ``PyYAML``.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import yaml

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


class Pod:
    def __init__(self, name: str, cfg: Dict):
        self.name = name
        self.uri = cfg["uri"]  # e.g. root@1.2.3.4
        self.port = cfg.get("port", 22)
        self.key = cfg.get("key")  # path on *local* machine
        self.role = cfg.get("role", "gpu")

    # Compose the ssh command prefix
    def ssh_prefix(self) -> List[str]:
        parts = [
            "ssh",
            "-o",
            "StrictHostKeyChecking=no",
            "-p",
            str(self.port),
        ]
        if self.key:
            parts += ["-i", str(Path(self.key).expanduser())]
        parts.append(self.uri)
        return parts


# -----------------------------------------------------------------------------


def run_remote(pod: Pod, remote_cmd: str, print_cmd: bool = True):
    """Runs a command on a remote pod and streams the output in real-time."""
    cmd = pod.ssh_prefix() + [remote_cmd]
    if print_cmd:
        print(f"[*] Running remote command: {remote_cmd}")

    # Use Popen to stream output in real-time. The parent script
    # (run_remote_experiment.py) will capture and prefix this output.
    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
    )

    if process.stdout:
        for line in iter(process.stdout.readline, ""):
            print(line, end="", flush=True)

    return process.wait()


def main(path: Path):
    with path.open() as f:
        inv = yaml.safe_load(f)

    repo = inv["repo"]
    branch = inv.get("branch", "main")
    py_ver = inv.get("python", "3.10")
    env = inv.get("conda_env", "rc")

    pods = [Pod(name, cfg) for name, cfg in inv["pods"].items()]

    for pod in pods:
        print(f"\n=== Bootstrapping {pod.name} ({pod.uri}) ===")
        # 1. Ensure git present
        run_remote(
            pod,
            "sudo apt-get update -qq && sudo apt-get install -y git wget bzip2 >/dev/null 2>&1 || true",
            print_cmd=False,
        )

        # 2. Ensure conda exists, otherwise install Miniconda silently
        run_remote(
            pod,
            (
                "command -v conda >/dev/null 2>&1 || ("
                "wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/mc.sh && "
                "bash /tmp/mc.sh -b -p $HOME/miniconda && "
                "echo 'export PATH=$HOME/miniconda/bin:$PATH' >> ~/.bashrc)"
            ),
            print_cmd=False,
        )

        # 3. Create env if needed
        create_env = f"$HOME/miniconda/bin/conda create -y -n {env} python={py_ver}"
        return_code = run_remote(pod, create_env)
        if return_code != 0:
            print(
                f"[!] Conda create failed with exit code {return_code}", file=sys.stderr
            )

        # 4. Clone / update repo first to get requirements.txt
        clone_cmd = (
            f"[ ! -d MemoryCapacitorTopologies ] && "
            f"git clone --depth 1 -b {branch} {repo} || "
            f"(cd MemoryCapacitorTopologies && git fetch origin {branch} && git checkout {branch} && git pull)"
        )
        return_code = run_remote(pod, clone_cmd)
        if return_code != 0:
            print(f"[!] Git clone/pull failed with exit code {return_code}", file=sys.stderr)
            continue  # Can't proceed if repo is not there

        # 5. Install python deps from requirements file
        pip_install = f"$HOME/miniconda/bin/conda run -n {env} pip install -r MemoryCapacitorTopologies/requirements.txt"
        return_code = run_remote(pod, pip_install)
        if return_code != 0:
            print(f"[!] Pip install failed with exit code {return_code}!", file=sys.stderr)
        else:
            print(f"[✔] {pod.name} ready")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("inventory", nargs="?", default="remote/pods.yml", help="Path to pods.yml")
    args = ap.parse_args()
    main(Path(args.inventory).expanduser())
