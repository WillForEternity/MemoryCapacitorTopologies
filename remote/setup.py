#!/usr/bin/env python3
"""Bootstrap one or more RunPod instances given an inventory file.

Usage
-----
    python remote/setup.py [pods.yml]

The script runs LOCALLY.  It reads the YAML inventory (default: ``remote/pods.yml``)
and then for each host runs a small shell bootstrap via ``ssh``:

1. Ensure git and (mini)conda are present.
2. Create / update the specified conda environment.
3. Install lightweight Python packages that are **not** present on the base
   image (networkx, matplotlib, tqdm, PyYAML).
4. Clone (or ``git pull``) the project repository at the requested branch.

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


def run_remote(pod: Pod, remote_cmd: str):
    cmd = pod.ssh_prefix() + [remote_cmd]
    return subprocess.run(cmd, capture_output=True, text=True)


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
        )

        # 3. Create env if needed
        create_env = f"$HOME/miniconda/bin/conda create -y -n {env} python={py_ver} || true"
        run_remote(pod, create_env)

        # 3. Install python deps (light extras only)
        pip_install = (
            f"$HOME/miniconda/bin/conda run -n {env} pip install --quiet --upgrade networkx matplotlib tqdm pyyaml"
        )
        run_remote(pod, pip_install)

        # 4. Clone / update repo
        clone_cmd = (
            f"[ ! -d MemoryCapacitorTopologies ] && "
            f"git clone --depth 1 -b {branch} {repo} || "
            f"(cd MemoryCapacitorTopologies && git fetch origin {branch} && git checkout {branch} && git pull)"
        )
        res = run_remote(pod, clone_cmd)
        if res.returncode == 0:
            print(f"[✔] {pod.name} ready")
        else:
            print(f"[✘] {pod.name} failed. stderr:\n{res.stderr}", file=sys.stderr)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("inventory", nargs="?", default="remote/pods.yml", help="Path to pods.yml")
    args = ap.parse_args()
    main(Path(args.inventory).expanduser())
