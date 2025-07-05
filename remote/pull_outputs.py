#!/usr/bin/env python3
"""Pull experiment artefacts (figures, .npz, logs) from all pods.

Usage
-----
    python remote/pull_outputs.py [pods.yml] [--dest outputs]

The script runs LOCALLY and copies *remote* `$REPO/outputs/*` into the same
relative path under *local* `dest/NAME/`, where NAME is the pod label from
`pods.yml`.

Only small artefacts (PNG, PDF, NPZ, JSON, CSV, TXT) are copied.  If you need to
sync large checkpoints add patterns in the `INCLUDE_EXT` list below.
"""
from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path
from typing import Dict, List
import yaml

INCLUDE_EXT = {".png", ".pdf", ".npz", ".json", ".csv", ".txt"}


class Pod:
    def __init__(self, name: str, cfg: Dict):
        self.name = name
        self.uri = cfg["uri"]  # e.g. root@1.2.3.4
        self.port = cfg.get("port", 22)
        self.key = cfg.get("key")

    def rsync_prefix(self) -> List[str]:
        parts = [
            "rsync",
            "-avz",  # archive, verbose, compress
            "-e",
            f"ssh -p {self.port} -o StrictHostKeyChecking=no" + (f" -i {Path(self.key).expanduser()}" if self.key else ""),
        ]
        return parts


def sync_pod(pod: Pod, remote_repo: str, dest_dir: Path):
    remote_outputs = f"{pod.uri}:{remote_repo}/training/outputs/"
    dest = dest_dir / pod.name
    dest.mkdir(parents=True, exist_ok=True)

    # Build include rules for the extensions and exclude everything else.
    filters = []
    for ext in INCLUDE_EXT:
        filters += ["--include", f"*{ext}"]
    filters += ["--exclude", "*"]  # exclude everything else

    cmd = pod.rsync_prefix() + filters + [remote_outputs, str(dest)]
    return subprocess.run(cmd, capture_output=True, text=True)


def main(inv_path: Path, dest: Path):
    with inv_path.open() as f:
        inv = yaml.safe_load(f)

    repo = inv["repo"].split("/")[-1].rstrip(".git")  # MemoryCapacitorTopologies
    pods = [Pod(name, cfg) for name, cfg in inv["pods"].items()]

    for pod in pods:
        print(f"\n⇣ Pulling outputs from {pod.name} ({pod.uri}) …")
        res = sync_pod(pod, repo, dest)
        if res.returncode == 0:
            print(f"[✔] {pod.name}: {res.stdout.strip() or 'no new files'}")
        else:
            print(f"[✘] {pod.name}: {res.stderr}", file=sys.stderr)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("inventory", nargs="?", default="remote/pods.yml", help="Path to pods.yml")
    ap.add_argument("--dest", default="outputs", help="Local destination root directory")
    args = ap.parse_args()
    main(Path(args.inventory).expanduser(), Path(args.dest).expanduser())
