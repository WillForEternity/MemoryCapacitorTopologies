"""Verbose, self-explaining tests for the Memcapacitor model.

Each scenario prints detailed natural-language commentary describing what is
being simulated, the analytically expected outcome, the numerically obtained
outcome from the model, and whether the behaviour matches expectations within a
specified tolerance.  Nothing is asserted; instead the script is readable by a
human or downstream agent to judge correctness based purely on its stdout.
"""

from __future__ import annotations

import os
import math
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.memcapacitor import Memcapacitor

# Numerical tolerance for floating-point comparisons
TOL = 1e-4


def banner(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def describe_result(label: str, expected: float, actual: float):
    err = abs(expected - actual)
    status = "PASS" if err < TOL else "FAIL"
    print(f"{label}: expected {expected:.6f}, got {actual:.6f}  ->  {status} (|error|={err:.2e})")


def scenario_pure_sine():
    """Sinusoidal voltage with an integer number of cycles – net flux should be ≈0."""
    banner("Scenario 1: Pure sine wave – zero net flux expected")

    # Parameters
    c0, k, dt, f, amp, cycles = 1.0, 0.5, 1e-3, 2.0, 1.0, 10  # 10 full cycles
    memc = Memcapacitor(c0, k, dt)
    memc.reset()

    t = np.arange(0, cycles / f, dt)
    v = amp * np.sin(2 * math.pi * f * t)

    # Run simulation
    for vv in torch.from_numpy(v).float():
        _ = memc(vv)

    expected_flux = 0.0  # integral of complete sine cycles = 0
    describe_result("Final flux", expected_flux, memc.flux.item())

    expected_cap = c0 + k * expected_flux
    describe_result("Final capacitance", expected_cap, (c0 + k * memc.flux).item())
    print("Interpretation: Because positive and negative halves cancel exactly, the device behaves as an ordinary linear capacitor with unchanged capacitance.\n")


def scenario_dc_bias():
    """Sine wave with DC offset – flux accumulates linearly with the bias voltage."""
    banner("Scenario 2: Sine wave + DC bias – positive flux accumulation expected")

    c0, k, dt, f, amp, bias, cycles = 1.0, 0.5, 1e-3, 2.0, 1.0, 0.3, 4
    memc = Memcapacitor(c0, k, dt)
    memc.reset()

    t = np.arange(0, cycles / f, dt)
    v = bias + amp * np.sin(2 * math.pi * f * t)

    # Analytic expectations
    total_time = t[-1] + dt  # include last step duration
    expected_flux = bias * total_time  # sine integrates to 0 over integer cycles
    expected_cap = c0 + k * expected_flux

    # Run simulation
    for vv in torch.from_numpy(v).float():
        _ = memc(vv)

    describe_result("Final flux", expected_flux, memc.flux.item())
    describe_result("Final capacitance", expected_cap, (c0 + k * memc.flux).item())
    print("Interpretation: The constant bias drives a monotonic increase in flux, thus capacitance grows by k·flux. The sinusoidal component merely oscillates around this ramp.\n")


def scenario_constant_dc():
    """Constant DC voltage – flux should grow linearly with time."""
    banner("Scenario 3: Constant DC – linear flux growth expected")

    c0, k, dt, Vdc, steps = 1.0, 0.5, 1e-3, 0.7, 5000
    memc = Memcapacitor(c0, k, dt)
    memc.reset()

    expected_flux = Vdc * dt * steps
    expected_cap = c0 + k * expected_flux

    for _ in range(steps):
        _ = memc(torch.tensor(Vdc))

    describe_result("Final flux", expected_flux, memc.flux.item())
    describe_result("Final capacitance", expected_cap, (c0 + k * memc.flux).item())
    print("Interpretation: With a fixed voltage, flux—and hence capacitance—should increase linearly with elapsed simulation time.\n")


def main():
    print("Verbose behavioural tests for Memcapacitor model (no assertions, explanatory output only).\n")
    scenario_pure_sine()
    scenario_dc_bias()
    scenario_constant_dc()

    print("All scenarios executed. Review PASS/FAIL tags to assess correctness under each stimulus.")

    # --- Comprehensive visual summary ---
    print("\nGenerating composite visual summary image...")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Scenario 1 data
    c0, k, dt, f, amp, cycles = 1.0, 0.5, 1e-3, 2.0, 1.0, 10
    t = np.arange(0, cycles / f, dt)
    v = amp * np.sin(2 * math.pi * f * t)
    memc = Memcapacitor(c0, k, dt)
    memc.reset()
    q = [memc(torch.tensor(vv, dtype=torch.float32)).item() for vv in v]
    axes[0].plot(v, q)
    axes[0].set_title("Scenario 1: Pure sine")
    axes[0].set_xlabel("Voltage (V)")
    axes[0].set_ylabel("Charge (C)")
    axes[0].grid(True)

    # Scenario 2 data
    c0, k, dt, f, amp, bias, cycles = 1.0, 0.5, 1e-3, 2.0, 1.0, 0.3, 4
    t = np.arange(0, cycles / f, dt)
    v = bias + amp * np.sin(2 * math.pi * f * t)
    memc = Memcapacitor(c0, k, dt)
    memc.reset()
    q = [memc(torch.tensor(vv, dtype=torch.float32)).item() for vv in v]
    axes[1].plot(v, q)
    axes[1].set_title("Scenario 2: DC bias + sine")
    axes[1].set_xlabel("Voltage (V)")
    axes[1].grid(True)

    # Scenario 3 data
    c0, k, dt, Vdc, steps = 1.0, 0.5, 1e-3, 0.7, 5000
    v = np.full(steps, Vdc)
    memc = Memcapacitor(c0, k, dt)
    memc.reset()
    q = [memc(torch.tensor(Vdc)).item() for _ in range(steps)]
    axes[2].plot(range(steps), q)
    axes[2].set_title("Scenario 3: Constant DC (q vs step)")
    axes[2].set_xlabel("Time step")
    axes[2].grid(True)

    plt.tight_layout()
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    summary_path = os.path.join(output_dir, "memcapacitor_behavior_summary.png")
    plt.savefig(summary_path, dpi=150)
    print(f"Composite image saved to {summary_path}\n")

    print("Visual summary generated. Script complete.")


if __name__ == "__main__":
    main()
