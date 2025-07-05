import torch
import torch.nn as nn

class Memcapacitor(nn.Module):
    """
    A simple voltage-controlled memcapacitor model.

    The capacitance is defined as C(t) = C0 + k * phi(t), where phi is the time integral of voltage (flux).
    The charge is q(t) = C(t) * v(t).

    Args:
        c0 (float): Initial capacitance.
        k (float): A constant that determines how strongly the capacitance changes with flux.
        dt (float): The time step for simulation.
    """
    def __init__(self, c0, k, dt):
        super(Memcapacitor, self).__init__()
        self.c0 = c0
        self.k = k
        self.dt = dt
        self.flux = nn.Parameter(torch.tensor(0.0), requires_grad=False)

    def forward(self, v):
        """
        Performs a forward pass of the memcapacitor model.

        Args:
            v (torch.Tensor): The input voltage at the current time step.

        Returns:
            torch.Tensor: The charge on the memcapacitor.
        """
        self.flux.data += v * self.dt
        capacitance = self.c0 + self.k * self.flux
        q = capacitance * v
        return q

    def reset(self):
        """
        Resets the internal state (flux) of the memcapacitor.
        """
        self.flux.data.fill_(0.0)
