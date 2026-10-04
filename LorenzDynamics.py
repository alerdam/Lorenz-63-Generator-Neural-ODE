"""
Parametric Lorenz-63 Dynamics Integrator & Data Generator
------------------------------------------------------------------------------------------------
Generates trajectories for the Lorenz-63 dynamical system on a 6D augmented manifold.
Implements classical RK4 integration with parameter sampling for physics-informed pipelines.
"""

import numpy as np

class ParametricLorenzDynamics:
    """Generates 6D phase-space trajectories augmented with physical parameters q = [x, y, z, sigma, rho, beta]."""
    def __init__(self):
        pass

    def _derivative(self, augmented_state: np.ndarray) -> np.ndarray:
        x, y, z, sigma, rho, beta = augmented_state
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z
        return np.array([dx, dy, dz, 0.0, 0.0, 0.0], dtype=np.float64)

    def generate_trajectory(self, init_state: list, init_params: list, t_max: float = 5.0, num_steps: int = 500) -> np.ndarray:
        dt = t_max / num_steps
        current_state = np.array(init_state + init_params, dtype=np.float64)
        trajectory = []

        for _ in range(num_steps):
            trajectory.append(current_state.copy())

            k1 = self._derivative(current_state)
            k2 = self._derivative(current_state + 0.5 * dt * k1)
            k3 = self._derivative(current_state + 0.5 * dt * k2)
            k4 = self._derivative(current_state + dt * k3)
            current_state = current_state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        return np.array(trajectory, dtype=np.float32)
