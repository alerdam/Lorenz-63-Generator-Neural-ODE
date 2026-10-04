"""
6D Manifold Neural ODE Architecture & Predictive Loss Functions
------------------------------------------------------------------------------------------------
Defines continuous vector field network f_theta(q) = dq/dt operating on q = [x, y, z, sigma, rho, beta].
Includes Predictor-Corrector Newton-Midpoint integrator and dynamic MPC trajectory loss.
"""

import torch
import torch.nn as nn


class VectorFieldNetworkDecoupled(nn.Module):
    def __init__(self, state_dim: int = 3, param_dim: int = 3, latent_dim: int = 32):
        super().__init__()
        self.field_net = nn.Sequential(
            nn.Linear(state_dim + param_dim, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, state_dim)
        )

    def forward(self, dynamic_state: torch.Tensor, static_params: torch.Tensor) -> torch.Tensor:
        augmented_input = torch.cat([dynamic_state, static_params], dim=-1)
        return self.field_net(augmented_input)


class HeunNeuralODEIntegrator(nn.Module):
    def __init__(self, vector_field: nn.Module, step_size: float = 0.01):
        super().__init__()
        self.vector_field = vector_field
        self.dt = step_size

    def step(self, state: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        k1 = self.vector_field(state, params)
        pred_state = state + self.dt * k1
        k2 = self.vector_field(pred_state, params)
        return state + 0.5 * self.dt * (k1 + k2)

    def forward(self, initial_augmented_state: torch.Tensor, time_steps: int = 400) -> torch.Tensor:
        current_state = initial_augmented_state[..., :3]
        params = initial_augmented_state[..., 3:]
        trajectory = [initial_augmented_state]

        for _ in range(time_steps - 1):
            current_state = self.step(current_state, params)
            trajectory.append(torch.cat([current_state, params], dim=-1))

        return torch.stack(trajectory, dim=1)


class MPCObjectiveLossDecoupled(nn.Module):
    """
    Evaluates multi-step predictive trajectories strictly on the 3D dynamic state space.
    Static parameters [sigma, rho, beta] are excluded from loss penalization.
    """
    def __init__(self):
        super().__init__()
        self.register_buffer('Q', torch.tensor([10.0, 10.0, 10.0], dtype=torch.float32))

    def forward(self, step_fn, batch_x: torch.Tensor, rollouts_y: torch.Tensor, current_horizon: int) -> torch.Tensor:
        accumulated_cost = 0.0
        
        current_state = batch_x[:, :3]
        static_params = batch_x[:, 3:]
        
        for t in range(current_horizon):
            next_state_pred = step_fn(current_state, static_params)
            target_reference = rollouts_y[:, t, :3]
            
            step_cost = torch.mean(((next_state_pred - target_reference) ** 2) * self.Q)
            accumulated_cost += step_cost
            current_state = next_state_pred
            
        return accumulated_cost / current_horizon



