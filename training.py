"""
Optimized Physics-Informed Neural ODE Training Pipeline (Decoupled Parameters)
------------------------------------------------------------------------------------------------
Implements a continuous vector field f_theta(x, p) = dx/dt of the Lorenz-63 system.
Dynamic states x=[x,y,z] and static parameters p=[sigma,rho,beta] are decoupled.
Includes RK2 integration, Dynamic MPC Horizon, and Gradient Clipping.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

from LorenzDynamics import ParametricLorenzDynamics
from modelArch import VectorFieldNetworkDecoupled, HeunNeuralODEIntegrator, MPCObjectiveLossDecoupled



def execute_neural_ode_pipeline(weights_path: str = "Lorenz_NeuralODE.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"| Execution Engine: Initialized on device [{device}]")

    lorenz_system = ParametricLorenzDynamics()
    integration_dt = 0.01
    max_mpc_horizon = 5

    # Initialized with the optimized 32-neuron latent dimension
    vector_field = VectorFieldNetworkDecoupled(state_dim=3, param_dim=3, latent_dim=32).to(device)
    ode_integrator = HeunNeuralODEIntegrator(vector_field, step_size=integration_dt).to(device)

    optimizer = optim.AdamW(ode_integrator.parameters(), lr=2e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    mpc_loss_fn = MPCObjectiveLossDecoupled().to(device)

    print("| Generating Parametric 6D Physical Trajectories...")
    num_trajectories = 5000
    steps_per_trajectory = 500
    t_max = 5.0

    input_states = []
    target_rollouts = [[] for _ in range(max_mpc_horizon)]

    for _ in range(num_trajectories):
        random_init_state = [
            np.random.uniform(-15, 15), 
            np.random.uniform(-15, 15), 
            np.random.uniform(10, 40)
        ]
        random_init_params = [
            np.random.uniform(5.0, 20.0),
            np.random.uniform(10.0, 50.0),
            np.random.uniform(1.0, 5.0)
        ]

        trajectory_matrix = lorenz_system.generate_trajectory(
            random_init_state, random_init_params, t_max=t_max, num_steps=steps_per_trajectory
        )

        for i in range(len(trajectory_matrix) - max_mpc_horizon):
            input_states.append(trajectory_matrix[i])
            for h in range(max_mpc_horizon):
                target_rollouts[h].append(trajectory_matrix[i + 1 + h])

    tensor_input_states = torch.tensor(np.array(input_states), dtype=torch.float32)
    tensor_target_rollouts = torch.tensor(np.array(target_rollouts), dtype=torch.float32).transpose(0, 1)

    dataset = TensorDataset(tensor_input_states, tensor_target_rollouts)
    data_loader = DataLoader(dataset, batch_size=256, shuffle=True)

    epochs = 6
    ode_integrator.train()

    print("| Beginning Continuum Mechanics Training Pipeline (Decoupled Parameters)...")
    for epoch in range(1, epochs + 1):
        current_horizon = min(max_mpc_horizon, 1 + (epoch // 2))
        accumulated_loss = 0.0
        
        for batch_states, batch_targets in data_loader:
            batch_states = batch_states.to(device)
            batch_targets = batch_targets.to(device)
            
            optimizer.zero_grad()

            loss = mpc_loss_fn(ode_integrator.step, batch_states, batch_targets, current_horizon)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(ode_integrator.parameters(), max_norm=1.0)
            
            optimizer.step()
            accumulated_loss += loss.item() * batch_states.size(0)

        mean_epoch_loss = accumulated_loss / tensor_input_states.shape[0]
        scheduler.step(mean_epoch_loss)

        if epoch % 2 == 0 or epoch == 1:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"  > Epoch {epoch:02d}/{epochs:02d} | Horizon: {current_horizon} | Loss: {mean_epoch_loss:.6f} | LR: {current_lr:.2e}")

    torch.save(vector_field.state_dict(), weights_path)
    print(f"| Model Checkpoint Saved: '{weights_path}'")

if __name__ == "__main__":
    execute_neural_ode_pipeline()