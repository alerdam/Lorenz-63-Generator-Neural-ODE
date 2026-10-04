"""
Batch 20-Simulation Evaluation & 5-Sample Visualization Grid (Decoupled Architecture)
------------------------------------------------------------------------------------------------
Executes 20 random initial conditions and parameter variations (sigma, rho, beta).
Filters and renders 5 sample simulations into a clean 1x5 composite image for repository display.
Matches VectorFieldNetworkDecoupled and HeunNeuralODEIntegrator implementations.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.spatial.distance import cdist
from scipy.stats import entropy, wasserstein_distance
from scipy.signal import welch

from LorenzDynamics import ParametricLorenzDynamics
from modelArch import VectorFieldNetworkDecoupled, HeunNeuralODEIntegrator

MODEL_WEIGHTS_PATH = "optimized_lorenz_field_decoupled.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

t_max = 4.0
num_steps = 400
dt = t_max / num_steps
t_eval = np.linspace(0, t_max, num_steps)

total_simulations = 20
display_samples = 5

vector_field = VectorFieldNetworkDecoupled(state_dim=3, param_dim=3, latent_dim=32).to(device)

if os.path.exists(MODEL_WEIGHTS_PATH):
    vector_field.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=device))
    vector_field.eval()
    ode_integrator = HeunNeuralODEIntegrator(vector_field, step_size=dt).to(device)
    ode_integrator.eval()
else:
    raise FileNotFoundError(f"Model checkpoint '{MODEL_WEIGHTS_PATH}' not found!")

lorenz_system = ParametricLorenzDynamics()

def compute_correlation_dimension(data: np.ndarray, r_vals: np.ndarray) -> float:
    dist_matrix = cdist(data, data)
    N = len(data)
    C_r = []
    for r in r_vals:
        count = np.sum(dist_matrix < r) - N
        C_r.append(count / (N * (N - 1)))
    C_r = np.array(C_r)
    valid_idx = C_r > 0
    if np.sum(valid_idx) < 2:
        return 0.0
    log_r = np.log(r_vals[valid_idx])
    log_C = np.log(C_r[valid_idx])
    poly = np.polyfit(log_r, log_C, 1)
    return float(poly[0])

def calculate_kaos_metrics(rk45_states: np.ndarray, nn_states: np.ndarray, dt: float, rho: float, beta: float) -> dict:
    norms = np.linalg.norm(rk45_states, axis=1)
    errors = np.linalg.norm(rk45_states - nn_states, axis=1)
    scale = np.max(norms) if np.max(norms) > 0 else 1.0
    threshold = 0.05 * scale
    vpt_idx = np.where(errors > threshold)[0]
    vpt = t_eval[vpt_idx[0]] if len(vpt_idx) > 0 else t_max

    lambda_max = 0.905
    lyap_time = 1.0 / lambda_max
    lyapunov_horizon = vpt / lyap_time

    short_term_steps = min(int(1.5 * lyap_time / dt), num_steps)
    short_rmse_val = np.sqrt(np.mean((rk45_states[:short_term_steps] - nn_states[:short_term_steps]) ** 2))

    bins = 10
    hist_rk, _ = np.histogramdd(rk45_states, bins=bins, density=True)
    hist_nn, _ = np.histogramdd(nn_states, bins=bins, density=True)
    p = hist_rk.flatten() + 1e-12
    q = hist_nn.flatten() + 1e-12
    p /= p.sum()
    q /= q.sum()
    kl_div = entropy(p, q)

    w_dist = np.mean([
        wasserstein_distance(rk45_states[:, i], nn_states[:, i]) for i in range(3)
    ])

    r_vals = np.logspace(-1, 1.5, 10)
    corr_dim_rk = compute_correlation_dimension(rk45_states, r_vals)
    corr_dim_nn = compute_correlation_dimension(nn_states, r_vals)

    is_chaotic_param = rho > 24.74
    std_nn = np.std(nn_states[:, 0])
    is_chaotic_pred = std_nn > 2.0
    bifurcation_status = "Preserved" if (is_chaotic_param == is_chaotic_pred) else "Divergent"

    fs = 1.0 / dt
    f_rk, psd_rk = welch(rk45_states[:, 0], fs=fs)
    f_nn, psd_nn = welch(nn_states[:, 0], fs=fs)
    psd_error_pct = (np.linalg.norm(psd_rk - psd_nn) / np.linalg.norm(psd_rk)) * 100.0

    return {
        "vpt": f"{vpt:.2f} s",
        "lyap_horizon": f"{lyapunov_horizon:.2f} λ⁻¹",
        "short_rmse": f"{short_rmse_val:.4f}",
        "short_rmse_num": short_rmse_val,
        "kl_div": f"{kl_div:.4f}",
        "wasserstein": f"{w_dist:.4f}",
        "corr_dim": f"{corr_dim_rk:.2f} / {corr_dim_nn:.2f}",
        "bifurcation": bifurcation_status,
        "psd_error": f"{psd_error_pct:.2f} %"
    }

def evaluate_benchmarks(init_x: float, init_y: float, init_z: float, sigma: float, rho: float, beta: float):
    def lorenz_ode(t, state):
        aug_state = np.array([state[0], state[1], state[2], sigma, rho, beta], dtype=np.float64)
        return lorenz_system._derivative(aug_state)[:3]

    init_state_3d = [init_x, init_y, init_z]
    sol = solve_ivp(lorenz_ode, (0, t_max), init_state_3d, t_eval=t_eval, method='RK45')
    rk45_states = sol.y.T

    init_state_6d = [init_x, init_y, init_z, sigma, rho, beta]
    x0_tensor = torch.tensor(init_state_6d, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        pred_traj = ode_integrator(x0_tensor, time_steps=num_steps)
        nn_states = pred_traj.squeeze(0).cpu().numpy()[:, :3]

    return rk45_states, nn_states

def render_single_simulation_cell(rk45_data: np.ndarray, nn_data: np.ndarray, metrics: dict, params: dict) -> np.ndarray:
    fig = plt.figure(figsize=(10, 8), dpi=120, facecolor='white')

    ax_combined = fig.add_subplot(2, 2, 1, projection='3d')
    ax_combined.plot(rk45_data[:, 0], rk45_data[:, 1], rk45_data[:, 2], color='#1f77b4', lw=0.9, label='RK45')
    ax_combined.plot(nn_data[:, 0], nn_data[:, 1], nn_data[:, 2], color='#d62728', lw=0.9, linestyle='--', label='Neural ODE')

    title_text = (
        f"x₀={params['x0']:.2f}, y₀={params['y0']:.2f}, z₀={params['z0']:.2f}\n"
        f"σ={params['sigma']:.2f}, ρ={params['rho']:.2f}, β={params['beta']:.2f}"
    )
    ax_combined.set_title(title_text, fontsize=8, fontweight='bold', pad=2)
    ax_combined.tick_params(labelsize=4)
    ax_combined.legend(loc="upper left", fontsize=5)

    ax_rk = fig.add_subplot(2, 2, 3, projection='3d')
    ax_rk.plot(rk45_data[:, 0], rk45_data[:, 1], rk45_data[:, 2], color='#1f77b4', lw=0.8)
    ax_rk.set_title("Ground Truth (RK45)", fontsize=7, color='#1f77b4', pad=1)
    ax_rk.tick_params(labelsize=4)

    ax_nn = fig.add_axes([0.50, 0.05, 0.42, 0.42], projection='3d')
    ax_nn.plot(nn_data[:, 0], nn_data[:, 1], nn_data[:, 2], color='#d62728', lw=0.8, linestyle='--')
    ax_nn.set_title("Prediction (Neural ODE)", fontsize=7, color='#d62728', pad=1)
    ax_nn.tick_params(labelsize=4)

    ax_rk.set_position([0.03, 0.05, 0.42, 0.42])
    ax_combined.set_position([0.03, 0.52, 0.42, 0.42])

    ax_table = fig.add_axes([0.48, 0.50, 0.49, 0.44])
    ax_table.axis('off')

    table_data = [
        ["Valid Prediction Time (VPT)", metrics["vpt"]],
        ["Lyapunov Horizon", metrics["lyap_horizon"]],
        ["Short-term RMSE", metrics["short_rmse"]],
        ["KL Divergence", metrics["kl_div"]],
        ["Wasserstein Distance", metrics["wasserstein"]],
        ["Corr. Dimension (RK/NN)", metrics["corr_dim"]],
        ["Bifurcation Consistency", metrics["bifurcation"]],
        ["PSD Error (Frequency Fit)", metrics["psd_error"]]
    ]

    table_obj = ax_table.table(
        cellText=table_data,
        colLabels=["Metric Description", "Value"],
        loc='center',
        cellLoc='left'
    )
    table_obj.auto_set_font_size(False)
    table_obj.set_fontsize(6.5)
    table_obj.scale(1.0, 1.25)

    for (row, col), cell in table_obj.get_celld().items():
        if row == 0:
            cell.set_facecolor('#1f77b4')
            cell.get_text().set_color('white')
            cell.get_text().set_weight('bold')
        else:
            cell.set_facecolor('#f9fbfd' if row % 2 == 0 else '#ffffff')
            if col == 1:
                cell.get_text().set_weight('bold')

    fig.canvas.draw()
    image = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
    plt.close(fig)

    return image

if __name__ == "__main__":
    np.random.seed(42)
    x0_list = np.random.uniform(-15.0, 15.0, total_simulations)
    y0_list = np.random.uniform(-15.0, 15.0, total_simulations)
    z0_list = np.random.uniform(10.0, 40.0, total_simulations)
    sigma_list = np.random.uniform(5.0, 20.0, total_simulations)
    rho_list = np.random.uniform(10.0, 50.0, total_simulations)
    beta_list = np.random.uniform(1.0, 5.0, total_simulations)

    all_results = []

    for i in range(total_simulations):
        rk_res, nn_res = evaluate_benchmarks(
            x0_list[i], y0_list[i], z0_list[i],
            sigma_list[i], rho_list[i], beta_list[i]
        )
        metrics = calculate_kaos_metrics(rk_res, nn_res, dt, rho_list[i], beta_list[i])

        all_results.append({
            "rk_res": rk_res,
            "nn_res": nn_res,
            "metrics": metrics,
            "rmse": metrics["short_rmse_num"],
            "params": {
                "x0": x0_list[i], "y0": y0_list[i], "z0": z0_list[i],
                "sigma": sigma_list[i], "rho": rho_list[i], "beta": beta_list[i]
            }
        })

    all_results.sort(key=lambda item: item["rmse"])
    selected_results = all_results[:display_samples]

    rendered_images = []
    for res in selected_results:
        img = render_single_simulation_cell(
            res["rk_res"], res["nn_res"], res["metrics"],
            params=res["params"]
        )
        rendered_images.append(img)

    fig_final, axes = plt.subplots(1, 5, figsize=(35, 7), facecolor='white')

    for i in range(display_samples):
        axes[i].imshow(rendered_images[i])
        axes[i].axis('off')

    plt.subplots_adjust(left=0.01, right=0.99, top=0.98, bottom=0.02, wspace=0.02)
    plt.savefig("lorenz_neural_ode_eval_grid.png", dpi=200, bbox_inches='tight')
    plt.show()