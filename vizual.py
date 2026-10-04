import os
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp

from LorenzDynamics import ParametricLorenzDynamics
from modelArch import VectorFieldNetwork, HeunNeuralODEIntegrator

MODEL_WEIGHTS_PATH = "Lorenz_NeuralODE.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

t_max = 4.0
num_steps = 400
dt = t_max / num_steps
t_eval = np.linspace(0, t_max, num_steps)

vector_field = VectorFieldNetwork(state_dim=3, param_dim=3, latent_dim=32).to(device)

if os.path.exists(MODEL_WEIGHTS_PATH):
    vector_field.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=device))
    vector_field.eval()
    ode_integrator = HeunNeuralODEIntegrator(vector_field, step_size=dt).to(device)
    ode_integrator.eval()
else:
    print(f"Warning: Checkpoint '{MODEL_WEIGHTS_PATH}' not found. Using initialized weights.")
    ode_integrator = HeunNeuralODEIntegrator(vector_field, step_size=dt).to(device)

lorenz_system = ParametricLorenzDynamics()

def evaluate_benchmarks(init_x, init_y, init_z, sigma, rho, beta):
    def lorenz_ode(t, state):
        aug_state = np.array([state[0], state[1], state[2], sigma, rho, beta], dtype=np.float64)
        return lorenz_system._derivative(aug_state)[:3]

    init_state_3d = [init_x, init_y, init_z]
    
    t0_rk = time.perf_counter()
    sol = solve_ivp(lorenz_ode, (0, t_max), init_state_3d, t_eval=t_eval, method='RK45')
    time_rk45 = time.perf_counter() - t0_rk
    rk45_states = sol.y.T

    init_state_6d = [init_x, init_y, init_z, sigma, rho, beta]
    x0_tensor = torch.tensor(init_state_6d, dtype=torch.float32).unsqueeze(0).to(device)

    t0_nn = time.perf_counter()
    with torch.no_grad():
        pred_traj = ode_integrator(x0_tensor, time_steps=num_steps)
    time_nn = time.perf_counter() - t0_nn
    nn_states = pred_traj.squeeze(0).cpu().numpy()[:, :3]

    return rk45_states, nn_states, time_rk45, time_nn

fig = plt.figure(figsize=(9, 7), facecolor='white')
fig.canvas.manager.set_window_title("Decoupled Neural ODE Lorenz Dashboard")

ax_main = fig.add_subplot(2, 1, 1, projection='3d')
ax_rk45_static = fig.add_subplot(2, 2, 3, projection='3d')
ax_nn_static = fig.add_subplot(2, 2, 4, projection='3d')

plt.subplots_adjust(left=0.01, right=0.68, top=0.96, bottom=0.04, hspace=0.15, wspace=0.05)

line_rk45, = ax_main.plot([], [], [], color='#1f77b4', lw=1.2, label='RK45')
line_nn, = ax_main.plot([], [], [], color='#d62728', lw=1.2, linestyle='--', label='Neural ODE')
point_rk45, = ax_main.plot([], [], [], color='#1f77b4', marker='o', ms=4)
point_nn, = ax_main.plot([], [], [], color='#d62728', marker='o', ms=4)

title_text = ax_main.set_title("Simulation Dynamics", fontsize=8, pad=2)
ax_main.tick_params(labelsize=6)
ax_main.legend(loc="upper left", fontsize=7)

for ax, title, color in [(ax_rk45_static, "Static RK45", '#1f77b4'),
                         (ax_nn_static, "Static Neural ODE", '#d62728')]:
    ax.set_title(title, fontsize=7, color=color, pad=1)
    ax.tick_params(labelsize=5)

line_rk_stat, = ax_rk45_static.plot([], [], [], color='#1f77b4', lw=1.0)
line_nn_stat, = ax_nn_static.plot([], [], [], color='#d62728', lw=1.0, linestyle='--')

sl_x, sl_w, sl_h = 0.72, 0.22, 0.025
sliders_config = [
    (r'$x_0$', -15.0, 15.0, 1.0, 0.88),
    (r'$y_0$', -15.0, 15.0, 1.0, 0.82),
    (r'$z_0$', 10.0, 40.0, 20.0, 0.76),
    (r'$\sigma$', 5.0, 20.0, 10.0, 0.65),
    (r'$\rho$', 10.0, 50.0, 28.0, 0.59),
    (r'$\beta$', 1.0, 5.0, 8.0/3.0, 0.53)
]

sliders_objects = {}
for name, val_min, val_max, val_init, y_pos in sliders_config:
    ax_sl = plt.axes([sl_x, y_pos, sl_w, sl_h], facecolor='#f8f9fa')
    slider = Slider(ax_sl, name, val_min, val_max, valinit=val_init, valfmt='%1.2f', color='#1f77b4')
    slider.label.set_size(8)
    slider.valtext.set_size(7)
    sliders_objects[name] = slider

ax_btn_reset = plt.axes([0.72, 0.42, 0.22, 0.04])
btn_reset = Button(ax_btn_reset, 'Reset', color='#f0f0f0', hovercolor='#e0e0e0')
btn_reset.label.set_size(8)

rk45_full, nn_full = None, None
ani_step = 0

def update_data_and_plots(val=None):
    global rk45_full, nn_full, ani_step

    ix = sliders_objects[r'$x_0$'].val
    iy = sliders_objects[r'$y_0$'].val
    iz = sliders_objects[r'$z_0$'].val
    
    sigma = sliders_objects[r'$\sigma$'].val
    rho = sliders_objects[r'$\rho$'].val
    beta = sliders_objects[r'$\beta$'].val

    rk45_full, nn_full, t_rk, t_nn = evaluate_benchmarks(ix, iy, iz, sigma, rho, beta)
    ani_step = 0

    title_text.set_text(f"Live Simulation | RK45: {t_rk*1000:.1f}ms | Neural ODE: {t_nn*1000:.1f}ms")

    all_data = np.vstack((rk45_full, nn_full))
    min_b = all_data.min(axis=0) - 2
    max_b = all_data.max(axis=0) + 2

    ax_main.set_xlim(min_b[0], max_b[0])
    ax_main.set_ylim(min_b[1], max_b[1])
    ax_main.set_zlim(min_b[2], max_b[2])

    line_rk_stat.set_data(rk45_full[:, 0], rk45_full[:, 1])
    line_rk_stat.set_3d_properties(rk45_full[:, 2])
    ax_rk45_static.set_xlim(min_b[0], max_b[0])
    ax_rk45_static.set_ylim(min_b[1], max_b[1])
    ax_rk45_static.set_zlim(min_b[2], max_b[2])

    line_nn_stat.set_data(nn_full[:, 0], nn_full[:, 1])
    line_nn_stat.set_3d_properties(nn_full[:, 2])
    ax_nn_static.set_xlim(min_b[0], max_b[0])
    ax_nn_static.set_ylim(min_b[1], max_b[1])
    ax_nn_static.set_zlim(min_b[2], max_b[2])

    fig.canvas.draw_idle()

for s_obj in sliders_objects.values():
    s_obj.on_changed(update_data_and_plots)

def reset_simulation(event):
    for slider in sliders_objects.values():
        slider.reset()
    update_data_and_plots()

btn_reset.on_clicked(reset_simulation)

def update_animation_frame(frame):
    global ani_step, rk45_full, nn_full
    if rk45_full is None or nn_full is None:
        return line_rk45, line_nn, point_rk45, point_nn

    if ani_step < num_steps:
        ani_step += 2
        if ani_step > num_steps:
            ani_step = num_steps

        rk45_slice = rk45_full[:ani_step]
        nn_slice = nn_full[:ani_step]

        line_rk45.set_data(rk45_slice[:, 0], rk45_slice[:, 1])
        line_rk45.set_3d_properties(rk45_slice[:, 2])
        line_nn.set_data(nn_slice[:, 0], nn_slice[:, 1])
        line_nn.set_3d_properties(nn_slice[:, 2])

        point_rk45.set_data([rk45_slice[-1, 0]], [rk45_slice[-1, 1]])
        point_rk45.set_3d_properties([rk45_slice[-1, 2]])
        point_nn.set_data([nn_slice[-1, 0]], [nn_slice[-1, 1]])
        point_nn.set_3d_properties([nn_slice[-1, 2]])
    else:
        ani_step = 0

    return line_rk45, line_nn, point_rk45, point_nn

global_anim = FuncAnimation(fig, update_animation_frame, interval=20, blit=False, cache_frame_data=False)

update_data_and_plots()
plt.show()