import torch
import matplotlib.pyplot as plt
import numpy as np


def objective_function(x, y):
    return x**2 + y**2


def run_sgd(lr=0.1, momentum=0.0, weight_decay=0.0, maximize=False, start_point=(-2.8, 2.8), steps=50):
    """
    Runs SGD optimization and returns the trajectory of (x, y).
    """
    params = torch.tensor(start_point, dtype=torch.float32, requires_grad=True)

    optimizer = torch.optim.SGD([params], lr=lr, momentum=momentum, weight_decay=weight_decay, maximize=maximize)

    trajectory = [params.detach().clone().numpy()]

    for _ in range(steps):
        optimizer.zero_grad()
        if maximize:
            loss = -params[0]**2 - params[1]**2
        else:
            loss = objective_function(params[0], params[1])
        
        loss.backward()
        optimizer.step()
        trajectory.append(params.detach().clone().numpy())

    return np.array(trajectory)


def plot_trajectories(trajectories, title, func_to_plot='minimization', save_path='sgd_trajectories.png'):
    """
    Generates a contour plot and overlays the optimization trajectories.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    x_range = np.linspace(-3, 3, 100)
    y_range = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x_range, y_range)
    
    if func_to_plot == 'maximization':
        Z = -X**2 - Y**2
        func_label = r'$f(x, y) = -x^2 - y^2$'
    else:
        Z = X**2 + Y**2
        func_label = r'$f(x, y) = x^2 + y^2$'
        
    contours = ax.contour(X, Y, Z, levels=15, cmap='viridis')
    ax.clabel(contours, inline=True, fontsize=8)
    
    for label, data in trajectories.items():

        ax.plot(data[:, 0], data[:, 1], 'o-', label=label, markersize=4)

    for point in trajectories.values():
        start_point = point[0]
        ax.plot(start_point[0], start_point[1], 'go', markersize=10, label='Start')
    ax.plot(0, 0, 'r*', markersize=15, label='Optimum')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(title)
    ax.legend(loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_aspect('equal', adjustable='box')
    plt.show()
    plt.savefig(save_path)
    plt.close()

import random
if __name__ == "__main__":
    trajectories_momentum = {
        'Momentum = 0.0': run_sgd(momentum=0.0,start_point=(random.uniform(-3, 3), random.uniform(-3, 3))),
        'Momentum = 0.1': run_sgd(momentum=0.1,start_point=(random.uniform(-3, 3), random.uniform(-3, 3))),
        'Momentum = 0.5': run_sgd(momentum=0.5,start_point=(random.uniform(-3, 3), random.uniform(-3, 3))),
        'Momentum = 0.7': run_sgd(momentum=0.7,start_point=(random.uniform(-3, 3), random.uniform(-3, 3))),
        'Momentum = 0.9': run_sgd(momentum=0.9,start_point=(random.uniform(-3, 3), random.uniform(-3, 3))),
    }
    plot_trajectories(trajectories_momentum, 'SGD with Varying Momentum', save_path='sgd_trajectories_momentum.png')

    trajectories_weight_decay = {
        'Momentum = 0.0, WD = 0.1': run_sgd(momentum=0.0, weight_decay=0.1,start_point=(random.uniform(-3, 3), random.uniform(-3, 3))),
        'Momentum = 0.1, WD = 0.1': run_sgd(momentum=0.1, weight_decay=0.1,start_point=(random.uniform(-3, 3), random.uniform(-3, 3))),
        'Momentum = 0.5, WD = 0.1': run_sgd(momentum=0.5, weight_decay=0.1,start_point=(random.uniform(-3, 3), random.uniform(-3, 3))),
        'Momentum = 0.7, WD = 0.1': run_sgd(momentum=0.7, weight_decay=0.1,start_point=(random.uniform(-3, 3), random.uniform(-3, 3))),
        'Momentum = 0.9, WD = 0.1': run_sgd(momentum=0.9, weight_decay=0.1,start_point=(random.uniform(-3, 3), random.uniform(-3, 3))),
    }
    plot_trajectories(trajectories_weight_decay, 'SGD with Varying Momentum and Weight Decay (0.1)', save_path='sgd_trajectories_weight_decay.png')


    
    trajectories_maximize = {
        'Minimize: f(x,y) = x^2+y^2': run_sgd(momentum=0.9, maximize=False,start_point=(random.uniform(-3, 3), random.uniform(-3, 3))),
        'Maximize: f(x,y) = -x^2-y^2': run_sgd(momentum=0.9, maximize=True,start_point=(random.uniform(-3, 3), random.uniform(-3, 3)))
    }
    plot_trajectories(trajectories_maximize, 'SGD for Minimization vs. Maximization', func_to_plot='maximization', save_path='sgd_trajectories_maximize.png')