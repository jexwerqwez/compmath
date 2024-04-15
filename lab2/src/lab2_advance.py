import numpy as np
import matplotlib.pyplot as plt
from lab2_base import composite_simpson, brachistochrone_curve_x, brachistochrone_curve_y, f_exact
from scipy.optimize import minimize
from matplotlib import rcParams
from scipy.spatial import distance

rcParams['font.family'] = 'serif'
rcParams['font.serif'] = 'Times New Roman'
rcParams.update({'font.size': 16})

g = 9.8
epsilon = 1e-7
C, T = 1.03439984, 1.75418438
a, y_a = epsilon, epsilon
b, y_b = 2, 1


def f_int(k, y):
    return np.sqrt((1 + k ** 2) / (2 * g * y))


def piecewise_linear_interpolation(x, x_i, x_next, y_i, y_next):
    if x_i == x_next:
        return 0, y_i
    k = (y_next - y_i) / (x_next - x_i)
    return k, k * (x - x_i) + y_i


def functional_linear(y_values, x_points, n_simpson):
    total_integral = 0
    for i in range(1, len(x_points)):
        x_i, x_prev = x_points[i], x_points[i-1]
        y_i, y_prev = y_values[i], y_values[i-1]

        f_segment = lambda x: f_int(*piecewise_linear_interpolation(x, x_i, x_prev, y_i, y_prev))
        total_integral += composite_simpson(x_prev, x_i, n_simpson, f_segment)
    return total_integral


def get_optim_nodes_linear(y_nodes, A, B, x_nodes, n_simpson):
    def constraints(y):
        return [y[0] - A, y[-1] - B]
    def objective(y):
        return functional_linear(y, x_nodes, n_simpson)
    cons = ({'type': 'eq', 'fun': constraints})
    result = minimize(objective, y_nodes, constraints=cons, method='SLSQP', options={'disp': True})
    return result


def l2_norm(y_optimized, y_analytical):
    return np.sqrt(np.sum((y_optimized - y_analytical) ** 2))


def error_optim(optim, N, C, T):
    t = np.linspace(0, T, N)
    x_c = brachistochrone_curve_x(t)
    y_c = brachistochrone_curve_y(t)
    return distance.euclidean(optim, y_c)


def interpolate_optimized_on_t(x_nodes, y_optimized, t):
    indices = np.searchsorted(x_nodes, t, side='right') - 1
    indices = np.clip(indices, 0, len(x_nodes) - 2)

    x_i = x_nodes[indices]
    x_next = x_nodes[indices + 1]
    y_i = y_optimized[indices]
    y_next = y_optimized[indices + 1]

    k = (y_next - y_i) / (x_next - x_i)
    y_interpolated = k * (t - x_i) + y_i
    return y_interpolated


def optimization_advance():
    exact_value = f_exact(T)

    N_range = np.logspace(0.5, 1, 30, dtype=int)
    n_range = np.logspace(0.5, 1, 30, dtype=int)

    interp_steps = (b - a) / (N_range - 1)
    simpson_steps = (b - a) / (n_range - 1)
    errors = np.zeros((len(N_range), len(n_range)))

    for i, N in enumerate(N_range):
        x_nodes = np.linspace(a, b, N)
        y_nodes = np.linspace(y_a, y_b, N)
        for j, n in enumerate(n_range):
            optim_result = get_optim_nodes_linear(y_nodes, y_a, y_b, x_nodes, n)
            num_integral = functional_linear(optim_result.x, x_nodes, n)
            error = l2_norm(num_integral, exact_value)
            errors[i, j] = error

    X, Y = np.meshgrid(simpson_steps, interp_steps)
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(np.log10(X), np.log10(Y), np.log10(errors), cmap='viridis')
    ax.set_xlabel(r'log10 $h_{integr}$')
    ax.set_ylabel(r'log10 $h_{interp}$')
    ax.set_zlabel(r'error')
    ax.view_init(elev=20, azim=-150)
    plt.savefig("diskret.pdf")
    plt.show()

    plt.figure(figsize=(12, 8))
    cp = plt.contour(np.log10(X), np.log10(Y), np.log10(errors), 20)
    plt.clabel(cp, inline=True, fontsize=8)
    plt.xlabel(r'log10 $h_{integr}$')
    plt.ylabel(r'log10 $h_{interp}$')
    plt.savefig("diskret_lines.pdf")
    plt.show()

if __name__ == '__main__':
    optimization_advance()
