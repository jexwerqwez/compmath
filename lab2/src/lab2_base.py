import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['font.family'] = 'serif'
rcParams['font.serif'] = 'Times New Roman'
rcParams.update({'font.size': 16})

C, T = 1.03439984, 1.75418438
g = 9.81
epsilon = 1e-7
a, y_a = 2, 1


def brachistochrone_curve_x(t: float) -> float:
    return C * (t - 0.5 * np.sin(2 * t))


def brachistochrone_curve_y(t: float) -> float:
    return C * (0.5 - 0.5 * np.cos(2 * t))


def dx_dt(t: float):
    return C * (1 - np.cos(2 * t))


def dy_dt(t: float):
    return C * np.sin(2 * t)


def y_derivative(t):
    dx = dx_dt(t)
    dy = dy_dt(t)
    return dy / dx


def f(t):
    num = 1 + y_derivative(t) ** 2
    den = 2 * g * brachistochrone_curve_y(t)
    return np.sqrt(num / den) * dx_dt(t)


def f_exact(t):
    return np.sqrt(2 * C / g) * t


def composite_simpson(a, b, n, f):
    h = (b - a) / (n - 1)
    points = np.linspace(a, b, n)
    f_values = f(points)
    result = f_values[0] + f_values[-1]
    result += 2 * np.sum(f_values[2:-1:2])
    result += 4 * np.sum(f_values[1:-1:2])
    return result * h / 3


def composite_trapezoid(a, b, n, f):
    h = (b - a) / (n - 1)
    points = np.linspace(a, b, n)
    f_values = f(points)
    result = np.sum(f_values) - 0.5 * (f_values[0] + f_values[-1])
    return result * h


def print_graph(h_values_simpson, h_values_trapezoid, simpson_errors, trapezoid_errors, logspace_h_values, steps):
    plt.figure(figsize=(11, 9))
    plt.loglog(h_values_simpson, simpson_errors, marker='o', markersize=2, label=r'$E_s$')
    plt.loglog(h_values_trapezoid, trapezoid_errors, marker='x', markersize=2, label=r'$E_t$')

    oh_line2 = 1 / 2 * 1e-4 * logspace_h_values ** 2
    plt.loglog(logspace_h_values, oh_line2, '--', label=r'$O(h^2)$', color='black')
    oh_line4 = 1 / 2 * 1e-4 * logspace_h_values ** 4
    plt.loglog(logspace_h_values, oh_line4, '-.', label=r'$O(h^4)$', color='black')

    slope_trapezoid = calculate_slope(steps, trapezoid_errors)
    slope_simpson = calculate_slope(steps, simpson_errors)

    print(f'Порядок точности составной формулы Сипсона {slope_simpson}')
    print(f'Порядок точности составной формулы трапеций {slope_trapezoid}')
    plt.loglog(steps, 1e-4 * steps ** slope_trapezoid, '--', label=r'$O(h^{0.97}$)', color='darkred')
    plt.loglog(steps, 1e-4 * steps ** slope_simpson, '-.', label=r'$O(h^{0.9}$)', color='darkblue')

    machine_epsilon = np.finfo(float).eps
    plt.loglog(steps, machine_epsilon * np.ones_like(steps), label=r'$\epsilon$', color='magenta', linestyle='dotted')

    plt.xlabel(r'$h$')
    plt.ylabel(r'$E_s$, $E_h$')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()


def calculate_simpson_errors(exact_value):
    n_start, n_finish = 3, 9999
    if n_start % 2 == 0:
        n_start += 1
    n_finish += 1
    n_values = np.array([i for i in range(n_start, n_finish, 2)])

    simpson_errors = []
    for n in n_values:
        approximate_value_simpson = composite_simpson(epsilon, T, n, f)
        error_simpson = np.abs(exact_value - approximate_value_simpson)
        simpson_errors.append(error_simpson)
    h_values = (T - epsilon) / (n_values - 1)
    return simpson_errors, h_values


def calculate_trapezoid_errors(exact_value):
    n_start, n_finish = 3, 9999
    n_finish += 1
    n_values = np.array([i for i in range(n_start, n_finish, 1)])
    trapezoid_errors = []
    for n in n_values:
        approximate_value_trapezoid = composite_trapezoid(epsilon, T, n, f)
        error_trapezoid = np.abs(exact_value - approximate_value_trapezoid)
        trapezoid_errors.append(error_trapezoid)
    h_values = (T - epsilon) / (n_values - 1)
    return trapezoid_errors, h_values


def calculate_slope(x, y):
    log_x = np.log(x)
    log_y = np.log(y)
    return (log_y[-1] - log_y[0]) / (log_x[-1] - log_x[0])


def determine_steps(a, b, start_n, end_n):
    steps = []
    for n in range(start_n, end_n + 1):
        h = (b - a) / n
        steps.append(h)
    return steps


def find_min_step_for_accuracy(errors, h_values, accuracy_threshold):
    for error, h in zip(errors, h_values):
        if error < accuracy_threshold:
            return h
    return None


def optimization_base():
    exact_value = f_exact(T)
    simpson_errors, h_values_simpson = calculate_simpson_errors(exact_value)
    trapezoid_errors, h_values_trapezoid = calculate_trapezoid_errors(exact_value)
    steps = determine_steps(epsilon, T, 3, 9999)
    logspace_h_values = np.logspace(-2.5, np.log10(max(h_values_trapezoid)), num=len(h_values_trapezoid))
    print_graph(h_values_simpson, h_values_trapezoid, simpson_errors, trapezoid_errors, logspace_h_values, steps)


if __name__ == '__main__':
    optimization_base()
