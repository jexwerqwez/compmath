import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.optimize import root

rcParams['font.family'] = 'serif'
rcParams['font.serif'] = 'Times New Roman'
rcParams.update({'font.size': 16})

initial_omega_values = np.linspace(1.85, 2.1, 15)
h = 0.1
t_n = 100


def f(t, w):
    theta, omega = w
    return np.array([omega, np.cos(t) - 0.1 * omega - np.sin(theta)])


def runge_kutta(x_0, t_n, f, h):
    n = int(t_n / h)
    w = np.zeros((n + 1, len(x_0)))
    w[0] = x_0
    t = np.linspace(0, t_n, n + 1)
    for i in range(n):
        k1 = h * f(t[i], w[i])
        k2 = h * f(t[i] + h / 2, w[i] + k1 / 2)
        k3 = h * f(t[i] + h / 2, w[i] + k2 / 2)
        k4 = h * f(t[i] + h, w[i] + k3)
        w[i + 1] = w[i] + (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return w


def milne_simpson(x_0, t_n, f, h):
    w_rk = runge_kutta(x_0, 4 * h, f, h)
    w = np.zeros((int(t_n / h) + 1, len(x_0)))
    w[:4] = w_rk[:4]
    t = np.linspace(0, t_n, int(t_n / h) + 1)
    for i in range(3, int(t_n / h)):
        wp = w[i - 3] + (4 * h / 3) * (2 * f(t[i], w[i]) - f(t[i - 1], w[i - 1]) + 2 * f(t[i - 2], w[i - 2]))
        w[i + 1] = w[i - 1] + (h / 3) * (f(t[i + 1], wp) + 4 * f(t[i], w[i]) + f(t[i - 1], w[i - 1]))
    return w


def adams_moulton(x_0, t_n, f, h):
    w_rk = runge_kutta(x_0, 3 * h, f, h)
    w = np.zeros((int(t_n / h) + 1, len(x_0)))
    w[:4] = w_rk[:4]
    for i in range(3, int(t_n / h)):
        t = (i + 1) * h
        def g(y):
            y = np.array(y)
            return y - w[i] - h * ((9 / 24) * f(t, y) +
                                   (19 / 24) * f(t - h, w[i]) -
                                   (5 / 24) * f(t - 2 * h, w[i - 1]) +
                                   (1 / 24) * f(t - 3 * h, w[i - 2]))
        sol = root(g, w[i])
        w[i + 1] = sol.x
    return w


def plot_graphs(filename):
    plt.figure(figsize=(15, 10))
    for omega in initial_omega_values:
        x_0 = [0, omega]
        if filename == 'rk':
            trajectory = runge_kutta(x_0, t_n, f, h)
        elif filename == 'ms':
            trajectory = milne_simpson(x_0, t_n, f, h)
        else:
            trajectory = adams_moulton(x_0, t_n, lambda t, y: f(t, y), h)
        plt.plot(np.arange(0, t_n + h, h), trajectory[:, 0], label=fr'$w_0={x_0[1]:.2f}$')
    plt.xlabel('$t$')
    plt.ylabel(r'$\theta(t)$')
    plt.grid(True)
    plt.legend(loc='lower right')
    save_filename = filename + '.pdf'
    plt.savefig(save_filename)
    plt.show()


def optimal_step(steps, filename):
    initial_omega = 2.06
    rows = len(steps) // 2 + len(steps) % 2
    cols = 2

    plt.figure(figsize=(8 * cols, 4 * rows))

    for i, h in enumerate(steps):
        plt.subplot(rows, cols, i + 1)
        x_0 = [0, initial_omega]
        if filename == 'rk':
            trajectory = runge_kutta(x_0, t_n, f, h)
        elif filename == 'ms':
            trajectory = milne_simpson(x_0, t_n, f, h)
        else:
            trajectory = adams_moulton(x_0, t_n, lambda t, y: f(t, y), h)
        t_values = np.linspace(0, t_n, len(trajectory))
        plt.plot(t_values, trajectory[:, 0], label=f'h={h:.3f}')
        plt.xlabel('$t$')
        plt.ylabel(r'$\theta(t)$')
        plt.grid(True)
        plt.legend(loc='lower right')

    plt.tight_layout()
    save_filename = filename + '_optimal.pdf'
    plt.savefig(save_filename)
    plt.show()
    plt.close()


def lab3_base():
    plot_graphs('rk')
    steps = np.arange(0.048, 0.058, 0.002)
    optimal_step(steps, 'rk')

    plot_graphs('am')
    steps = np.arange(0.267, 0.272, 0.001)
    optimal_step(steps, 'am')

    plot_graphs('ms')
    steps = np.arange(0.048, 0.058, 0.002)
    optimal_step(steps, 'ms')


if __name__ == '__main__':
    lab3_base()