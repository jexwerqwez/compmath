import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import time
from lab3_base import runge_kutta, milne_simpson, adams_moulton

rcParams['font.family'] = 'serif'
rcParams['font.serif'] = 'Times New Roman'
rcParams.update({'font.size': 16})

initial_omega_values = np.linspace(1.85, 2.1, 15)
h = 0.1
t_n = 100


def phase_graphs(filename):
    plt.figure(figsize=(15, 10))
    for omega in initial_omega_values:
        x_0 = [0, omega]
        if filename == 'rk':
            trajectory = runge_kutta(x_0, t_n, f, h)
        elif filename == 'ms':
            trajectory = milne_simpson(x_0, t_n, f, h)
        else:
            trajectory = adams_moulton(x_0, t_n, lambda t, y: f(t, y), h)
        plt.plot(trajectory[:, 0], trajectory[:, 1], label=fr'$w_0={omega:.2f}$')

    plt.xlabel(r'$\theta$')
    plt.ylabel(r'$\frac{d\theta}{dt}$')
    plt.grid(True)
    plt.legend(loc='lower right')
    save_filename = filename + '_phase.pdf'
    plt.savefig(save_filename)
    plt.show()


def fixed_phase(omega_fixed, filename):
    x_0 = [0, omega_fixed]
    trajectory_rk = runge_kutta(x_0, t_n, f, h)
    trajectory_ms = milne_simpson(x_0, t_n, f, h)
    trajectory_am = adams_moulton(x_0, t_n, lambda t, y: f(t, y), h)
    plt.figure(figsize=(15, 10))
    plt.plot(trajectory_rk[:, 0], trajectory_rk[:, 1], label='Метод Рунге-Кутты', color='blue')
    plt.plot(trajectory_ms[:, 0], trajectory_ms[:, 1], label='Метод Милна-Симпсона', color='red')
    plt.plot(trajectory_am[:, 0], trajectory_am[:, 1], label='Метод Адамса-Моултона', color='green')

    plt.xlabel(r'$\theta$')
    plt.ylabel(r'$\frac{d\theta}{dt}$')
    plt.grid(True)
    plt.legend(loc='lower right')
    plt.savefig(filename)
    plt.show()


def measure_time(method, x_0, t_n, f, h):
    start_time = time.time()
    method(x_0, t_n, f, h)
    return time.time() - start_time


def plot_time():
    x_0 = [0, 2.06]
    h_values = np.linspace(0.01, 0.2, 20)
    times_rk = [measure_time(runge_kutta, x_0, t_n, f, h) for h in h_values]
    times_ms = [measure_time(milne_simpson, x_0, t_n, f, h) for h in h_values]
    times_am = [measure_time(adams_moulton, x_0, t_n, lambda t, y: f(t, y), h) for h in h_values]

    plt.figure(figsize=(12, 6))
    plt.plot(h_values, times_rk, label='Метод Рунге-Кутты')
    plt.plot(h_values, times_ms, label='Метод Милна-Симпсона')
    plt.plot(h_values, times_am, label='Метод Адамса-Моултона')

    plt.xlabel(r'$h$')
    plt.ylabel(r'$\tau$')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig('time_compare.pdf')
    plt.show()


def lab3_advance():
    phase_graphs('rk')
    phase_graphs('am')
    phase_graphs('ms')

    fixed_phase(2.1, "phase_comparison.pdf")
    fixed_phase(1.85, "phase_comparison_eq.pdf")

    plot_time()

if __name__ == '__main__':
    lab3_advance()