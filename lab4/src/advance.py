import numpy as np
from base import gauss, calculate_error, generate_matrix, thomas
import matplotlib.pyplot as plt


num_matrices = 1000
b = np.array([1, 1, 1, 1, 1, 1], dtype=np.float32)

def generate_symmetric_matrix(size):
    random_matrix = np.random.rand(size, size).astype(np.float32) * 2 - 1
    return (random_matrix + random_matrix.T) / 2

def is_positive_definite(matrix):
    return np.all(np.linalg.eigvals(matrix) > 0)

def generate_positive_definite_matrix(dimension=6):
    while True:
        symmetric_matrix = generate_symmetric_matrix(dimension)
        pd_matrix = symmetric_matrix + np.eye(dimension) * dimension
        if is_positive_definite(pd_matrix) and np.abs(np.linalg.det(pd_matrix)) > 0.01:
            return pd_matrix

def cholesky_decomposition(A):
    n = A.shape[0]
    L = np.zeros_like(A, dtype=np.float32)
    for i in range(n):
        for j in range(i+1):
            temp_sum = np.dot(L[i, :j], L[j, :j])
            if i == j:
                L[i, j] = np.sqrt(A[i, i] - temp_sum)
            else:
                L[i, j] = (A[i, j] - temp_sum) / L[j, j]
    return L

def cholesky(A, b):
    L = cholesky_decomposition(A)
    y = np.linalg.solve(L, b)
    x = np.linalg.solve(L.T, y)
    return x

def run_experiment_positive_definite(num_matrices, b):
    sq_errors = []
    sup_errors = []

    for _ in range(num_matrices):
        A = generate_positive_definite_matrix(6)
        print(A)
        x_universal = gauss(A, b, True)
        x_cholesky = cholesky(A, b)
        print(f"x_universal={x_universal}, x_cholesky={x_cholesky}")
        sq_error, sup_error = calculate_error(x_universal, x_cholesky)
        print(f"sq_error={sq_error}, sup_error={sup_error}")
        sq_errors.append(sq_error)
        sup_errors.append(sup_error)
    print(sq_errors, sup_errors)
    return sq_errors, sup_errors

def plot_histogram(errors, filename, xlabel=r'$\delta$', ylabel=r'$n$', clr='blue'):
    plt.figure(figsize=(12, 6))
    # x = np.linspace(10 ** (-6), 2 * 10 ** (-4), 200)
    plt.hist(errors, bins=100, color=clr, alpha=0.7)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(filename)
    plt.show()

def perform_experiment_positive_definite(num_matrices, b):
    sq_errors, sup_errors = run_experiment_positive_definite(num_matrices, b)
    plot_histogram(sq_errors,  "sq_positive_definite.pdf", clr='blue')
    plot_histogram(sup_errors, "sup_positive_definite.pdf", clr='green')



perform_experiment_positive_definite(num_matrices, b)

#
def generate_data(num_matrices, matrix_generator):
    spectral_radii = []
    condition_numbers = []

    for _ in range(num_matrices):
        A = matrix_generator()
        spectral_radii.append(np.max(np.abs(np.linalg.eigvals(A))))
        condition_numbers.append(np.linalg.cond(A))

    return spectral_radii, condition_numbers
#
def plot_histogram1(data, xlabel, ylabel, filename):
    plt.figure(figsize=(12, 6))
    plt.hist(data, bins=100, range=(0, 10), color='blue', alpha=0.7)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(filename)
    plt.show()


def plot_histogram2(data, xlabel, ylabel, filename):
    plt.figure(figsize=(12, 6))
    plt.hist(data, bins=100, color='blue', alpha=0.7)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(filename)
    plt.show()


# эксперимент 1 (используя функцию генерации матриц общего вида)
spectral_radii, condition_numbers = generate_data(num_matrices, generate_matrix)
plot_histogram2(spectral_radii, r'$\rho$', 'n', "spectral_general.pdf")
plot_histogram1(condition_numbers, r'$K$', 'n', "obuslov_general.pdf")

# эксперимент 2 (используя функцию генерации 3-х диагональных матриц)
spectral_radii, condition_numbers = generate_data(num_matrices, lambda: generate_matrix(tridiagonal=True))
plot_histogram2(spectral_radii, r'$\rho$', 'n', "spectral_tridiagonal.pdf")
plot_histogram1(condition_numbers, r'$K$', 'n', "obuslov_tridiagonal.pdf")

# эксперимент 3 (аналогично, для положительно определенных матриц)
spectral_radii, condition_numbers = generate_data(num_matrices, generate_positive_definite_matrix)
plot_histogram2(spectral_radii, r'$\rho$', 'n', "spectral_positive_defined.pdf")
plot_histogram2(condition_numbers, r'$K$', 'n', "obuslov_positive_defined.pdf")


def run_analysis(num_matrices, matrix_generator, method_universal, method_special, b):
    spectral_radii = []
    eigenvalue_ratios = []
    condition_numbers = []
    relative_errors = []

    for _ in range(num_matrices):
        A = matrix_generator()
        spectral_radius = max(abs(np.linalg.eigvals(A)))
        eigenvalue_ratio = max(abs(np.linalg.eigvals(A))) / min(abs(np.linalg.eigvals(A)))
        condition_number = np.linalg.cond(A)

        x_universal = method_universal(A, b)
        x_special = method_special(A, b)
        error = x_special - x_universal
        relative_error = np.linalg.norm(error, ord=2) / np.linalg.norm(x_universal, ord=2)

        spectral_radii.append(spectral_radius)
        eigenvalue_ratios.append(eigenvalue_ratio)
        condition_numbers.append(condition_number)
        relative_errors.append(relative_error)

    return spectral_radii, eigenvalue_ratios, condition_numbers, relative_errors

def plot_scatter(x_data, y_data, xlabel, ylabel, filename):
    plt.figure(figsize=(9, 9))
    plt.scatter(x_data, y_data, color='blue', alpha=0.7)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xscale('log')
    plt.yscale('log')
    plt.savefig(filename)
    plt.show()


spectral_radii, eigenvalue_ratios, condition_numbers, relative_errors = run_analysis(
    num_matrices,
    generate_matrix,
    lambda A, b: gauss(A, b, True),
    lambda A, b: thomas(A, b),
    b
)
plot_scatter(spectral_radii, relative_errors, r'$\rho$', r'$\delta$', "spectral_delta_general.pdf")
plot_scatter(eigenvalue_ratios, relative_errors, r'$\frac{|\lambda_{max}|}{|\lambda_{min}|}$', r'$\delta$', "eigenvalue_ratio_delta_general.pdf")
plot_scatter(condition_numbers, relative_errors, r'$K$', r'$\delta$', "condition_number_delta_general.pdf")

# для трехдиагональных матриц
spectral_radii, eigenvalue_ratios, condition_numbers, relative_errors = run_analysis(
    num_matrices,
    lambda: generate_matrix(tridiagonal=True),
    lambda A, b: gauss(A, b, True),
    lambda A, b: thomas(A, b),
    b
)
plot_scatter(spectral_radii, relative_errors, r'$\rho$', r'$\delta$', "spectral_delta_tridiagonal.pdf")
plot_scatter(eigenvalue_ratios, relative_errors, r'$\frac{|\lambda_{max}|}{|\lambda_{min}|}$', r'$\delta$', "eigenvalue_ratio_delta_tridiagonal.pdf")
plot_scatter(condition_numbers, relative_errors, r'$K$', r'$\delta$', "condition_number_delta_tridiagonal.pdf")

# для положительно определенных матриц
spectral_radii, eigenvalue_ratios, condition_numbers, relative_errors = run_analysis(
    num_matrices,
    generate_positive_definite_matrix,
    lambda A, b: gauss(A, b, True),
    lambda A, b: cholesky(A, b),
    b
)
plot_scatter(spectral_radii, relative_errors, r'$\rho$', r'$\delta$', "spectral_delta_positive_definite.pdf")
plot_scatter(eigenvalue_ratios, relative_errors, r'$\frac{|\lambda_{max}|}{|\lambda_{min}|}$', r'$\delta$', "eigenvalue_ratio_delta_positive_definite.pdf")
plot_scatter(condition_numbers, relative_errors, r'$K$', r'$\delta$', "condition_number_delta_positive_definite.pdf")