import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family': 'serif', 'font.serif': 'Times New Roman', 'font.size': 18})


# реализация методов Гаусса
def gauss(A, b, pivoting=False):
    n = A.shape[0]
    x = np.zeros(n, dtype=np.float32)
    aug_matrix = np.hstack((A.astype(np.float32), b.astype(np.float32).reshape(-1, 1)))
    # прямой ход метода Гаусса
    for k in range(n):
        if pivoting:
            # частичный выбор главного элемента
            max_row = k + np.argmax(np.abs(aug_matrix[k:, k]))
            aug_matrix[[k, max_row]] = aug_matrix[[max_row, k]]
        # обнуление элементов под главной диагональю
        for i in range(k + 1, n):
            factor = aug_matrix[i, k] / aug_matrix[k, k]
            aug_matrix[i, k:] -= factor * aug_matrix[k, k:]
    # обратный ход метода Гаусса
    for i in range(n - 1, -1, -1):
        sum_val = np.dot(aug_matrix[i, i + 1:n], x[i + 1:n])
        x[i] = (aug_matrix[i, -1] - sum_val) / aug_matrix[i, i]
    return x.reshape(-1, 1)


# реализация метода прогонки
def thomas(A, b):
    n = b.shape[0]
    alph = np.zeros(n - 1, dtype=np.float32)
    bet = np.zeros(n - 1, dtype=np.float32)
    x = np.zeros((n, 1), dtype=np.float32)

    # Расчет начальных коэффициентов
    alph[0] = -A[0, 1] / A[0, 0]
    bet[0] = b[0] / A[0, 0]
    for i in range(1, n - 1):
        a = A[i, i - 1]
        b1 = A[i, i]
        c = A[i, i + 1]
        alph[i] = -c / (a * alph[i - 1] + b1)
        bet[i] = (b[i] - a * bet[i - 1]) / (a * alph[i - 1] + b1)
    x[-1] = (b[-1] - A[-1, -2] * bet[-1]) / (A[-1, -1] + A[-1, -2] * alph[-1])

    for i in range(n - 2, -1, -1):
        x[i] = alph[i] * x[i + 1] + bet[i]

    return x

# генерация невырожденых и 3-х диагональных матриц
def generate_matrix(size=6, tridiagonal=False):
    while True:
        if tridiagonal:
            A = np.zeros((size, size), dtype=np.float32)
            # заполнение главной и соседних диагоналей
            for i in range(size):
                A[i, i] = np.random.uniform(-1, 1)
                if i > 0:
                    A[i, i - 1] = A[i - 1, i] = np.random.uniform(-1, 1)
        else:
            A = np.random.uniform(-1, 1, (size, size)).astype(np.float32)
        # проверка на невырожденность
        if np.linalg.det(A) != 0:
            return A


# определение ошибок
def calculate_error(true_solution, computed_solution):
    # вычисление относительной погрешности
    relative_error = np.abs((true_solution - computed_solution) / true_solution)
    # среднеквадратичная норма
    sq_error = np.sqrt(np.mean(np.square(relative_error)))
    # супремум-норма
    sup_error = np.max(relative_error)
    return sq_error, sup_error


def run_experiment(matrix_type, num_matrices, b):
    sq_errors = []
    sup_errors = []

    for _ in range(num_matrices):
        A = generate_matrix(tridiagonal=matrix_type == 'tridiagonal')
        x_universal = gauss(A, b, True)  # Метод Гаусса с выбором главного элемента
        x_special = thomas(A, b) if matrix_type == 'tridiagonal' else gauss(A, b, False)
        sq_error, sup_error = calculate_error(x_universal, x_special)
        sq_errors.append(sq_error)
        sup_errors.append(sup_error)
    return sq_errors, sup_errors

def plot_histogram(errors, filename, xlabel=r'$\delta$', ylabel=r'$n$', clr='blue'):
    plt.figure(figsize=(12, 6))
    x = np.linspace(10 ** (-7), 2 * 10 ** (-5), 200)
    plt.hist(errors, x, color=clr, alpha=0.7)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(filename)
    plt.show()


def perform_experiment_general(num_matrices, b):
    sq_errors, sup_errors = run_experiment('general', num_matrices, b)
    plot_histogram(sq_errors, "sq_general.pdf", clr='blue')
    plot_histogram(sup_errors, "sup_general.pdf", clr='green')

def perform_experiment_tridiagonal(num_matrices, b):
    sq_errors, sup_errors = run_experiment('tridiagonal', num_matrices, b)
    plot_histogram(sq_errors, "sq_tridiagonal.pdf", clr='blue')
    # print(len(sup_errors))
    plot_histogram(sup_errors, "sup_tridiagonal.pdf", clr='green')


num_matrices = 1000
b = np.array([1, 1, 1, 1, 1, 1], dtype=np.float32)
perform_experiment_general(num_matrices, b)
perform_experiment_tridiagonal(num_matrices, b)
