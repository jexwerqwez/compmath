import numpy as np
from base import gauss, thomas

# генерация случайных тестовых матриц и векторов
def generate_test_matrices_and_vectors(num_matrices, size):
    matrices = []
    vectors = []
    for _ in range(num_matrices):
        # генерация случайной квадратной матрицы и вектора
        A = np.random.rand(size, size).astype(np.float32)
        b = np.random.rand(size).astype(np.float32)
        matrices.append(A)
        vectors.append(b)
    return matrices, vectors


test_matrices, test_vectors = generate_test_matrices_and_vectors(1000, 6)

# определение "универсального" метода
def evaluate_methods(matrices, vectors):
    errors = {
        'Метод Гаусса без выбора главного элемента': [],
        'Метод Гаусса с выбором главного элемента': [],
        'Метод прогонки': []
    }

    for A, b in zip(matrices, vectors):
        x_gauss_no_pivot = gauss(A.copy(), b.copy(), pivoting=False)
        error_gauss_no_pivot = np.linalg.norm(A @ x_gauss_no_pivot - b)
        errors['Метод Гаусса без выбора главного элемента'].append(error_gauss_no_pivot)

        x_gauss_pivot = gauss(A.copy(), b.copy(), pivoting=True)
        error_gauss_pivot = np.linalg.norm(A @ x_gauss_pivot - b)
        errors['Метод Гаусса с выбором главного элемента'].append(error_gauss_pivot)

        x_thomas = thomas(A.copy(), b.copy())
        error_thomas = np.linalg.norm(A @ x_thomas - b)
        errors['Метод прогонки'].append(error_thomas)

    # вычисление средних погрешностей
    average_errors = {method: np.mean(error_list) for method, error_list in errors.items() if error_list}
    return average_errors

# вычисление погрешностей для тестовых данных
average_errors = evaluate_methods(test_matrices, test_vectors)
print(average_errors)
