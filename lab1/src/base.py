import numpy as np
from matplotlib import rcParams
from visualization import plot_points
from auxiliary_modules import calculate_and_show_distance, get_spline_value, load_points_from_file
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = 'Times New Roman'

rcParams.update({'font.size': 22})

def select_points(x_points, y_points, m):
    selected_x_points = x_points[::m]
    selected_y_points = y_points[::m]
    if selected_x_points[-1] != x_points[-1] or selected_y_points[-1] != y_points[-1]:
        selected_x_points = np.append(selected_x_points, x_points[-1])
        selected_y_points = np.append(selected_y_points, y_points[-1])
    return selected_x_points, selected_y_points

def fill_matrix(n, t_dist):
    matrix_A = np.zeros((n, n))
    matrix_A[0][0] = 1
    matrix_A[-1][-1] = 1
    for i in range(1, matrix_A.shape[0] - 1):
        matrix_A[i][i - 1] = t_dist[i - 1]
        matrix_A[i][i] = 2 * (t_dist[i - 1] + t_dist[i])
        matrix_A[i][i + 1] = t_dist[i]
    return matrix_A

def fill_vector(n, t_dist, selected_points):
    vector_b = np.zeros((n))
    vector_b[0] = vector_b[-1] = 0
    for i in range(1, vector_b.shape[0] - 1):
        vector_b[i] = 3 / t_dist[i] * (selected_points[i + 1] - selected_points[i]) - 3 / t_dist[i - 1] * (
                    selected_points[i] - selected_points[i - 1])
    return vector_b

def get_natural_coordinate(x_points, M, factor, h):
    t = np.arange(0, x_points.shape[0] / M, factor * h)
    if t[-1] != x_points.shape[0] / M:
        t = np.append(t, x_points.shape[0] / M)
    return t

def get_t_dist(t):
    return np.diff(t)

def compute_c(matrix_A, vector_b):
    return np.linalg.solve(matrix_A, vector_b)

def compute_b(n, t_dist, selected_points, coeff_c):
    coeff_b = np.zeros((n - 1))
    for i in range(coeff_b.shape[0]):
        coeff_b[i] = 1 / t_dist[i] * (selected_points[i + 1] - selected_points[i]) - t_dist[i] / 3 * (
                    coeff_c[i + 1] + 2 * coeff_c[i])
    return coeff_b

def compute_d(n, t_dist, coeff_c):
    coeff_d = np.zeros((n - 1))
    for i in range(coeff_d.shape[0]):
        coeff_d[i] = (coeff_c[i + 1] - coeff_c[i]) / (3 * t_dist[i])
    return coeff_d

def compute_a(selected_points):
    return selected_points[:-1]

def compute_spline_coefficients(selected_points, t_dist):
    n = selected_points.shape[0]
    matrix_A = fill_matrix(n, t_dist)
    vector_b = fill_vector(n, t_dist, selected_points)
    coeff_c = compute_c(matrix_A, vector_b)
    coeff_b = compute_b(n, t_dist, selected_points, coeff_c)
    coeff_d = compute_d(n, t_dist, coeff_c)
    coeff_a = compute_a(selected_points)
    coeff_c = coeff_c[:-1]
    return coeff_a, coeff_b, coeff_c, coeff_d

def fill_spline(a_x, b_x, c_x, d_x, a_y, b_y, c_y, d_y, t_dist, factor, last_spline_points_count):
    x_spline = []
    spline_y = []
    for i, (ax, bx, cx, dx, ay, by, cy, dy) in enumerate(zip(a_x, b_x, c_x, d_x, a_y, b_y, c_y, d_y)):
        if i == len(a_x) - 1:
            dt_for_plotting = np.linspace(0, t_dist[i] - 0.1, num=last_spline_points_count)
        else:
            dt_for_plotting = np.linspace(0, t_dist[i] - 0.1, num=factor)
        x_spline.extend(get_spline_value(ax, bx, cx, dx, dt_for_plotting))
        spline_y.extend(get_spline_value(ay, by, cy, dy, dt_for_plotting))
    return x_spline, spline_y

def write_coefficients_to_file(filename_out, coef_matrix):
    with open(filename_out, 'w') as file:
        file.write("# Combined Coefficients\n")
        np.savetxt(file, coef_matrix)

def get_spline_value(a, b, c, d, dt):
    return a + b * dt + c * dt ** 2 + d * dt ** 3

def lab1_base(filename_in: str, factor: int, filename_out: str):
    # задание 2
    x_points, y_points = load_points_from_file(filename_in)
    if x_points is None or y_points is None:
        print("Failed to load points from file.")
        return
    plot_points(x_points, y_points)

    # задание 3
    selected_x_points, selected_y_points = select_points(x_points, y_points, factor)
    plot_points(selected_x_points, selected_y_points, title='Визуализация выбранных точек')

    # задание 4
    t = get_natural_coordinate(x_points, 10, factor, 0.1)
    t_dist = get_t_dist(t)
    a_x, b_x, c_x, d_x = compute_spline_coefficients(selected_x_points, t_dist)
    a_y, b_y, c_y, d_y = compute_spline_coefficients(selected_y_points, t_dist)

    last_spline_points_count = len(x_points) - (len(selected_x_points) - 2) * factor
    x_spline, spline_y = fill_spline(a_x, b_x, c_x, d_x, a_y, b_y, c_y, d_y, t_dist, factor, last_spline_points_count)

    # задание 5
    calculate_and_show_distance(x_points, y_points, x_spline, spline_y)

    # задание 6
    plot_points(x_points=x_points, y_points=y_points, x_spline=x_spline, spline_y=spline_y,
                title='Визуализация полученных сплайнов', selected_x_points=selected_x_points,
                selected_y_points=selected_y_points)

    # задание 7
    coefficients_x = np.column_stack((a_x, b_x, c_x, d_x))
    coefficients_y = np.column_stack((a_y, b_y, c_y, d_y))
    coef_matrix = np.hstack((coefficients_x, coefficients_y))
    write_coefficients_to_file(filename_out, coef_matrix)
    return selected_x_points, selected_y_points, x_spline, spline_y, t, t_dist