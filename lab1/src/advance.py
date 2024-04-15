import numpy as np
from visualization import plot_points
from auxiliary_modules import get_spline_value, load_points_from_file

class AutoDiffNum:
    def __init__(self, a, b=0):
        self._a = a
        self._b = b

    def __add__(self, other):
        if isinstance(other, AutoDiffNum):
            return AutoDiffNum(self._a + other._a, self._b + other._b)
        return AutoDiffNum(self._a + other, self._b)

    def __sub__(self, other):
        if isinstance(other, AutoDiffNum):
            return AutoDiffNum(self._a - other._a, self._b - other._b)
        return AutoDiffNum(self._a - other, self._b)

    def __mul__(self, other):
        if isinstance(other, AutoDiffNum):
            return AutoDiffNum(self._a * other._a, self._a * other._b + self._b * other._a)
        return AutoDiffNum(self._a * other, self._b * other)

def spline_deriv(ax, bx, cx, dx, ay, by, cy, dy, t_j, t):
    dxdt = (t - t_j) * (t - t_j) * (t - t_j) * dx + (t - t_j) * (t - t_j) * cx + (t - t_j) * bx + ax
    dydt = (t - t_j) * (t - t_j) * (t - t_j) * dy + (t - t_j) * (t - t_j) * cy + (t - t_j) * by + ay
    return dxdt._b, dydt._b

def normalize_vector(dxdt, dydt):
    magnitude = ((dxdt ** 2 + dydt ** 2) ** 0.5) * 300
    return dxdt / magnitude, dydt / magnitude

def compute_vector_fields(selected_x_points, selected_y_points, a_x, b_x, c_x, d_x, a_y, b_y, c_y, d_y, t, z):
    vectors = []
    for i, (ax, bx, cx, dx, ay, by, cy, dy) in enumerate(
            zip(a_x[::z], b_x[::z], c_x[::z], d_x[::z], a_y[::z], b_y[::z], c_y[::z], d_y[::z])):
        dxdt, dydt = spline_deriv(ax, bx, cx, dx, ay, by, cy, dy, t[i], AutoDiffNum(t[i], 1))
        normalized_dxdt, normalized_dydt = normalize_vector(dxdt, dydt)
        vectors.append((selected_x_points[i * z], selected_y_points[i * z], normalized_dxdt, normalized_dydt))

    return vectors

def read_coefficients_from_file(filename_in):
    with open(filename_in, 'r') as file:
        next(file)
        coef_matrix = np.loadtxt(file)
        coefficients_x = coef_matrix[:, :4]
        coefficients_y = coef_matrix[:, 4:]
    return coefficients_x, coefficients_y
def distance_from_spline_to_point(ax, bx, cx, dx, ay, by, cy, dy, dt, x: float, y: float) -> float:
    x_t = get_spline_value(ax, bx, cx, dx, dt)
    y_t = get_spline_value(ay, by, cy, dy, dt)

    return ((x - x_t) ** 2 + (y - y_t) ** 2) ** 0.5

def find_closest_point(x: float, y: float, ax, bx, cx, dx, ay, by, cy, dy, t_last: float):
    a = 0
    b = t_last
    for _ in range(10):
        t_mid1 = a + (b - a) / 3
        t_mid2 = a + (b - a) * 2 / 3
        f_mid1 = distance_from_spline_to_point(ax, bx, cx, dx, ay, by, cy, dy, t_mid1, x, y)
        f_mid2 = distance_from_spline_to_point(ax, bx, cx, dx, ay, by, cy, dy, t_mid2, x, y)
        if f_mid1 < f_mid2:
            b = t_mid2
        else:
            a = t_mid1
    return (a + b) / 2

def calculate_nearest_points(x_points, y_points, a_x, b_x, c_x, d_x, a_y, b_y, c_y, d_y):
    h_i = 0.1
    nearest_x = np.empty_like(x_points)
    nearest_y = np.empty_like(y_points)
    factor = 10
    for i, (x, y) in enumerate(zip(x_points, y_points)):
        spline_idx = int(i // factor)
        if i % factor == 0:
            nearest_x[i] = get_spline_value(a_x[spline_idx], 0, 0, 0, 0)
            nearest_y[i] = get_spline_value(a_y[spline_idx], 0, 0, 0, 0)
            continue
        optimal_t = find_closest_point(x, y, a_x[spline_idx], b_x[spline_idx], c_x[spline_idx],
                                       d_x[spline_idx], a_y[spline_idx], b_y[spline_idx],
                                       c_y[spline_idx], d_y[spline_idx], h_i * factor)
        nearest_x[i] = get_spline_value(a_x[spline_idx], b_x[spline_idx], c_x[spline_idx], d_x[spline_idx],
                                        optimal_t)
        nearest_y[i] = get_spline_value(a_y[spline_idx], b_y[spline_idx], c_y[spline_idx], d_y[spline_idx],
                                        optimal_t)

    return nearest_x, nearest_y

def lab1_advance(points_filename, coeffs_filename, selected_x_points, selected_y_points, x_spline, spline_y, t, t_dist,
                 z=6):
    x_points, y_points = load_points_from_file(points_filename)
    coefficients_x, coefficients_y = read_coefficients_from_file(coeffs_filename)
    a_x, b_x, c_x, d_x = coefficients_x.T
    a_y, b_y, c_y, d_y = coefficients_y.T

    # задания 8-11
    vectors = compute_vector_fields(selected_x_points, selected_y_points, a_x, b_x, c_x, d_x, a_y, b_y, c_y, d_y, t, z)
    plot_points(x_points, y_points, title='Визуализация полученных сплайнов',
                x_spline=x_spline, spline_y=spline_y, show_vector_field=True, vectors=vectors)

    # задание 12
    nearest_x, nearest_y = calculate_nearest_points(x_points, y_points, a_x, b_x, c_x, d_x, a_y, b_y, c_y, d_y)
    plot_points(x_points, y_points, title='Визуализация оптимизационной задачи', x_spline=x_spline, spline_y=spline_y,
                nearest_x=nearest_x, nearest_y=nearest_y, optimize=True)