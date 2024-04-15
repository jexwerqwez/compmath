import numpy as np

def calculate_and_show_distance(x_points, y_points, x_spline, spline_y):
    distance = []
    for real_x, real_y, approx_x, approx_y in zip(x_points, y_points, x_spline, spline_y):
        d = ((real_x - approx_x) ** 2 + (real_y - approx_y) ** 2) ** 0.5
        distance.append(d)
    print("Среднее отклонение:", np.array(distance).mean())
    print("Стандартное отклонение:", np.array(distance).std())

def get_spline_value(a, b, c, d, dt):
    return a + b * dt + c * dt ** 2 + d * dt ** 3

def load_points_from_file(filename):
    try:
        points = np.loadtxt(filename)
        x_points, y_points = points[:, 0], points[:, 1]
        return x_points, y_points
    except Exception as e:
        print(f"Error reading the file: {e}")
        return None, None