from base import lab1_base
from advance import lab1_advance

if __name__ == '__main__':
    selected_x_points, selected_y_points, x_spline, spline_y, t, t_dist = lab1_base('contour.txt', 10, 'coeffs.txt')
    lab1_advance('contour.txt', 'coeffs.txt', selected_x_points, selected_y_points, x_spline, spline_y, t, t_dist)