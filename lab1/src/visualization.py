import matplotlib.pyplot as plt
from auxiliary_modules import calculate_and_show_distance
def plot_spline_and_points(x_points, y_points, x_spline, spline_y, nearest_x, nearest_y, title, show_points=True):
    if x_spline is not None and spline_y is not None:
        plt.plot(x_spline, spline_y, color='r', label='$S(t)$')

    if nearest_x is not None and nearest_y is not None:
        plt.scatter(nearest_x, nearest_y, marker='o', color='g', label=r'$P*$', s=15)

    label_map = {
        'Визуализация всех точек': '$P$',
        'Визуализация выбранных точек': r'$\hat{P}$',
        'Визуализация полученных сплайнов': '$P$'
    }

    label = label_map.get(title, '$P$')
    if show_points:
        plt.scatter(x_points, y_points, marker='o', color='b', label=label, linewidth=0.5, s=10)

def plot_vector_fields(vectors):
    for i, (x, y, dxdt, dydt) in enumerate(vectors):
        if i == 0:
            plt.quiver(x, y, dxdt, dydt, color='g', angles='xy',
                       label=r'$G(t)=\frac{d}{dt}S(t)$', scale_units='xy',
                       scale=0.5, width=0.002, zorder=2)
            plt.quiver(x, y, -dydt, dxdt, color='b', angles='xy',
                       label='$R(t)$', scale_units='xy', scale=0.5,
                       width=0.002, zorder=2)
        else:
            plt.quiver(x, y, -dydt, dxdt, color='b', angles='xy',
                       scale_units='xy', scale=0.5, width=0.002, zorder=2)
            plt.quiver(x, y, dxdt, dydt, color='g', angles='xy',
                       scale_units='xy', scale=0.5, width=0.002, zorder=2)
    plt.ylim(0.5, 0.6)
    plt.xlim(-0.5, -0.4)

def plot_optimized_points(x_points, y_points, nearest_x, nearest_y):
    for i in range(len(x_points)):
        plt.plot([x_points[i], nearest_x[i]], [y_points[i], nearest_y[i]], linestyle='-', color='b')
    calculate_and_show_distance(x_points, y_points, nearest_x, nearest_y)
    plt.ylim(0.56, 0.58)
    plt.xlim(-0.44, -0.42)

def plot_points(x_points, y_points, selected_x_points=None, selected_y_points=None, title='Визуализация всех точек',
                **kwargs):
    plt.figure(figsize=(10, 10))

    x_spline = kwargs.get('x_spline', None)
    spline_y = kwargs.get('spline_y', None)
    nearest_x = kwargs.get('nearest_x', None)
    nearest_y = kwargs.get('nearest_y', None)
    show_vector_field = kwargs.get('show_vector_field', False)

    plot_spline_and_points(x_points, y_points, x_spline, spline_y, nearest_x, nearest_y, title,
                           show_points=not show_vector_field)

    plt.grid(True)
    plt.axis('equal')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(title)
    if show_vector_field:
        vectors = kwargs.get('vectors')
        plot_vector_fields(vectors)

    if kwargs.get('optimize'):
        plot_optimized_points(x_points, y_points, nearest_x, nearest_y)

    if selected_x_points is not None and selected_y_points is not None:
        plt.scatter(selected_x_points, selected_y_points, marker='o', color='g', label=r'$\hat{P}$', s=20, zorder=4)

    plt.legend(loc='lower right')
    plt.savefig('splines.pdf', format='pdf', bbox_inches='tight')
    plt.show()