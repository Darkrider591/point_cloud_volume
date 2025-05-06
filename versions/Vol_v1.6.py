import os
import numpy as np
from scipy.spatial import Delaunay, ConvexHull, KDTree
from shapely.geometry import Polygon, Point
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Функция для расчета площади треугольника по формуле Герона
def triangle_area(a, b, c):
    sides = np.array([np.linalg.norm(b-a), np.linalg.norm(c-b), np.linalg.norm(a-c)])
    s = sum(sides)/2
    return np.sqrt(s*(s-sides[0])*(s-sides[1])*(s-sides[2]))

def load_points(filename):
    try:
        if not os.path.exists(filename):
            print(f"Файл не найден: {filename}")
            return None

        data = np.loadtxt(filename)
        if data.size == 0:
            print(f"Файл пустой: {filename}")
            return None

        if data.shape[1] < 3:
            print(f"Недостаточно столбцов в {filename}. Нужны X,Y,Z координаты")
            return None

        print(f"Загружено {len(data)} точек из {filename}")
        return data[:, :3]

    except Exception as e:
        print(f"Ошибка при чтении {filename}: {str(e)}")
        return None

def create_polygon_from_hull(hull_vertices):
    polygon = Polygon(hull_vertices)
    return polygon

def point_inside_polygon(point, polygon):
    pnt = Point(*point[:2])
    return polygon.contains(pnt)

def calculate_2d_projection_area(points):
    """Рассчитывает площадь проекции поверхности на плоскость XY"""
    hull = ConvexHull(points[:, :2])
    return hull.volume  # Для ConvexHull 2D — это площадь

def calculate_3d_surface_area(points):
    """Рассчитывает реальную площадь поверхности с учётом Z-координат"""
    tri = Delaunay(points[:, :2])
    total_area = 0.0
    for simplex in tri.simplices:
        A, B, C = points[simplex]
        total_area += triangle_area(A, B, C)
    return total_area

def visualize_surfaces(upper_points, lower_points, result=None):
    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_box_aspect((np.ptp(upper_points[:, 0]), np.ptp(upper_points[:, 1]), np.ptp(upper_points[:, 2])))

    ax.scatter(upper_points[:, 0], upper_points[:, 1], upper_points[:, 2],
               color='blue', label='Верхняя поверхность', marker='o')
    ax.scatter(lower_points[:, 0], lower_points[:, 1], lower_points[:, 2],
               color='green', label='Нижняя поверхность', marker='^')

    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.legend()
    plt.title("Визуализация поверхностей с результатами расчёта")

    if result is not None:
        nasyp = result['nasyp']
        vyemka = result['vyemka']
        final = result['final_volume']

        area_upper_2d = result['area_upper_2d']
        area_lower_2d = result['area_lower_2d']
        area_upper_3d = result['area_upper_3d']
        area_lower_3d = result['area_lower_3d']

        text_lines = [
            f'Объем насыпи:         {nasyp:.2f} куб. ед.',
            f'Объем выемки:         {vyemka:.2f} куб. ед.'
        ]
        if final >= 0:
            text_lines.append(f'Итоговый объем (насыпь):   {final:.2f} куб. ед.')
        else:
            text_lines.append(f'Итоговый объем (выемка): {-final:.2f} куб. ед.')

        text_lines.extend([
            '',
            f'Площадь верхней поверхности:',
            f'  - 2D проекция: {area_upper_2d:.2f} кв. ед.',
            f'  - 3D фактическая: {area_upper_3d:.2f} кв. ед.',
            f'Площадь нижней поверхности:',
            f'  - 2D проекция: {area_lower_2d:.2f} кв. ед.',
            f'  - 3D фактическая: {area_lower_3d:.2f} кв. ед.',
        ])

        text_str = '\n'.join(text_lines)

        plt.figtext(
            0.15, 0.95, text_str,
            ha='left', va='top',
            fontsize=10,
            bbox=dict(facecolor='white', edgecolor='lightgray', alpha=0.8)
        )

    plt.tight_layout()
    plt.show()

def calculate_volume(upper_points, lower_points):
    try:
        if len(upper_points) < 4 or len(lower_points) < 4:
            print("Нужно минимум 4 точки в каждом файле для триангуляции")
            return None

        kd_lower = KDTree(lower_points[:, :2])
        tri_upper = Delaunay(upper_points[:, :2])

        hull_upper = ConvexHull(upper_points[:, :2])
        poly_upper = create_polygon_from_hull(hull_upper.points[hull_upper.vertices])

        total_volume_nasyp = 0.0
        total_volume_vyemka = 0.0

        for simplex in tri_upper.simplices:
            A, B, C = upper_points[simplex]
            centroid = (A[:2] + B[:2] + C[:2]) / 3

            if point_inside_polygon(centroid, poly_upper):
                area = triangle_area(A[:2], B[:2], C[:2])
                z_upper = (A[2] + B[2] + C[2]) / 3

                _, idx_A = kd_lower.query(A[:2])
                _, idx_B = kd_lower.query(B[:2])
                _, idx_C = kd_lower.query(C[:2])

                z_lower = (lower_points[idx_A, 2] + lower_points[idx_B, 2] + lower_points[idx_C, 2]) / 3
                height_diff = z_upper - z_lower

                if height_diff >= 0:
                    total_volume_nasyp += area * height_diff
                else:
                    total_volume_vyemka -= area * height_diff

        final_volume = total_volume_nasyp - total_volume_vyemka

        return {
            'nasyp': total_volume_nasyp,
            'vyemka': total_volume_vyemka,
            'final_volume': final_volume,

            'area_upper_2d': calculate_2d_projection_area(upper_points),
            'area_lower_2d': calculate_2d_projection_area(lower_points),

            'area_upper_3d': calculate_3d_surface_area(upper_points),
            'area_lower_3d': calculate_3d_surface_area(lower_points)
        }

    except Exception as e:
        print(f"\nОшибка при расчете: {str(e)}")
        return None

def select_files():
    root = tk.Tk()
    root.withdraw()
    filetypes = (
        ("Text files", "*.txt"),
        ("All files", "*.*")
    )
    upper_file_path = filedialog.askopenfilename(title="Выберите файл верхней поверхности", filetypes=filetypes)
    lower_file_path = filedialog.askopenfilename(title="Выберите файл нижней поверхности", filetypes=filetypes)
    return upper_file_path, lower_file_path

def main():
    print("=== Точный расчет объема и площади между поверхностями ===")
    upper_file, lower_file = select_files()
    
    if not upper_file or not lower_file:
        print("\nНеобходимо выбрать оба файла!")
        return

    print(f"\nВерхняя поверхность: {upper_file}")
    print(f"Нижняя поверхность: {lower_file}")

    upper_points = load_points(upper_file)
    lower_points = load_points(lower_file)

    if upper_points is None or lower_points is None:
        print("\nОстановка из-за проблем с данными")
        return

    result = calculate_volume(upper_points, lower_points)

    if result is not None:
        visualize_surfaces(upper_points, lower_points, result)
    else:
        print("\nНе удалось рассчитать объем")

if __name__ == "__main__":
    main()