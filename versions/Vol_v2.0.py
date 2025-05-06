import os
import numpy as np
from scipy.spatial import Delaunay, ConvexHull, KDTree
from shapely.geometry import Polygon, Point
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import matplotlib.pyplot as plt
import pyvista as pv
import laspy
import pandas as pd


# --- Загрузка точек из разных форматов ---
def load_points(filename):
    ext = os.path.splitext(filename)[1].lower()
    if ext in [".txt", ".csv", ".xyz"]:
        return load_points_txt(filename)
    elif ext == ".las":
        return load_points_las(filename)
    else:
        print(f"Неизвестный формат файла: {ext}")
        return None


def load_points_txt(filename):
    try:
        data = np.loadtxt(filename)
        if data.shape[1] < 3:
            print("Нужны X,Y,Z координаты")
            return None
        print(f"Загружено {len(data)} точек из {filename}")
        return data[:, :3]
    except Exception as e:
        print(f"Ошибка при чтении файла: {str(e)}")
        return None


def load_points_las(filename):
    try:
        las = laspy.read(filename)
        points = np.vstack((las.x, las.y, las.z)).transpose()
        print(f"Загружено {len(points)} точек из LAS-файла")
        return points
    except Exception as e:
        print(f"Ошибка чтения LAS-файла: {e}")
        return None


# --- Расчёт площади треугольника по формуле Герона ---
def triangle_area(a, b, c):
    sides = np.array([np.linalg.norm(b - a), np.linalg.norm(c - b), np.linalg.norm(a - c)])
    s = sum(sides) / 2
    return np.sqrt(s * (s - sides[0]) * (s - sides[1]) * (s - sides[2]))


# --- Площадь проекции (2D) ---
def calculate_2d_projection_area(points):
    if len(points) < 3:
        print("Недостаточно точек для Convex Hull")
        return 0.0
    try:
        hull = ConvexHull(points[:, :2])
        return hull.volume
    except:
        print("Convex Hull не может быть создан")
        return 0.0


# --- Реальная площадь поверхности (3D) ---
def calculate_3d_surface_area(points):
    if len(points) < 3:
        print("Недостаточно точек для триангуляции")
        return 0.0
    try:
        tri = Delaunay(points[:, :2])
        total_area = 0.0
        for simplex in tri.simplices:
            A, B, C = points[simplex]
            total_area += triangle_area(A, B, C)
        return total_area
    except:
        print("Триангуляция не выполнена")
        return 0.0


# --- Расчёт объёма ---
def calculate_volume(upper_points, lower_points):
    try:
        if len(upper_points) < 4 or len(lower_points) < 4:
            raise ValueError("Минимум 4 точки в каждом файле")

        kd_lower = KDTree(lower_points[:, :2])

        # Проверка перед ConvexHull
        if len(upper_points) >= 3:
            try:
                hull_upper = ConvexHull(upper_points[:, :2])
                poly_upper = Polygon(hull_upper.points[hull_upper.vertices])
            except:
                print("Не удалось создать Convex Hull для верхней поверхности")
                poly_upper = None
        else:
            poly_upper = None

        total_volume_nasyp = 0.0
        total_volume_vyemka = 0.0

        tri_upper = Delaunay(upper_points[:, :2])
        for simplex in tri_upper.simplices:
            A, B, C = upper_points[simplex]
            centroid = (A[:2] + B[:2] + C[:2]) / 3

            if poly_upper is None or Polygon(poly_upper).contains(Point(*centroid)):
                area = triangle_area(A[:2], B[:2], C[:2])
                z_upper = (A[2] + B[2] + C[2]) / 3

                dists, idxs = kd_lower.query(centroid, k=3)
                weights = 1 / (dists + 1e-8)
                z_lower = np.average(lower_points[idxs, 2], weights=weights)

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
        messagebox.showerror("Ошибка", f"Ошибка расчёта: {str(e)}")
        return None


# --- Визуализация с помощью PyVista ---
def visualize_with_pyvista(upper_points, lower_points, result=None):
    try:
        plotter = pv.Plotter()
        ...
        if result:
            lines = [ ... ]
            plotter.add_text('\n'.join(lines), position='lower_left', font_size=12)
        ...
        plotter.show()
    except Exception as e:
        print("Ошибка при визуализации:", str(e))
    
    # Объединение координат для масштабирования
    all_points = np.vstack((upper_points, lower_points))
    min_coords = np.min(all_points, axis=0)
    max_coords = np.max(all_points, axis=0)
    ranges = max_coords - min_coords
    center = (min_coords + max_coords) / 2
    scale = np.max(ranges)

    # Создание сеток точек
    upper_cloud = pv.PolyData(upper_points)
    lower_cloud = pv.PolyData(lower_points)

    # Цветовая карта высот
    def get_colors(points):
        z_values = points[:, 2]
        norm_z = (z_values - np.min(z_values)) / (np.max(z_values) - np.min(z_values))
        cmap = plt.get_cmap('viridis')
        return cmap(norm_z)

    upper_colors = (get_colors(upper_points)[:, :3] * 255).astype(int)
    lower_colors = (get_colors(lower_points)[:, :3] * 255).astype(int)

    # Добавление точек
    plotter.add_mesh(upper_cloud, point_size=5, color='red', label="Верхняя поверхность")
    plotter.add_mesh(lower_cloud, point_size=5, color='blue', label="Нижняя поверхность")

    # Равномерный масштаб по всем осям
    plotter.set_scale(xscale=1, yscale=1, zscale=1)
    plotter.view_isometric()

    # Подпись результатов
    if result:
     lines = [
        f'Volume Fill:         {result["nasyp"]:.2f} куб. ед.',
        f'Volume Cut:         {result["vyemka"]:.2f} куб. ед.',
        f'Summary volume:       {result["final_volume"]:.2f} куб. ед.',
        '',
        f'Upper surface area',
        f'  - 2D projection:      {result["area_upper_2d"]:.2f}',
        f'  - 3D actual:   {result["area_upper_3d"]:.2f}',
        f'Bottom surface area:',
        f'  - 2D projection:      {result["area_lower_2d"]:.2f}',
        f'  - 3D actual:   {result["area_lower_3d"]:.2f}'
    ]
    plotter.add_text('\n'.join(lines), position='lower_left', font_size=9)

    plotter.add_legend([
        ('Upper surface', 'red'),
        ('Bottom surface', 'blue')
    ])

    plotter.show()


# --- GUI ---
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Calculation of volumes between surfaces")
        self.geometry("1000x600")
        self.upper_points = None
        self.lower_points = None
        self.result = None
        self.create_widgets()

    def create_widgets(self):
        frame_top = ttk.Frame(self)
        frame_top.pack(pady=10, fill=tk.X)

        ttk.Button(frame_top, text="Select top surface",
                   command=self.load_upper).pack(side=tk.LEFT, padx=5)
        ttk.Button(frame_top, text="Select bottom surface",
                   command=self.load_lower).pack(side=tk.LEFT, padx=5)
        ttk.Button(frame_top, text="Calculate",
                   command=self.calculate).pack(side=tk.LEFT, padx=5)
        ttk.Button(frame_top, text="Show chart",
                   command=self.show_plot).pack(side=tk.LEFT, padx=5)
        ttk.Button(frame_top, text="Save results",
                   command=self.save_results).pack(side=tk.LEFT, padx=5)
        ttk.Button(frame_top, text="Export to Excel",
                   command=self.save_results_excel).pack(side=tk.LEFT, padx=5)

    def load_upper(self):
        path = filedialog.askopenfilename(filetypes=[("Поддерживаемые форматы", "*.txt *.csv *.xyz *.las")])
        if path:
            self.upper_points = load_points(path)

    def load_lower(self):
        path = filedialog.askopenfilename(filetypes=[("Поддерживаемые форматы", "*.txt *.csv *.xyz *.las")])
        if path:
            self.lower_points = load_points(path)

    def calculate(self):
        if self.upper_points is None or self.lower_points is None:
            messagebox.showwarning("Ошибка", "Не выбраны оба файла!")
            return
        self.result = calculate_volume(self.upper_points, self.lower_points)
        messagebox.showinfo("Готово", "Расчёт выполнен успешно!")

    def show_plot(self):
        if self.result is None:
            messagebox.showwarning("Ошибка", "Сначала выполните расчёт!")
            return
        visualize_with_pyvista(self.upper_points, self.lower_points, self.result)

    def save_results(self):
        if self.result is None:
            messagebox.showwarning("Ошибка", "Сначала выполните расчёт!")
            return
        path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text files", "*.txt")])
        if path:
            with open(path, 'w') as f:
                f.write("=== Результаты расчёта ===\n")
                f.write(f"Объём насыпи:      {self.result['nasyp']:.2f} куб. ед.\n")
                f.write(f"Объём выемки:      {self.result['vyemka']:.2f} куб. ед.\n")
                f.write(f"Итоговый объём:    {self.result['final_volume']:.2f} куб. ед.\n")
                f.write(f"Площадь верхней поверхности:\n")
                f.write(f"  - 2D проекция:   {self.result['area_upper_2d']:.2f} кв. ед.\n")
                f.write(f"  - 3D фактическая: {self.result['area_upper_3d']:.2f} кв. ед.\n")
                f.write(f"Площадь нижней поверхности:\n")
                f.write(f"  - 2D проекция:   {self.result['area_lower_2d']:.2f} кв. ед.\n")
                f.write(f"  - 3D фактическая: {self.result['area_lower_3d']:.2f} кв. ед.\n")
            messagebox.showinfo("Сохранение", "Результаты успешно сохранены!")

    def save_results_excel(self):
        if self.result is None:
            messagebox.showwarning("Ошибка", "Сначала выполните расчёт!")
            return
        df = pd.DataFrame({
            "Параметр": [
                "Объём насыпи",
                "Объём выемки",
                "Итоговый объём",
                "Площадь верхней поверхности (2D)",
                "Площадь верхней поверхности (3D)",
                "Площадь нижней поверхности (2D)",
                "Площадь нижней поверхности (3D)"
            ],
            "Значение": [
                f"{self.result['nasyp']:.2f}",
                f"{self.result['vyemka']:.2f}",
                f"{self.result['final_volume']:.2f}",
                f"{self.result['area_upper_2d']:.2f}",
                f"{self.result['area_upper_3d']:.2f}",
                f"{self.result['area_lower_2d']:.2f}",
                f"{self.result['area_lower_3d']:.2f}"
            ]
        })
        path = filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=[("Excel файл", "*.xlsx")])
        if path:
            df.to_excel(path, index=False)
            messagebox.showinfo("Готово", "Результаты сохранены в Excel!")


if __name__ == "__main__":
    app = App()
    app.mainloop()