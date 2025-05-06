import os
import numpy as np
from scipy.spatial import Delaunay, ConvexHull, KDTree
from shapely.geometry import Polygon, Point
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.animation import FuncAnimation

# Функция для расчета площади треугольника по формуле Герона
def triangle_area(a, b, c):
    sides = np.array([np.linalg.norm(b-a), np.linalg.norm(c-b), np.linalg.norm(a-c)])
    s = sum(sides)/2
    return np.sqrt(s*(s-sides[0])*(s-sides[1])*(s-sides[2]))

# Загрузка точек
def load_points(filename):
    try:
        if not os.path.exists(filename):
            print(f"Файл не найден: {filename}")
            return None
        data = np.loadtxt(filename)
        if data.shape[1] < 3:
            print("Нужны X,Y,Z координаты")
            return None
        print(f"Загружено {len(data)} точек из {filename}")
        return data[:, :3]
    except Exception as e:
        print(f"Ошибка при чтении файла: {str(e)}")
        return None

# Площадь проекции (2D)
def calculate_2d_projection_area(points):
    hull = ConvexHull(points[:, :2])
    return hull.volume

# Реальная площадь (3D)
def calculate_3d_surface_area(points):
    tri = Delaunay(points[:, :2])
    total_area = 0.0
    for simplex in tri.simplices:
        A, B, C = points[simplex]
        total_area += triangle_area(A, B, C)
    return total_area

# Расчёт объёма
def calculate_volume(upper_points, lower_points):
    try:
        if len(upper_points) < 4 or len(lower_points) < 4:
            raise ValueError("Минимум 4 точки в каждом файле")

        kd_lower = KDTree(lower_points[:, :2])
        tri_upper = Delaunay(upper_points[:, :2])

        hull_upper = ConvexHull(upper_points[:, :2])
        poly_upper = Polygon(hull_upper.points[hull_upper.vertices])

        total_volume_nasyp = 0.0
        total_volume_vyemka = 0.0

        for simplex in tri_upper.simplices:
            A, B, C = upper_points[simplex]
            centroid = (A[:2] + B[:2] + C[:2]) / 3

            if Polygon(poly_upper).contains(Point(*centroid)):
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
        messagebox.showerror("Ошибка", f"Ошибка расчёта: {str(e)}")
        return None

# Визуализация
def visualize_surfaces_gui(fig, upper_points, lower_points, result=None):
    fig.clear()

    ax = fig.add_subplot(111, projection='3d')
    ax.set_box_aspect((np.ptp(upper_points[:, 0]), np.ptp(upper_points[:, 1]), np.ptp(upper_points[:, 2])))

    # Цветовая шкала высот
    all_z = np.concatenate([upper_points[:, 2], lower_points[:, 2]])
    norm = plt.Normalize(vmin=np.min(all_z), vmax=np.max(all_z))
    cmap = cm.get_cmap('viridis')

    upper_colors = [cmap(norm(z)) for z in upper_points[:, 2]]
    lower_colors = [cmap(norm(z)) for z in lower_points[:, 2]]

    ax.scatter(upper_points[:, 0], upper_points[:, 1], upper_points[:, 2],
               c=upper_colors, label='Верхняя поверхность', s=5, alpha=0.8)
    ax.scatter(lower_points[:, 0], lower_points[:, 1], lower_points[:, 2],
               c=lower_colors, label='Нижняя поверхность', s=5, alpha=0.8)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    ax.set_title("Поверхности с градиентом высоты")

    # Цветовая шкала
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.6, aspect=10, pad=0.1)
    cbar.set_label('Высота (Z)')

    if result:
        nasyp = result['nasyp']
        vyemka = result['vyemka']
        final = result['final_volume']

        area_upper_2d = result['area_upper_2d']
        area_lower_2d = result['area_lower_2d']
        area_upper_3d = result['area_upper_3d']
        area_lower_3d = result['area_lower_3d']

        text_lines = [
            f'Объем насыпи:         {nasyp:.2f} куб. ед.',
            f'Объем выемки:         {vyemka:.2f} куб. ед.',
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

        props = dict(facecolor='white', edgecolor='lightgray', alpha=0.8)
        ax.text2D(0.02, 0.98, '\n'.join(text_lines), transform=ax.transAxes,
                  fontsize=9, va='top', bbox=props)

    fig.tight_layout()
    plt.draw()

    return ax


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Расчёт объёмов между поверхностями")
        self.geometry("1200x700")
        self.upper_points = None
        self.lower_points = None
        self.result = None
        self.ax = None
        self.anim = None  # Для хранения анимации

        self.create_widgets()

    def create_widgets(self):
        frame_top = ttk.Frame(self)
        frame_top.pack(pady=10, fill=tk.X)

        ttk.Button(frame_top, text="Выбрать верхнюю поверхность",
                   command=self.load_upper).pack(side=tk.LEFT, padx=5)
        ttk.Button(frame_top, text="Выбрать нижнюю поверхность",
                   command=self.load_lower).pack(side=tk.LEFT, padx=5)
        ttk.Button(frame_top, text="Рассчитать и построить график",
                   command=self.calculate_and_plot).pack(side=tk.LEFT, padx=5)
        ttk.Button(frame_top, text="Вращать график",
                   command=self.animate_plot).pack(side=tk.LEFT, padx=5)
        ttk.Button(frame_top, text="Сохранить результаты",
                   command=self.save_results).pack(side=tk.LEFT, padx=5)

        self.fig = plt.figure(figsize=(6, 5))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def load_upper(self):
        path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
        if path:
            self.upper_points = load_points(path)

    def load_lower(self):
        path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
        if path:
            self.lower_points = load_points(path)

    def calculate_and_plot(self):
        if self.upper_points is None or self.lower_points is None:
            messagebox.showwarning("Ошибка", "Не выбраны оба файла!")
            return

        self.result = calculate_volume(self.upper_points, self.lower_points)
        self.ax = visualize_surfaces_gui(self.fig, self.upper_points, self.lower_points, self.result)
        self.canvas.draw()

    def animate_plot(self):
        if self.ax is None:
            messagebox.showwarning("Ошибка", "Сначала постройте график!")
            return

        def update(frame):
            self.ax.view_init(elev=20., azim=frame % 360)
            return self.fig,

        # Создаем новую анимацию каждый раз — чтобы можно было перезапускать
        self.anim = FuncAnimation(self.fig, update, frames=360, interval=50, blit=False, repeat=False)
        self.canvas.draw()

    def save_results(self):
        if self.result is None:
            messagebox.showwarning("Ошибка", "Сначала выполните расчёт!")
            return
        path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text files", "*.txt")])
        if path:
            with open(path, 'w') as f:
                f.write("=== Результаты расчёта ===\n\n")
                f.write(f"Объём насыпи:      {self.result['nasyp']:.2f} куб. ед.\n")
                f.write(f"Объём выемки:      {self.result['vyemka']:.2f} куб. ед.\n")
                f.write(f"Итоговый объём:    {self.result['final_volume']:.2f} куб. ед.\n\n")

                f.write(f"Площадь верхней поверхности:\n")
                f.write(f"  - 2D проекция:   {self.result['area_upper_2d']:.2f} кв. ед.\n")
                f.write(f"  - 3D фактическая: {self.result['area_upper_3d']:.2f} кв. ед.\n")
                f.write(f"Площадь нижней поверхности:\n")
                f.write(f"  - 2D проекция:   {self.result['area_lower_2d']:.2f} кв. ед.\n")
                f.write(f"  - 3D фактическая: {self.result['area_lower_3d']:.2f} кв. ед.\n")

            messagebox.showinfo("Сохранение", "Результаты успешно сохранены!")


if __name__ == "__main__":
    app = App()
    app.mainloop()