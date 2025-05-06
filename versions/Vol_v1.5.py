import os
import numpy as np
from scipy.spatial import Delaunay, ConvexHull, KDTree
from shapely.geometry import Polygon, Point
import tkinter as tk
from tkinter import filedialog, messagebox
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Для 3D-графиков

# Функция для расчета площади треугольника по формуле Герона
def triangle_area(a, b, c):
    sides = np.array([np.linalg.norm(b-a), np.linalg.norm(c-b), np.linalg.norm(a-c)])
    s = sum(sides)/2
    return np.sqrt(s*(s-sides[0])*(s-sides[1])*(s-sides[2]))

def load_points(filename):
    """Загрузка точек из файла с проверками"""
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
    """
    Конструируем полигон из вершин выпуклой оболочки
    """
    polygon = Polygon(hull_vertices)
    return polygon

def point_inside_polygon(point, polygon):
    """
    Проверяет, лежит ли точка внутри полигона
    """
    pnt = Point(*point[:2])
    return polygon.contains(pnt)

def visualize_surfaces(upper_points, lower_points):
    """
    График 3D-визуализации поверхности с правильным соотношением осей
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Устанавливаем равные пропорции осей
    ax.set_box_aspect((np.ptp(upper_points[:, 0]), np.ptp(upper_points[:, 1]), np.ptp(upper_points[:, 2])))
    
    # Верхняя поверхность
    ax.scatter(upper_points[:, 0], upper_points[:, 1], upper_points[:, 2],
               color='blue', label='Верхняя поверхность', marker='o')
    
    # Нижняя поверхность
    ax.scatter(lower_points[:, 0], lower_points[:, 1], lower_points[:, 2],
               color='green', label='Нижняя поверхность', marker='^')
    
    # Оформление
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.legend()
    plt.title("Визуализация поверхностных слоёв")
    plt.show()

def calculate_volume(upper_points, lower_points):
    """Расчет объема между поверхностями с учетом пересечений и ограничений по площади"""
    try:
        # Проверка минимального количества точек
        if len(upper_points) < 4 or len(lower_points) < 4:
            print("Нужно минимум 4 точки в каждом файле для триангуляции")
            return None
        
        # Создаем KD-дерево для быстрой близости точек
        kd_lower = KDTree(lower_points[:, :2])
        
        # Триангулируем верхнюю поверхность
        tri_upper = Delaunay(upper_points[:, :2])
        
        # Выпуклая оболочка верхней поверхности
        hull_upper = ConvexHull(upper_points[:, :2])
        poly_upper = create_polygon_from_hull(hull_upper.points[hull_upper.vertices])
        
        # Переменные для накопления объемов
        total_volume_nasyp = 0.0
        total_volume_vyemka = 0.0
        
        # Обработаем все треугольники верхней поверхности
        for simplex in tri_upper.simplices:
            A, B, C = upper_points[simplex]
            
            # Только если центр тяжести треугольника находится внутри полигона верхней поверхности
            centroid = ((A[:2]+B[:2]+C[:2])/3)
            if point_inside_polygon(centroid, poly_upper):
                # Площадь треугольника
                area = triangle_area(A[:2], B[:2], C[:2])
                
                # Средняя высота верхнего треугольника
                z_upper = (A[2] + B[2] + C[2]) / 3
                
                # Найдем соответствующие точки на нижней поверхности
                _, idx_A = kd_lower.query(A[:2])
                _, idx_B = kd_lower.query(B[:2])
                _, idx_C = kd_lower.query(C[:2])
                
                z_lower = (lower_points[idx_A, 2] + 
                           lower_points[idx_B, 2] + 
                           lower_points[idx_C, 2]) / 3
                
                # Отличия по высоте
                height_diff = z_upper - z_lower
                
                if height_diff >= 0:
                    total_volume_nasyp += area * height_diff  # Объём насыпи
                else:
                    total_volume_vyemka -= area * height_diff  # Объём выемки
        
        # Определяем преобладание объёма насыпи или выемки
        final_volume = total_volume_nasyp - total_volume_vyemka
        
        return {
            'nasyp': total_volume_nasyp,
            'vyemka': total_volume_vyemka,
            'final_volume': final_volume
        }
        
    except Exception as e:
        print(f"\nОшибка при расчете: {str(e)}")
        return None

def select_files():
    root = tk.Tk()
    root.withdraw()  # Скрываем основное окно Tkinter
    
    filetypes = (
        ("Text files", "*.txt"),
        ("All files", "*.*")
    )
    
    upper_file_path = filedialog.askopenfilename(title="Выберите файл верхней поверхности",
                                                filetypes=filetypes)
    lower_file_path = filedialog.askopenfilename(title="Выберите файл нижней поверхности",
                                                filetypes=filetypes)
    
    return upper_file_path, lower_file_path

def show_results(result):
    """
    Показываем результат в удобочитаемом сообщении
    """
    msg = f"Объем насыпи: {result['nasyp']:.2f} куб. единиц\n"
    msg += f"Объем выемки: {result['vyemka']:.2f} куб. единиц\n"
    
    final_volume = result['final_volume']
    if final_volume >= 0:
        msg += f"Итоговый объем (насыпь): {final_volume:.2f} куб. единиц"
    else:
        msg += f"Итоговый объем (выемка): {-final_volume:.2f} куб. единиц"
    
    messagebox.showinfo("Результаты расчета", msg)

def main():
    """Основная функция"""
    print("=== Точный расчет объема между поверхностями ===")
    
    # Пользователь выбирает файлы
    upper_file, lower_file = select_files()
    
    if not upper_file or not lower_file:
        print("\nНеобходимо выбрать оба файла!")
        return
    
    print(f"\nВерхняя поверхность: {upper_file}")
    print(f"Нижняя поверхность: {lower_file}")
    
    # Загрузка данных
    upper_points = load_points(upper_file)
    lower_points = load_points(lower_file)
    
    if upper_points is None or lower_points is None:
        print("\nОстановка из-за проблем с данными")
        return
    
    # Расчет объема
    result = calculate_volume(upper_points, lower_points)
    
    if result is not None:
        show_results(result)  # Сообщение с результатами
        visualize_surfaces(upper_points, lower_points)  # Визуализируем поверхности
    else:
        print("\nНе удалось рассчитать объем")

if __name__ == "__main__":
    main()