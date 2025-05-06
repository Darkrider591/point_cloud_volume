import os
import numpy as np
from scipy.spatial import Delaunay, ConvexHull, KDTree
from scipy.interpolate import LinearNDInterpolator

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

def calculate_volume(upper_points, lower_points):
    """Расчет объема между поверхностями"""
    try:
        # Проверка минимального количества точек
        if len(upper_points) < 4 or len(lower_points) < 4:
            print("Нужно минимум 4 точки в каждом файле для триангуляции")
            return None
        
        # Создаем KD-деревья для быстрого поиска ближайших точек
        kd_upper = KDTree(upper_points[:, :2])
        kd_lower = KDTree(lower_points[:, :2])
        
        # Триангулируем поверхности
        tri_upper = Delaunay(upper_points[:, :2])
        tri_lower = Delaunay(lower_points[:, :2])
        
        # Вычисляем ПЛОЩАДИ поверхностей (используя триангуляцию)
        def compute_area(points, triangles):
            areas = []
            for t in triangles:
                a, b, c = points[t]
                areas.append(triangle_area(a[:2], b[:2], c[:2]))  
            return sum(areas)
        
        area_upper = compute_area(upper_points, tri_upper.simplices)
        area_lower = compute_area(lower_points, tri_lower.simplices)
        
        print(f"\nПлощадь верхней поверхности: {area_upper:.2f}")
        print(f"Площадь нижней поверхности: {area_lower:.2f}")
        
        # Выбираем основную поверхность (с большей площадью)
        if area_upper < area_lower:
            main_tri = tri_upper
            main_points = upper_points
            other_points = lower_points
            other_kd = kd_lower
            print("\nИспользуем ВЕРХНЮЮ поверхность как основу для ограничения площади")
        else:
            main_tri = tri_lower
            main_points = lower_points
            other_points = upper_points
            other_kd = kd_upper
            print("\nИспользуем НИЖНЮЮ поверхность как основу")
        
        total_volume = 0.0
        
        # Обрабатываем каждый треугольник основной поверхности
        for simplex in main_tri.simplices:
            A, B, C = main_points[simplex]
            
            # Площадь треугольника
            area = triangle_area(A[:2], B[:2], C[:2])
            
            # Средняя высота основной поверхности
            z_main = (A[2] + B[2] + C[2]) / 3
            
            # Находим ближайшие точки на другой поверхности
            _, idx_A = other_kd.query(A[:2])
            _, idx_B = other_kd.query(B[:2])
            _, idx_C = other_kd.query(C[:2])
            
            z_other = (other_points[idx_A, 2] +
                       other_points[idx_B, 2] +
                       other_points[idx_C, 2]) / 3
            
            # Разница высот с учетом какая поверхность верхняя
            height_diff = abs(z_main - z_other)
            
            total_volume += area * height_diff
        
        return total_volume
        
    except Exception as e:
        print(f"\nОшибка при расчете: {str(e)}")
        return None

def main():
    """Основная функция"""
    print("=== Точный расчет объема между поверхностями ===")
    
    # Указать путь к файлам
    base_dir = r"D:\DigTP_Data\17_Radar\Vol\Calc_Python\Radar"
    upper_file = os.path.join(base_dir, "upper_surface.txt")
    lower_file = os.path.join(base_dir, "lower_surface.txt")
    
    print(f"\nВерхняя поверхность: {upper_file}")
    print(f"Нижняя поверхность: {lower_file}")
    
    # Загрузка данных
    upper_points = load_points(upper_file)
    lower_points = load_points(lower_file)
    
    if upper_points is None or lower_points is None:
        print("\nОстановка из-за проблем с данными")
        return
    
    # Расчет объема
    volume = calculate_volume(upper_points, lower_points)
    
    if volume is not None:
        print(f"\nРЕЗУЛЬТАТ: Объем = {volume:.2f} куб. единиц")
    else:
        print("\nНе удалось рассчитать объем")

if __name__ == "__main__":
    main()