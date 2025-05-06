import os
import numpy as np
from scipy.spatial import Delaunay, ConvexHull, KDTree
from shapely.geometry import Polygon, Point

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

def main():
    """Основная функция"""
    print("=== Расчет объема между поверхностями методом триангуляции ===")
    
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
    result = calculate_volume(upper_points, lower_points)
    
    if result is not None:
        print(f"\nОбъем насыпи: {result['nasyp']:.2f} куб. единиц")
        print(f"Объем выемки: {result['vyemka']:.2f} куб. единиц")
        
        # Итоговый объём
        final_volume = result['final_volume']
        if final_volume >= 0:
            print(f"Итоговый объем (насыпь): {final_volume:.2f} куб. единиц")
        else:
            print(f"Итоговый объем (выемка): {-final_volume:.2f} куб. единиц")  # Отрицательное значение показывает выемку
    else:
        print("\nНе удалось рассчитать объем")

if __name__ == "__main__":
    main()