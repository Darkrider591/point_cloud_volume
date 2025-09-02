English README

Surface Volume Calculator

Surface Volume Calculator is a tool designed to compute the volume between two surfaces defined by sets of 3D points — an upper and a lower surface. The boundary of each surface is determined based on the minimum convex hull (or minimal surface area, depending on configuration), ensuring accurate spatial enclosure.

This application calculates and displays:

- Volume between the two surfaces
  
- 2D Projected Area (area projected onto the XY plane)
  
- 3D Surface Area of both upper and lower surfaces
  
Features
Input: Two sets of 3D point clouds (upper and lower surfaces)
Automatic boundary detection using convex hull or alpha shapes
Calculation of enclosed volume
2D footprint area (projection on horizontal plane)
Total 3D surface area of both surfaces
Visual representation of surfaces and volume (if GUI is included)
Usage
Prepare your point data in a supported format (e.g., CSV, TXT) with columns: X, Y, Z
Load upper and lower surface point files
Run the calculation
View results: volume, 2D area, and 3D surface areas
Input Format Example (txt, las)

X,Y,Z
0.0,0.0,10.0
1.0,0.0,10.5
0.0,1.0,10.2
...


Русский README

Калькулятор объема между поверхностями

Калькулятор объема между поверхностями — это инструмент для вычисления объема между двумя поверхностями, заданными наборами точек в трехмерном пространстве: верхней и нижней. Граница каждой поверхности определяется по минимальной охватывающей области (например, выпуклая оболочка или альфа-форма), что обеспечивает корректное выделение границы поверхности.

Программа рассчитывает и отображает:

- Объем между двумя поверхностями
  
- Площадь в 2D (проекция на горизонтальную плоскость XY)
  
- Площадь в 3D (реальная площадь поверхности с учетом рельефа)
  
Возможности
Ввод: два набора точек (облака точек) для верхней и нижней поверхностей
Автоматическое определение границы поверхности (по выпуклой оболочке или минимальной площади)
Вычисление объема между поверхностями
Расчет площади проекции на плоскость (2D)
Вычисление фактической площади поверхности (3D)
Визуализация поверхностей и объема (при наличии графического интерфейса)
Использование
Подготовьте данные в поддерживаемом формате (например, CSV или TXT) с колонками: X, Y, Z
Загрузите файлы с точками для верхней и нижней поверхности
Запустите вычисление
Получите результаты: объем, 2D площадь и 3D площади поверхностей
