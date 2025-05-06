import os
import numpy as np
from scipy.spatial import Delaunay, ConvexHull, KDTree
from shapely.geometry import Polygon, Point
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import matplotlib.pyplot as plt
from pyvistaqt import BackgroundPlotter
import pyvista as pv
import laspy
import pandas as pd

# --- Loading points from different formats ---
def load_points(filename):
    ext = os.path.splitext(filename)[1].lower()
    if ext in [".txt", ".csv", ".xyz"]:
        return load_points_txt(filename)
    elif ext == ".las":
        return load_points_las(filename)
    else:
        print(f"Unknown file format: {ext}")
        return None

def load_points_txt(filename):
    try:
        data = np.loadtxt(filename)
        if data.shape[1] < 3:
            print("Needed X,Y,Z coordinates")
            return None
        print(f"Uploaded {len(data)} points from {filename}")
        return data[:, :3]
    except Exception as e:
        print(f"Error reading file: {str(e)}")
        return None

def load_points_las(filename):
    try:
        las = laspy.read(filename)
        points = np.vstack((las.x, las.y, las.z)).transpose()
        print(f"Uploaded {len(points)} points from LAS")
        return points
    except Exception as e:
        print(f"Error read LAS: {e}")
        return None

# --- Calculating the area of ​​a triangle using Heron's formula ---
def triangle_area(a, b, c):
    sides = np.array([np.linalg.norm(b - a), np.linalg.norm(c - b), np.linalg.norm(a - c)])
    s = sum(sides) / 2
    return np.sqrt(s * (s - sides[0]) * (s - sides[1]) * (s - sides[2]))

# --- Projected area (2D) ---
def calculate_2d_projection_area(points):
    if len(points) < 3:
        print("Not enough points for Convex Hull")
        return 0.0
    try:
        hull = ConvexHull(points[:, :2])
        return hull.volume
    except:
        print("Convex Hull cannot be created")
        return 0.0

# --- Real surface area (3D) ---
def calculate_3d_surface_area(points):
    if len(points) < 3:
        print("Not enough points for triangulation")
        return 0.0
    try:
        tri = Delaunay(points[:, :2])
        total_area = 0.0
        for simplex in tri.simplices:
            A, B, C = points[simplex]
            total_area += triangle_area(A, B, C)
        return total_area
    except:
        print("Triangulation not completed")
        return 0.0

# --- Calculating the volume between surfaces ---
def calculate_volume(upper_points, lower_points):
    try:
        if len(upper_points) < 4 or len(lower_points) < 4:
            raise ValueError("Minimum 4 dots in each file")

        # Let's calculate 2D projections of areas
        area_upper_2d = calculate_2d_projection_area(upper_points)
        area_lower_2d = calculate_2d_projection_area(lower_points)

        # Choose the minimum area
        use_upper = area_upper_2d <= area_lower_2d

        kd_lower = KDTree(lower_points[:, :2])

        if use_upper:
            points_to_use = upper_points
            is_upper = True
        else:
            points_to_use = lower_points
            is_upper = False

        if len(points_to_use) >= 3:
            try:
                hull = ConvexHull(points_to_use[:, :2])
                poly = Polygon(hull.points[hull.vertices])
            except:
                print("Failed to create Convex Hull")
                poly = None
        else:
            poly = None

        total_volume_nasyp = 0.0
        total_volume_vyemka = 0.0

        tri = Delaunay(points_to_use[:, :2])

        for simplex in tri.simplices:
            A, B, C = points_to_use[simplex]
            centroid = (A[:2] + B[:2] + C[:2]) / 3

            if poly is not None and not poly.contains(Point(*centroid)):
                continue

            area = triangle_area(A[:2], B[:2], C[:2])

            z_ref = (A[2] + B[2] + C[2]) / 3
            dists, idxs = kd_lower.query(centroid, k=3)
            weights = 1 / (dists + 1e-8)
            z_other = np.average(lower_points[idxs, 2], weights=weights)

            height_diff = z_ref - z_other

            if height_diff >= 0:
                total_volume_nasyp += area * height_diff
            else:
                total_volume_vyemka -= area * height_diff

        final_volume = total_volume_nasyp - total_volume_vyemka

        return {
            'nasyp': total_volume_nasyp,
            'vyemka': total_volume_vyemka,
            'final_volume': final_volume,
            'area_upper_2d': area_upper_2d,
            'area_lower_2d': area_lower_2d,
            'area_upper_3d': calculate_3d_surface_area(upper_points),
            'area_lower_3d': calculate_3d_surface_area(lower_points),
            'used_surface': 'Upper' if use_upper else 'Lower'
        }

    except Exception as e:
        messagebox.showerror("Error", f"Error calc: {str(e)}")
        return None

# --- Visualization of a point cloud ---
def visualize_point_cloud(upper_points, lower_points, result=None):
    plotter = BackgroundPlotter()
    plotter.set_background('white')
    upper_cloud = pv.PolyData(upper_points)
    lower_cloud = pv.PolyData(lower_points)

    def get_colors(points):
        z_values = points[:, 2]
        norm_z = (z_values - np.min(z_values)) / (np.max(z_values) - np.min(z_values))
        cmap = plt.get_cmap('viridis')
        return cmap(norm_z)

    upper_colors = (get_colors(upper_points)[:, :3] * 255).astype(int)
    lower_colors = (get_colors(lower_points)[:, :3] * 255).astype(int)

    plotter.add_mesh(upper_cloud, point_size=4, scalars=upper_colors, rgb=True, label="Upper surface")
    plotter.add_mesh(lower_cloud, point_size=4, scalars=lower_colors, rgb=True, label="Bottom surface")
    plotter.view_isometric()
    plotter.add_axes()
    plotter.show_grid()

    if result:
        lines = [
            f'Volume fill:         {result["nasyp"]:.2f} cubic units',
            f'Volume cut:         {result["vyemka"]:.2f} cubic units',
            f'Summary volume:       {result["final_volume"]:.2f} cubic units',
            '',
            f'Used surface:          {result["used_surface"]}',
            '',
            f'Upper surface area:',
            f'  - 2D projection:      {result["area_upper_2d"]:.2f}',
            f'  - 3D actual:   {result["area_upper_3d"]:.2f}',
            f'Bottom surface area:',
            f'  - 2D projection:      {result["area_lower_2d"]:.2f}',
            f'  - 3D actual:   {result["area_lower_3d"]:.2f}'
        ]
        plotter.add_text('\n'.join(lines), position='upper_left', font_size=8)

    plotter.add_legend([
        ('Upper surface', 'red'),
        ('Bottom surface', 'blue')
    ])

# --- Visualization of triangulated surfaces ---
def visualize_triangular_surfaces(upper_points, lower_points, result=None):
    plotter = BackgroundPlotter()
    plotter.set_background('white')

    def create_surface(points):
        tri = Delaunay(points[:, :2])
        surf = pv.PolyData(points)
        surf.faces = np.hstack([np.full((len(tri.simplices), 1), 3), tri.simplices]).ravel()
        return surf

    upper_surf = create_surface(upper_points)
    lower_surf = create_surface(lower_points)

    plotter.add_mesh(upper_surf, color='red', opacity=0.5, label="Upper surface")
    plotter.add_mesh(lower_surf, color='blue', opacity=0.5, label="Bottom surface")
    plotter.view_isometric()
    plotter.add_axes()
    plotter.show_grid()

    if result:
        lines = [
            f'Volume fill:         {result["nasyp"]:.2f} cubic units',
            f'Volume cut:         {result["vyemka"]:.2f} cubic units',
            f'Summary volume:       {result["final_volume"]:.2f} cubic units',
            '',
            f'Used surface:          {result["used_surface"]}',
            '',
            f'Upper surface area:',
            f'  - 2D projection:      {result["area_upper_2d"]:.2f}',
            f'  - 3D actual:   {result["area_upper_3d"]:.2f}',
            f'Bottom surface area:',
            f'  - 2D projection:      {result["area_lower_2d"]:.2f}',
            f'  - 3D actual:   {result["area_lower_3d"]:.2f}'
        ]
        plotter.add_text('\n'.join(lines), position='upper_left', font_size=8)

    plotter.add_legend([
        ('Upper surface', 'red'),
        ('Bottom surface', 'blue')
    ])

# --- GUI ---
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Calculation of volumes between surfaces")
        self.geometry("800x200")
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
        ttk.Button(frame_top, text="Show points",
                   command=self.show_point_cloud).pack(side=tk.LEFT, padx=5)
        ttk.Button(frame_top, text="Show surfaces",
                   command=self.show_triangular_surfaces).pack(side=tk.LEFT, padx=5)
        ttk.Button(frame_top, text="Save results",
                   command=self.save_results).pack(side=tk.LEFT, padx=5)
        ttk.Button(frame_top, text="Export to Excel",
                   command=self.save_results_excel).pack(side=tk.LEFT, padx=5)

    def load_upper(self):
        path = filedialog.askopenfilename(filetypes=[("Supported formats", "*.txt *.csv *.xyz *.las")])
        if path:
            self.upper_points = load_points(path)

    def load_lower(self):
        path = filedialog.askopenfilename(filetypes=[("Supported formats", "*.txt *.csv *.xyz *.las")])
        if path:
            self.lower_points = load_points(path)

    def calculate(self):
        if self.upper_points is None or self.lower_points is None:
            messagebox.showwarning("Error", "Both files are not selected!")
            return
        self.result = calculate_volume(self.upper_points, self.lower_points)
        messagebox.showinfo("Ready", "Calculation completed successfully!")

    def show_point_cloud(self):
        if self.result is None:
            messagebox.showwarning("Error", "First, do the calculation!")
            return
        visualize_point_cloud(self.upper_points, self.lower_points, self.result)

    def show_triangular_surfaces(self):
        if self.result is None:
            messagebox.showwarning("Error", "First, do the calculation!")
            return
        visualize_triangular_surfaces(self.upper_points, self.lower_points, self.result)

    def save_results(self):
        if self.result is None:
            messagebox.showwarning("Error", "First, do the calculation!")
            return
        path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text files", "*.txt")])
        if path:
            with open(path, 'w') as f:
                f.write("=== Calculation results ===\n")
                f.write(f"Volume fill:         {self.result['nasyp']:.2f} cubic units\n")
                f.write(f"Volume cut:         {self.result['vyemka']:.2f} cubic units\n")
                f.write(f"Summary volume:       {self.result['final_volume']:.2f} cubic units\n")
                f.write(f"Used surface:         {self.result['used_surface']}\n")
                f.write("\nUpper surface area:\n")
                f.write(f"  - 2D projection:   {self.result['area_upper_2d']:.2f} cubic units\n")
                f.write(f"  - 3D actual:       {self.result['area_upper_3d']:.2f} cubic units\n")
                f.write("Bottom surface area:\n")
                f.write(f"  - 2D projection:   {self.result['area_lower_2d']:.2f} cubic units\n")
                f.write(f"  - 3D actual:       {self.result['area_lower_3d']:.2f} cubic units\n")
            messagebox.showinfo("Savings", "Results saved successfully!")

    def save_results_excel(self):
        if self.result is None:
            messagebox.showwarning("Error", "First, do the calculation!")
            return
        df = pd.DataFrame({
            "Parametr": [
                "Volume fill",
                "Volume cut",
                "Summary volume",
                "Used surface",
                "Upper surface area (2D)",
                "Upper surface area (3D)",
                "Bottom surface area (2D)",
                "Bottom surface area (3D)"
            ],
            "Meaning": [
                f"{self.result['nasyp']:.2f}",
                f"{self.result['vyemka']:.2f}",
                f"{self.result['final_volume']:.2f}",
                f"{self.result['used_surface']}",
                f"{self.result['area_upper_2d']:.2f}",
                f"{self.result['area_upper_3d']:.2f}",
                f"{self.result['area_lower_2d']:.2f}",
                f"{self.result['area_lower_3d']:.2f}"
            ]
        })
        path = filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=[("Excel file", "*.xlsx")])
        if path:
            df.to_excel(path, index=False)
            messagebox.showinfo("Ready", "Results saved to Excel!")

if __name__ == "__main__":
    app = App()
    app.mainloop()