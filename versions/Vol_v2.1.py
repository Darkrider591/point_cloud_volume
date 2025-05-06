import os
import numpy as np
from scipy.spatial import Delaunay, ConvexHull, KDTree
from shapely.geometry import Polygon, Point
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import matplotlib.pyplot as plt
import pyvista as pv
from pyvistaqt import BackgroundPlotter
import laspy
import pandas as pd


# --- Load points from different formats ---
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
            print("Need X,Y,Z coordinates")
            return None
        print(f"Loaded {len(data)} points from {filename}")
        return data[:, :3]
    except Exception as e:
        print(f"Error reading file: {str(e)}")
        return None


def load_points_las(filename):
    try:
        las = laspy.read(filename)
        points = np.vstack((las.x, las.y, las.z)).transpose()
        print(f"Loaded {len(points)} points from LAS file")
        return points
    except Exception as e:
        print(f"Error reading LAS file: {e}")
        return None


# --- Calculate triangle area using Heron's formula ---
def triangle_area(a, b, c):
    sides = np.array([np.linalg.norm(b - a), np.linalg.norm(c - b), np.linalg.norm(a - c)])
    s = sum(sides) / 2
    return np.sqrt(s * (s - sides[0]) * (s - sides[1]) * (s - sides[2]))


# --- 2D projection area ---
def calculate_2d_projection_area(points):
    if len(points) < 3:
        print("Not enough points for Convex Hull")
        return 0.0
    try:
        hull = ConvexHull(points[:, :2])
        return hull.volume
    except:
        print("Convex Hull could not be created")
        return 0.0


# --- Real 3D surface area ---
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
        print("Triangulation failed")
        return 0.0


# --- Volume calculation ---
def calculate_volume(upper_points, lower_points):
    try:
        if len(upper_points) < 4 or len(lower_points) < 4:
            raise ValueError("At least 4 points per file")
        kd_lower = KDTree(lower_points[:, :2])

        # Create polygon for upper convex hull
        if len(upper_points) >= 3:
            try:
                hull_upper = ConvexHull(upper_points[:, :2])
                poly_upper = Polygon(hull_upper.points[hull_upper.vertices])
            except:
                print("Upper Convex Hull could not be created")
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
        messagebox.showerror("Error", f"Calculation error: {str(e)}")
        return None


# --- Visualization with PyVista ---
def visualize_with_pyvista(upper_points, lower_points, result=None):
    try:
        plotter = BackgroundPlotter()

        def create_surface(points):
            tri = Delaunay(points[:, :2])
            surf = pv.PolyData(points)
            surf.faces = np.hstack([np.full((len(tri.simplices), 1), 3), tri.simplices]).ravel()
            return surf

        upper_cloud = pv.PolyData(upper_points)
        lower_cloud = pv.PolyData(lower_points)

        # Add point clouds with height-based color maps
        plotter.add_mesh(upper_cloud, scalars=upper_points[:, 2], cmap='viridis', point_size=5, show_scalar_bar=True, label="Top Surface")
        plotter.add_mesh(lower_cloud, scalars=lower_points[:, 2], cmap='plasma', point_size=5, show_scalar_bar=True, label="Bottom Surface")

        plotter.view_isometric()

        # Result labels
        if result:
            lines = [
                f'Volume Fill:         {result["nasyp"]:.2f} cu.m',
                f'Volume Cut:          {result["vyemka"]:.2f} cu.m',
                f'Summary volume:      {result["final_volume"]:.2f} cu.m',
                '',
                f'Surface areas:',
                f'  - Upper (2D):       {result["area_upper_2d"]:.2f}',
                f'  - Upper (3D):       {result["area_upper_3d"]:.2f}',
                f'  - Lower (2D):       {result["area_lower_2d"]:.2f}',
                f'  - Lower (3D):       {result["area_lower_3d"]:.2f}'
            ]
            plotter.add_text('\n'.join(lines), position='lower_left', font_size=12)

        plotter.add_legend([
            ('Top Surface', 'red'),
            ('Bottom Surface', 'blue')
        ])

        plotter.add_axes()
        plotter.show_grid()
        return plotter  # Return plotter to manage it externally

    except Exception as e:
        print("Visualization error:", str(e))
        return None


# --- GUI Application ---
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Surface Volume Calculator")
        self.geometry("1200x600")
        self.upper_points = None
        self.lower_points = None
        self.result = None
        self.plotter_window = None  # To store the BackgroundPlotter window
        self.create_widgets()

    def create_widgets(self):
        frame_top = ttk.Frame(self)
        frame_top.pack(pady=10, fill=tk.X)

        ttk.Button(frame_top, text="Select Top Surface",
                   command=self.load_upper).pack(side=tk.LEFT, padx=2)
        ttk.Button(frame_top, text="Select Bottom Surface",
                   command=self.load_lower).pack(side=tk.LEFT, padx=2)
        ttk.Button(frame_top, text="Calculate",
                   command=self.calculate).pack(side=tk.LEFT, padx=2)
        ttk.Button(frame_top, text="Show Chart",
                   command=self.show_plot).pack(side=tk.LEFT, padx=2)
        ttk.Button(frame_top, text="Save Results",
                   command=self.save_results).pack(side=tk.LEFT, padx=2)
        ttk.Button(frame_top, text="Export to Excel",
                   command=self.save_results_excel).pack(side=tk.LEFT, padx=2)

        # View control buttons
        frame_view = ttk.Frame(self)
        frame_view.pack(pady=5, fill=tk.X)
        ttk.Button(frame_view, text="Isometric", command=lambda: self.change_view('isometric')).pack(side=tk.LEFT, padx=2)
        ttk.Button(frame_view, text="Top View", command=lambda: self.change_view('top')).pack(side=tk.LEFT, padx=2)
        ttk.Button(frame_view, text="Front View", command=lambda: self.change_view('front')).pack(side=tk.LEFT, padx=2)
        ttk.Button(frame_view, text="Side View", command=lambda: self.change_view('side')).pack(side=tk.LEFT, padx=2)
        ttk.Button(frame_view, text="Save Screenshot", command=self.save_screenshot).pack(side=tk.LEFT, padx=2)

    def change_view(self, view):
        if not hasattr(self, 'plotter_window') or self.plotter_window is None:
            return
        if view == 'isometric':
            self.plotter_window.view_isometric()
        elif view == 'top':
            self.plotter_window.view_xy()
        elif view == 'front':
            self.plotter_window.view_xz()
        elif view == 'side':
            self.plotter_window.view_yz()
        self.plotter_window.render()

    def save_screenshot(self):
        if not hasattr(self, 'plotter_window') or self.plotter_window is None:
            messagebox.showwarning("Warning", "No chart displayed yet!")
            return
        path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG file", "*.png")])
        if path:
            self.plotter_window.screenshot(path)

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
            messagebox.showwarning("Error", "Please select both files first!")
            return
        self.result = calculate_volume(self.upper_points, self.lower_points)
        messagebox.showinfo("Done", "Calculation completed successfully!")

    def show_plot(self):
        if self.result is None:
            messagebox.showwarning("Error", "First run the calculation!")
            return
        if self.plotter_window:
            self.plotter_window.close()
        self.plotter_window = visualize_with_pyvista(self.upper_points, self.lower_points, self.result)

    def save_results(self):
        if self.result is None:
            messagebox.showwarning("Error", "First run the calculation!")
            return
        path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text files", "*.txt")])
        if path:
            with open(path, 'w') as f:
                f.write("=== Calculation Results ===\n")
                f.write(f"Fill Volume:           {self.result['nasyp']:.2f} cu.m\n")
                f.write(f"Cut Volume:            {self.result['vyemka']:.2f} cu.m\n")
                f.write(f"Net Volume:            {self.result['final_volume']:.2f} cu.m\n")
                f.write("\nSurface Areas:\n")
                f.write(f"  - Upper (2D):         {self.result['area_upper_2d']:.2f}\n")
                f.write(f"  - Upper (3D):         {self.result['area_upper_3d']:.2f}\n")
                f.write(f"  - Lower (2D):         {self.result['area_lower_2d']:.2f}\n")
                f.write(f"  - Lower (3D):         {self.result['area_lower_3d']:.2f}\n")
            messagebox.showinfo("Saved", "Results saved successfully!")

    def save_results_excel(self):
        if self.result is None:
            messagebox.showwarning("Error", "First run the calculation!")
            return
        df = pd.DataFrame({
            "Parameter": [
                "Fill Volume", "Cut Volume", "Net Volume",
                "Upper Surface Area (2D)", "Upper Surface Area (3D)",
                "Lower Surface Area (2D)", "Lower Surface Area (3D)"
            ],
            "Value": [
                f"{self.result['nasyp']:.2f}", f"{self.result['vyemka']:.2f}", f"{self.result['final_volume']:.2f}",
                f"{self.result['area_upper_2d']:.2f}", f"{self.result['area_upper_3d']:.2f}",
                f"{self.result['area_lower_2d']:.2f}", f"{self.result['area_lower_3d']:.2f}"
            ]
        })
        path = filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=[("Excel file", "*.xlsx")])
        if path:
            df.to_excel(path, index=False)
            messagebox.showinfo("Done", "Results saved to Excel!")


# --- Run application ---
if __name__ == "__main__":
    app = App()
    app.mainloop()