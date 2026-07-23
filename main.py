import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, colorchooser, Toplevel, scrolledtext, simpledialog, ttk
from scipy.ndimage import gaussian_filter, sobel, prewitt, laplace
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
import time
import plotly.graph_objects as go
import webbrowser
import cv2
from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage.filters import scharr, threshold_triangle, threshold_yen
import os
from matplotlib.patches import Rectangle
import pickle
import json
from math import radians, sin, cos, sqrt, atan2

# Splash screen
def splash_screen():
    splash_root = tk.Tk()
    splash_root.overrideredirect(True)
    splash_root.geometry("400x400+800+300")
    splash_image = Image.open("mh.png")
    splash_image = splash_image.resize((300, 300), Image.LANCZOS)
    splash_photo = ImageTk.PhotoImage(splash_image)
    splash_label = tk.Label(splash_root, image=splash_photo)
    splash_label.pack()
    footer_label = tk.Label(splash_root, text="GPR VORTEX", font=('Bzar', 15, 'bold'), fg='#fd1356')
    footer_label.pack(pady=10)
    splash_root.update()
    time.sleep(2)
    splash_root.destroy()

# File reading functions
def read_iprh_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            header_info = {}
            for line in file:
                if ':' in line:
                    key, value = line.split(':', 1)
                    header_info[key.strip()] = value.strip()
            return header_info
    except FileNotFoundError:
        messagebox.showerror("Error", f"File {file_path} not found.")
        return None

def read_iprb_file(file_path, num_traces, num_samples):
    try:
        data = np.fromfile(file_path, dtype=np.float32)
        expected_size = num_traces * num_samples
        actual_size = data.size
        if actual_size < expected_size:
            data = np.pad(data, (0, expected_size - actual_size), 'constant')
        elif actual_size > expected_size:
            data = data[:expected_size]
            messagebox.showwarning("Warning", f"Data size {actual_size} was greater than expected size {expected_size}, truncating data.")
        return data.reshape((num_traces, num_samples))
    except Exception as e:
        messagebox.showerror("Error", f"Error reading file: {e}")
        return None

# Filtering functions
def background_subtraction(data, window_size):
    avg_background = np.zeros_like(data)
    for i in range(data.shape[0]):
        start = max(0, i - window_size // 2)
        end = min(data.shape[0], i + window_size // 2 + 1)
        avg_background[i] = np.mean(data[start:end], axis=0)
    return data - avg_background

def gaussian_smoothing(data, sigma):
    return gaussian_filter(data, sigma=sigma)

def flatten_to_horizon(data, horizon_level):
    flattened_data = data.copy()
    for i in range(data.shape[0]):
        shift = int(horizon_level - np.argmax(data[i]))
        flattened_data[i] = np.roll(data[i], shift=shift)
    return flattened_data

def sobel_transform(data):
    return sobel(data)

def absolute_value(data):
    return np.abs(data)

def remove_air_wave_reflections(data, threshold):
    data[data < threshold] = 0
    return data

def prewitt_transform(data):
    return prewitt(data)

def laplace_transform(data):
    return laplace(data)

def binomial_filter(data, window_size=5):
    weights = np.array([1, 4, 6, 4, 1])[:window_size]
    weights = weights / np.sum(weights)
    filtered_data = np.zeros_like(data)
    for i in range(data.shape[0]):
        filtered_data[i] = np.convolve(data[i], weights, mode='same')
    return filtered_data

def moving_average(data, window_size=5):
    filtered_data = np.zeros_like(data)
    for i in range(data.shape[0]):
        filtered_data[i] = np.convolve(data[i], np.ones(window_size)/window_size, mode='same')
    return filtered_data

def max_filter(data, window_size=3):
    filtered_data = np.zeros_like(data)
    half_win = window_size // 2
    for i in range(data.shape[0]):
        for j in range(half_win, data.shape[1] - half_win):
            window = data[i, j-half_win:j+half_win+1]
            filtered_data[i, j] = np.max(window)
    return filtered_data

def min_filter(data, window_size=3):
    filtered_data = np.zeros_like(data)
    half_win = window_size // 2
    for i in range(data.shape[0]):
        for j in range(half_win, data.shape[1] - half_win):
            window = data[i, j-half_win:j+half_win+1]
            filtered_data[i, j] = np.min(window)
    return filtered_data

def adaptive_thresholding(data, block_size=11, C=2):
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 10))
    scaled_data = scaler.fit_transform(data)
    thresholded = np.zeros_like(data)
    if block_size % 2 == 0: block_size += 1
    if block_size <= 1: block_size = 3
    for i in range(data.shape[0]):
        trace = scaled_data[i].astype(np.uint8)
        thresholded_trace = cv2.adaptiveThreshold(trace, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, C)
        thresholded[i] = thresholded_trace.flatten()
    return thresholded

def local_entropy_thresholding(data, window_size=3):
    data_uint8 = cv2.normalize(data, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    ent = entropy(data_uint8, disk(window_size))
    threshold = np.mean(ent)
    return (ent > threshold).astype(np.float32)

def apply_filters(data, filters):
    for filter_type, params in filters:
        if filter_type == 'Background Subtraction':
            data = background_subtraction(data, window_size=params['window_size'])
        elif filter_type == 'Gaussian':
            data = gaussian_smoothing(data, sigma=params['sigma'])
        elif filter_type == 'Sobel Transform':
            data = sobel_transform(data)
        elif filter_type == 'Absolute Value':
            data = absolute_value(data)
        elif filter_type == 'Remove Air-wave Reflections':
            data = remove_air_wave_reflections(data, threshold=params['threshold'])
        elif filter_type == 'Prewitt':
            data = prewitt_transform(data)
        elif filter_type == 'Laplace':
            data = laplace_transform(data)
        elif filter_type == 'Shift':
            data = flatten_to_horizon(data, horizon_level=params['Shift_level'])
        elif filter_type == 'Binomial Filter':
            data = binomial_filter(data, window_size=params['Binomial_size'])
        elif filter_type == 'Moving Average':
            data = moving_average(data, window_size=params['Moving_size'])
        elif filter_type == 'Max Filter':
            data = max_filter(data, window_size=params['Max_size'])
        elif filter_type == 'Min Filter':
            data = min_filter(data, window_size=params['Min_size'])
        elif filter_type == 'Adaptive Thresholding':
            data = adaptive_thresholding(data, block_size=params['block_size'], C=params['C'])
        elif filter_type == 'Local Entropy Thresholding':
            data = local_entropy_thresholding(data, window_size=params['Entropy_window_size'])
        elif filter_type == 'Scharr Transform':
            data = scharr(data)
    
    return data

class GridWindow:
    def __init__(self, parent, distance_interval, time_interval, s_v, profile_name, radar_data=None):
        self.parent = parent
        self.distance_interval = distance_interval
        self.time_interval = time_interval
        self.s_v = s_v
        self.profile_name = profile_name
        self.radar_data = radar_data
        
        # Grid attributes
        self.grid_data = None
        self.grid_rows = 0
        self.grid_cols = 0
        self.grid_cell_size = 1.0
        self.grid_active = False
        self.fill_mode = False
        
        # Marker properties
        self.marker_shape = 'o'
        self.marker_color = 'red'
        self.marker_size = 8
        
        # Create the window
        self.window = Toplevel(parent)
        self.window.title(f"Grid Tool - {profile_name}")
        self.window.geometry("1000x700")
        
        # Create widgets
        self.create_widgets()
        
    def create_widgets(self):
        # Main container
        main_frame = tk.Frame(self.window)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel - Plot
        plot_frame = tk.Frame(main_frame, bg="#ffffff", bd=2, relief=tk.GROOVE)
        plot_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # Create matplotlib figure
        self.figure, self.ax = plt.subplots(figsize=(8, 6))
        self.canvas = FigureCanvasTkAgg(self.figure, master=plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Right panel - Controls
        control_frame = tk.Frame(main_frame, bg="#f9f9f9", bd=2, relief=tk.GROOVE, width=300)
        control_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        control_frame.pack_propagate(False)
        
        # Profile info
        info_frame = tk.LabelFrame(control_frame, text="Profile Information", padx=10, pady=10)
        info_frame.pack(fill=tk.X, padx=10, pady=10)
        
        tk.Label(info_frame, text=f"Name: {self.profile_name}", font=('Helvetica', 10, 'bold')).pack(anchor=tk.W)
        tk.Label(info_frame, text=f"Distance: {self.distance_interval:.2f} m", font=('Helvetica', 10)).pack(anchor=tk.W)
        tk.Label(info_frame, text=f"Time Window: {self.time_interval:.2f} ns", font=('Helvetica', 10)).pack(anchor=tk.W)
        tk.Label(info_frame, text=f"Soil Velocity: {self.s_v*1000:.2f} m/μs", font=('Helvetica', 10)).pack(anchor=tk.W)
        
        # Calculate max depth
        self.max_depth = (self.s_v * self.time_interval) / 2
        tk.Label(info_frame, text=f"Max Depth: {self.max_depth:.2f} m", font=('Helvetica', 10)).pack(anchor=tk.W)
        
        # Grid controls
        grid_frame = tk.LabelFrame(control_frame, text="Grid Controls", padx=10, pady=10)
        grid_frame.pack(fill=tk.X, padx=10, pady=10)
        
        tk.Label(grid_frame, text="Cell Size (m):", font=('Helvetica', 10, 'bold')).grid(row=0, column=0, sticky=tk.W, pady=5)
        self.cell_size_var = tk.DoubleVar(value=1.0)
        cell_size_spin = tk.Spinbox(grid_frame, from_=0.1, to=5.0, increment=0.1, textvariable=self.cell_size_var, width=10)
        cell_size_spin.grid(row=0, column=1, padx=5)
        
        create_grid_button = tk.Button(grid_frame, text="Create Grid", command=self.create_grid, bg="#4CAF50", fg="white")
        create_grid_button.grid(row=1, column=0, columnspan=2, pady=10, sticky=tk.EW)
        
        # Radar profile toggle
        self.show_radar_var = tk.BooleanVar(value=False)
        self.show_radar_checkbox = tk.Checkbutton(grid_frame, text="Show Radar Profile", 
                                                 variable=self.show_radar_var,
                                                 command=self.update_plot)
        self.show_radar_checkbox.grid(row=2, column=0, columnspan=2, pady=5, sticky=tk.W)
        
        self.fill_mode_button = tk.Button(grid_frame, text="Enable Fill Mode", command=self.toggle_fill_mode, bg="#2196F3", fg="white")
        self.fill_mode_button.grid(row=3, column=0, columnspan=2, pady=5, sticky=tk.EW)
        
        clear_grid_button = tk.Button(grid_frame, text="Clear Grid", command=self.clear_grid, bg="#f44336", fg="white")
        clear_grid_button.grid(row=4, column=0, columnspan=2, pady=5, sticky=tk.EW)
        
        # Marker properties
        marker_frame = tk.LabelFrame(control_frame, text="Marker Properties", padx=10, pady=10)
        marker_frame.pack(fill=tk.X, padx=10, pady=10)
        
        tk.Label(marker_frame, text="Shape:", font=('Helvetica', 10, 'bold')).grid(row=0, column=0, sticky=tk.W, pady=5)
        self.shape_var = tk.StringVar(value='o')
        shape_menu = ttk.Combobox(marker_frame, textvariable=self.shape_var, 
                                 values=['o', 's', '^', 'D', '*', 'p', 'h'], width=8)
        shape_menu.grid(row=0, column=1, padx=5)
        shape_menu.bind('<<ComboboxSelected>>', self.update_marker_properties)
        
        tk.Label(marker_frame, text="Color:", font=('Helvetica', 10, 'bold')).grid(row=1, column=0, sticky=tk.W, pady=5)
        self.color_button = tk.Button(marker_frame, text="    ", bg=self.marker_color,
                                     command=self.choose_color, width=8)
        self.color_button.grid(row=1, column=1, padx=5)
        
        tk.Label(marker_frame, text="Size:", font=('Helvetica', 10, 'bold')).grid(row=2, column=0, sticky=tk.W, pady=5)
        self.size_var = tk.IntVar(value=8)
        size_spin = tk.Spinbox(marker_frame, from_=4, to=20, textvariable=self.size_var, width=8)
        size_spin.grid(row=2, column=1, padx=5)
        size_spin.bind('<ButtonRelease-1>', self.update_marker_properties)
        
        # File operations
        file_frame = tk.LabelFrame(control_frame, text="File Operations", padx=10, pady=10)
        file_frame.pack(fill=tk.X, padx=10, pady=10)
        
        save_button = tk.Button(file_frame, text="Save Grid Data", command=self.save_grid, bg="#FF9800", fg="white")
        save_button.pack(fill=tk.X, pady=5)
        
        save_image_button = tk.Button(file_frame, text="Save Plot as Image", command=self.save_plot_as_image, bg="#4CAF50", fg="white")
        save_image_button.pack(fill=tk.X, pady=5)
        
        load_button = tk.Button(file_frame, text="Load Grid", command=self.load_grid, bg="#9C27B0", fg="white")
        load_button.pack(fill=tk.X, pady=5)
        
        # Status
        self.status_var = tk.StringVar(value="Ready")
        status_label = tk.Label(control_frame, textvariable=self.status_var, bd=1, relief=tk.SUNKEN, anchor=tk.W)
        status_label.pack(fill=tk.X, side=tk.BOTTOM, padx=10, pady=10)
        
        # Connect mouse events
        self.canvas.mpl_connect('button_press_event', self.on_canvas_click)
        
        # Initial plot
        self.plot_grid_only()
        
    def update_plot(self):
        self.plot_grid_only()
        
    def plot_grid_only(self):
        self.ax.clear()
        
        self.ax.set_xlim(0, self.distance_interval)
        self.ax.set_ylim(self.max_depth, 0)
        self.ax.set_xlabel("Distance (m)")
        self.ax.set_ylabel("Depth (m)")
        self.ax.set_title(f"Grid - {self.profile_name}")
        self.ax.grid(True, linestyle='--', alpha=0.5)
        
        if self.show_radar_var.get() and hasattr(self, 'radar_data') and self.radar_data is not None:
            rotated_data = np.rot90(self.radar_data, k=1)
            self.ax.imshow(rotated_data, extent=[0, self.distance_interval, 0, self.max_depth], 
                          aspect='auto', cmap='gray', alpha=0.7, origin='upper')
            
        if self.grid_active:
            self.draw_grid()
            
        self.canvas.draw()
        
    def create_grid(self):
        self.grid_cell_size = self.cell_size_var.get()
        
        self.grid_rows = int(np.ceil(self.max_depth / self.grid_cell_size))
        self.grid_cols = int(np.ceil(self.distance_interval / self.grid_cell_size))
        
        self.grid_data = [[None for _ in range(self.grid_cols)] for _ in range(self.grid_rows)]
        self.grid_active = True
        
        self.status_var.set(f"Grid created: {self.grid_rows} x {self.grid_cols} cells")
        
        self.plot_grid_only()
        
        messagebox.showinfo("Grid Created", 
                           f"Grid created with {self.grid_rows} rows and {self.grid_cols} columns.\n"
                           f"Each cell is {self.grid_cell_size}m x {self.grid_cell_size}m.\n"
                           f"Enable Fill Mode to add markers to cells.")
        
    def draw_grid(self):
        for i in range(self.grid_rows + 1):
            y = self.max_depth - i * self.grid_cell_size
            self.ax.axhline(y=y, color='gray', linestyle='-', alpha=0.5)
            
        for j in range(self.grid_cols + 1):
            x = j * self.grid_cell_size
            self.ax.axvline(x=x, color='gray', linestyle='-', alpha=0.5)
            
        for i in range(self.grid_rows):
            for j in range(self.grid_cols):
                if self.grid_data[i][j] is not None:
                    marker = self.grid_data[i][j]
                    x = j * self.grid_cell_size + self.grid_cell_size / 2
                    y = self.max_depth - (i * self.grid_cell_size + self.grid_cell_size / 2)
                    self.ax.scatter(x, y, marker=marker['shape'], color=marker['color'], s=marker['size']**2)
                    
    def toggle_fill_mode(self):
        if not self.grid_active:
            messagebox.showwarning("Grid Not Active", "Please create a grid first.")
            return
            
        self.fill_mode = not self.fill_mode
        
        if self.fill_mode:
            self.fill_mode_button.config(text="Disable Fill Mode", bg="#f44336")
            self.status_var.set("Fill Mode: ON - Click on grid cells to fill them")
            messagebox.showinfo("Fill Mode Enabled", 
                              "Fill Mode is now active.\n"
                              "Click on grid cells to fill them with the selected marker.\n"
                              "Use the marker properties controls to change shape, color, and size.")
        else:
            self.fill_mode_button.config(text="Enable Fill Mode", bg="#2196F3")
            self.status_var.set("Fill Mode: OFF")
            
    def on_canvas_click(self, event):
        if event.inaxes != self.ax or not self.fill_mode:
            return
            
        x, y = event.xdata, event.ydata
        
        col = int(x / self.grid_cell_size)
        row = int((self.max_depth - y) / self.grid_cell_size)
        
        if 0 <= row < self.grid_rows and 0 <= col < self.grid_cols:
            self.grid_data[row][col] = {
                'shape': self.marker_shape,
                'color': self.marker_color,
                'size': self.marker_size
            }
            
            self.status_var.set(f"Filled cell at row {row+1}, col {col+1}")
            
            self.plot_grid_only()
            
    def update_marker_properties(self, event=None):
        self.marker_shape = self.shape_var.get()
        self.marker_size = self.size_var.get()
        
    def choose_color(self):
        color = colorchooser.askcolor(title="Choose Marker Color")
        if color[1]:
            self.marker_color = color[1]
            self.color_button.config(bg=self.marker_color)
            
    def clear_grid(self):
        if not self.grid_active:
            messagebox.showwarning("Grid Not Active", "No grid to clear.")
            return
            
        self.grid_data = [[None for _ in range(self.grid_cols)] for _ in range(self.grid_rows)]
        
        self.status_var.set("Grid cleared")
        
        self.plot_grid_only()
        
        messagebox.showinfo("Grid Cleared", "All markers have been removed from the grid.")
        
    def save_grid(self):
        if not self.grid_active:
            messagebox.showwarning("Grid Not Active", "No grid to save.")
            return
            
        file_path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialfile=f"{self.profile_name}_grid.json"
        )
        
        if file_path:
            try:
                grid_info = {
                    'profile_name': self.profile_name,
                    'grid_rows': self.grid_rows,
                    'grid_cols': self.grid_cols,
                    'grid_cell_size': self.grid_cell_size,
                    'grid_data': self.grid_data
                }
                
                with open(file_path, 'w') as f:
                    json.dump(grid_info, f)
                    
                self.status_var.set(f"Grid data saved to {os.path.basename(file_path)}")
                messagebox.showinfo("Grid Saved", f"Grid data saved to {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save grid: {str(e)}")
                
    def save_plot_as_image(self):
        if not self.grid_active:
            messagebox.showwarning("Grid Not Active", "No grid to save as image.")
            return
            
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[
                ("PNG files", "*.png"), 
                ("JPEG files", "*.jpg"), 
                ("All files", "*.*")
            ],
            initialfile=f"{self.profile_name}_grid.png"
        )
        
        if file_path:
            try:
                self.figure.savefig(
                    file_path, 
                    dpi=300, 
                    bbox_inches='tight', 
                    pad_inches=0.1,
                    transparent=False,
                    facecolor='white'
                )
                
                self.status_var.set(f"Plot saved as {os.path.basename(file_path)}")
                messagebox.showinfo("Success", f"Grid plot saved as image to {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save plot as image: {str(e)}")
                
    def load_grid(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                with open(file_path, 'r') as f:
                    grid_info = json.load(f)
                    
                if grid_info.get('profile_name') != self.profile_name:
                    result = messagebox.askyesno("Profile Mismatch", 
                                                 f"This grid was created for profile '{grid_info.get('profile_name')}'.\n"
                                                 f"Current profile is '{self.profile_name}'.\n"
                                                 f"Do you want to load it anyway?")
                    if not result:
                        return
                        
                self.grid_rows = grid_info['grid_rows']
                self.grid_cols = grid_info['grid_cols']
                self.grid_cell_size = grid_info['grid_cell_size']
                self.grid_data = grid_info['grid_data']
                self.grid_active = True
                
                self.cell_size_var.set(self.grid_cell_size)
                
                self.status_var.set(f"Grid loaded from {os.path.basename(file_path)}")
                
                self.plot_grid_only()
                
                messagebox.showinfo("Grid Loaded", f"Grid loaded from {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load grid: {str(e)}")

class GPRImageReader:
    def __init__(self, root):
        self.root = root
        self.root.title(" GPR VORTEX1.0")
        self.root.geometry("1200x1200")
        self.data = None
        self.filtered_data = None
        self.markers = []
        self.marker_shape = 'o'
        self.marker_color = 'red'
        self.num_traces = 0
        self.num_samples = 0
        self.header_info = {}
        self.distance_interval = None
        self.time_interval = None
        self.s_v = None
        self.saved_2d_plots = []
        self.file_path = None
        self.profile_name = None
        self.distance_line = None  # Added for distance line
        
        self.create_widgets()
        
    def create_widgets(self):
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open IPRH/IPRB", command=self.browse_file)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        # Grid menu
        grid_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Grid", menu=grid_menu)
        grid_menu.add_command(label="Open Grid Tool", command=self.open_grid_tool)
        
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self.show_about)
        
        # File Selection Frame
        file_frame = tk.Frame(self.root, bg="#f0f0f0", padx=10, pady=10)
        file_frame.pack(fill=tk.X, pady=10)
        
        self.file_label = tk.Label(file_frame, text="Select an Iprh and Iprb:", font=("Helvetica", 12), bg="#f0f0f0")
        self.file_label.pack(side=tk.LEFT, padx=5)
        
        self.file_entry = tk.Entry(file_frame, width=60, font=("Helvetica", 12))
        self.file_entry.pack(side=tk.LEFT, padx=5)
        
        self.browse_button = tk.Button(
            file_frame,
            text="Browse",
            font=("Helvetica", 12),
            command=self.browse_file,
            bg="#4CAF50",
            fg="white"
        )
        self.browse_button.pack(side=tk.LEFT, padx=5)
        
        # Plot Frame
        plot_frame = tk.Frame(self.root, bg="#ffffff", bd=2, relief=tk.GROOVE)
        plot_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        self.figure, self.ax = plt.subplots(figsize=(8, 6))
        self.canvas = FigureCanvasTkAgg(self.figure, master=plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        control_frame = tk.Frame(plot_frame, bg="#f9f9f9", bd=2, relief=tk.GROOVE)
        control_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)
        
        self.create_gain_sliders(control_frame)
        self.create_cmap_selection(control_frame)
        
        # Buttons
        filter_button = tk.Button(
            control_frame,
            text="Open Filter Window",
            font=("Helvetica", 12),
            command=self.open_filter_window,
            bg="#008CBA",
            fg="white"
        )
        filter_button.pack(pady=10, fill=tk.X)
        
        save_button = tk.Button(
            control_frame,
            text="Save Plot",
            font=("Helvetica", 12),
            command=self.save_plot,
            bg="#FF9800",
            fg="white"
        )
        save_button.pack(pady=10, fill=tk.X)
        
        two_d_button = tk.Button(
            control_frame,
            text="Plot 2D",
            font=("Helvetica", 12),
            command=self.plot_2d_data,
            bg="#2196F3",
            fg="white"
        )
        two_d_button.pack(pady=10, fill=tk.X)
        
        slice_interpretation_button = tk.Button(
            control_frame,
            text="3D-Slice",
            font=("Helvetica", 12),
            command=self.open_slice_interpretation_window,
            bg="#E91E63",
            fg="white"
        )
        slice_interpretation_button.pack(pady=10, fill=tk.X)
        
        self.add_corner_image()
        
    def add_corner_image(self):
        corner_image = Image.open("mh.png")
        corner_image = corner_image.resize((240, 240), Image.LANCZOS)
        corner_photo = ImageTk.PhotoImage(corner_image)
        canvas = tk.Canvas(self.root, width=215, height=270, highlightthickness=0, bg=self.root["bg"])
        canvas.place(relx=1, rely=1, anchor="se", x=-13, y=-40)
        canvas.create_image(0, 0, anchor="nw", image=corner_photo)
        canvas.image = corner_photo
        canvas.create_text(120, 255, text="GPR VORTEX", font=("Arial", 15, "bold"), fill="red")
        
    def show_about(self):
        html_content = """
        <!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GPR VORTEX — Advanced Ground Penetrating Radar Software</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://code.iconify.design/iconify-icon/1.0.7/iconify-icon.min.js"></script>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <link href="https://fonts.googleapis.com/css2?family=Geist:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <script>
        tailwind.config = {
            theme: {
                extend: {
                    fontFamily: {
                        heading: ['Space Grotesk', 'sans-serif'],
                        body: ['Geist', 'sans-serif'],
                    }
                }
            }
        }
    </script>
    <style>
        :root {
            --brand-bg: #030303;
            --brand-text: #ffffff;
            --brand-text-muted: #888888;
            --brand-accent: #a855f7;
            --brand-accent-light: #c084fc;
            --brand-border: rgba(255,255,255,0.05);
            --brand-border-sub: rgba(255,255,255,0.1);
        }

        * { margin: 0; padding: 0; box-sizing: border-box; }
        html { scroll-behavior: smooth; }

        body {
            background: var(--brand-bg);
            color: var(--brand-text);
            font-family: 'Geist', sans-serif;
            font-feature-settings: "cv11", "ss01";
            overflow-x: hidden;
        }

        /* Grid Background */
        .grid-bg {
            position: fixed;
            inset: 0;
            z-index: 0;
            pointer-events: none;
            display: flex;
            justify-content: center;
        }
        .grid-bg-inner {
            width: 100%;
            max-width: 80rem;
            display: flex;
            justify-content: space-between;
            padding: 0 1rem;
        }
        .grid-line {
            width: 1px;
            height: 100%;
            background: rgba(255,255,255,0.02);
        }

        /* Animations */
        @keyframes animationIn {
            0% { opacity: 0; transform: translateY(30px); filter: blur(8px); }
            100% { opacity: 1; transform: translateY(0); filter: blur(0px); }
        }
        .anim-in { opacity: 0; animation: animationIn 0.8s ease-out both; }
        .anim-delay-1 { animation-delay: 0.1s; }
        .anim-delay-2 { animation-delay: 0.2s; }
        .anim-delay-3 { animation-delay: 0.3s; }
        .anim-delay-4 { animation-delay: 0.4s; }
        .anim-delay-5 { animation-delay: 0.5s; }
        .anim-delay-6 { animation-delay: 0.6s; }
        .anim-delay-7 { animation-delay: 0.7s; }
        .anim-delay-8 { animation-delay: 0.8s; }
        .anim-delay-10 { animation-delay: 1.0s; }

        @keyframes radarSweep { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        @keyframes radarPulse { 0% { transform: scale(0.8); opacity: 0.6; } 50% { transform: scale(1); opacity: 0.3; } 100% { transform: scale(0.8); opacity: 0.6; } }
        @keyframes radarRingExpand { 0% { transform: scale(0.5); opacity: 0.4; } 100% { transform: scale(2.5); opacity: 0; } }
        @keyframes float { 0%, 100% { transform: translateY(0px); } 50% { transform: translateY(-10px); } }
        @keyframes glowPulse { 0%, 100% { opacity: 0.4; } 50% { opacity: 0.8; } }
        @keyframes lineScan { 0% { top: 0%; opacity: 0; } 10% { opacity: 1; } 90% { opacity: 1; } 100% { top: 100%; opacity: 0; } }
        @keyframes waveMove { 0% { transform: translateX(0); } 100% { transform: translateX(-50%); } }

        .radar-sweep { animation: radarSweep 4s linear infinite; }
        .radar-pulse { animation: radarPulse 3s ease-in-out infinite; }
        .radar-ring { animation: radarRingExpand 3s ease-out infinite; }
        .radar-ring-delay { animation: radarRingExpand 3s ease-out infinite 1.5s; }
        .float-anim { animation: float 6s ease-in-out infinite; }
        .glow-pulse { animation: glowPulse 3s ease-in-out infinite; }
        .line-scan { animation: lineScan 4s ease-in-out infinite; }

        /* Glass */
        .glass {
            background: rgba(255,255,255,0.03);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border: 1px solid rgba(255,255,255,0.05);
        }
        .glass-hover { transition: all 300ms ease; }
        .glass-hover:hover {
            border-color: rgba(168, 85, 247, 0.3);
            box-shadow: 0 0 30px rgba(168, 85, 247, 0.15);
        }

        /* Gradient Text */
        .gradient-text {
            background: linear-gradient(to right, #c084fc, #60a5fa, #fff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        .gradient-text-purple {
            background: linear-gradient(135deg, #a855f7, #c084fc);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        .divider { height: 1px; background: linear-gradient(to right, transparent, #262626, transparent); }

        .btn-brand {
            background: linear-gradient(135deg, #7c3aed, #2563eb);
            transition: all 300ms ease;
        }
        .btn-brand:hover { transform: scale(1.05); box-shadow: 0 0 25px rgba(124, 58, 237, 0.5); }

        /* Section Reveal */
        .reveal-section {
            opacity: 0;
            transform: translateY(40px);
            transition: all 0.8s ease-out;
        }
        .reveal-section.visible { opacity: 1; transform: translateY(0); }

        /* Feature Card */
        .feature-card { position: relative; overflow: hidden; }
        .feature-card::before {
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0;
            height: 1px;
            background: linear-gradient(to right, transparent, rgba(168, 85, 247, 0.3), transparent);
            opacity: 0;
            transition: opacity 300ms ease;
        }
        .feature-card:hover::before { opacity: 1; }

        /* Waveform */
        .waveform-container { overflow: hidden; position: relative; height: 50px; }
        .waveform { display: flex; align-items: center; gap: 2px; animation: waveMove 8s linear infinite; }
        .wave-bar { width: 2px; background: rgba(168, 85, 247, 0.3); border-radius: 1px; flex-shrink: 0; }

        /* Scrollbar */
        ::-webkit-scrollbar { width: 6px; }
        ::-webkit-scrollbar-track { background: #030303; }
        ::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.1); border-radius: 3px; }
        ::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.2); }

        .radar-visual { position: relative; width: 280px; height: 280px; }
        @media (min-width: 768px) { .radar-visual { width: 360px; height: 360px; } }

        .nav-link { transition: color 150ms ease; }
        .nav-link:hover { color: #ffffff; }

        .stat-number { font-variant-numeric: tabular-nums; }

        .sep-dot { width: 4px; height: 4px; border-radius: 50%; background: rgba(255,255,255,0.3); }

        /* Application card hover */
        .app-card { transition: all 300ms ease; }
        .app-card:hover { transform: translateY(-4px); border-color: rgba(168,85,247,0.25); }
    </style>
</head>
<body>

    <!-- Grid Background -->
    <div class="grid-bg">
        <div class="grid-bg-inner">
            <div class="grid-line"></div>
            <div class="grid-line hidden md:block"></div>
            <div class="grid-line hidden lg:block"></div>
            <div class="grid-line hidden lg:block"></div>
            <div class="grid-line"></div>
        </div>
    </div>

    <!-- Ambient Glow -->
    <div class="fixed top-0 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[600px] rounded-full bg-purple-600/5 blur-[120px] pointer-events-none z-0 glow-pulse"></div>
    <div class="fixed bottom-0 right-0 translate-x-1/3 translate-y-1/3 w-[400px] h-[400px] rounded-full bg-blue-600/5 blur-[100px] pointer-events-none z-0 glow-pulse" style="animation-delay:1.5s;"></div>

    <!-- Navigation -->
    <nav class="fixed top-6 left-1/2 -translate-x-1/2 w-[calc(100%-2rem)] max-w-5xl z-50 glass rounded-2xl px-4 md:px-6 py-3 flex items-center justify-between anim-in anim-delay-1" style="box-shadow:0 25px 50px -12px rgba(0,0,0,0.25);">
        <a href="#hero" class="flex items-center gap-2.5">
            <div class="w-8 h-8 rounded-lg flex items-center justify-center" style="background:linear-gradient(135deg,#7c3aed,#2563eb);">
                <iconify-icon icon="mdi:radar" width="18" style="color:white;"></iconify-icon>
            </div>
            <span class="font-heading font-semibold text-sm tracking-wider uppercase">GPR VORTEX</span>
        </a>
        <div class="hidden md:flex items-center gap-8">
            <a href="#about" class="nav-link text-xs uppercase tracking-widest text-neutral-400">About</a>
            <a href="#features" class="nav-link text-xs uppercase tracking-widest text-neutral-400">Features</a>
            <a href="#applications" class="nav-link text-xs uppercase tracking-widest text-neutral-400">Applications</a>
            <a href="#developer" class="nav-link text-xs uppercase tracking-widest text-neutral-400">Developer</a>
        </div>
        <button id="mobileMenuBtn" class="md:hidden text-white">
            <iconify-icon icon="mdi:menu" width="24"></iconify-icon>
        </button>
    </nav>

    <!-- Mobile Menu -->
    <div id="mobileMenu" class="fixed inset-0 z-40 bg-black/90 backdrop-blur-xl flex flex-col items-center justify-center gap-8 hidden">
        <button id="mobileMenuClose" class="absolute top-8 right-8 text-white">
            <iconify-icon icon="mdi:close" width="28"></iconify-icon>
        </button>
        <a href="#about" class="mobile-link text-lg uppercase tracking-widest text-neutral-300 hover:text-white transition-colors">About</a>
        <a href="#features" class="mobile-link text-lg uppercase tracking-widest text-neutral-300 hover:text-white transition-colors">Features</a>
        <a href="#applications" class="mobile-link text-lg uppercase tracking-widest text-neutral-300 hover:text-white transition-colors">Applications</a>
        <a href="#developer" class="mobile-link text-lg uppercase tracking-widest text-neutral-300 hover:text-white transition-colors">Developer</a>
    </div>

    <!-- ==================== HERO ==================== -->
    <section id="hero" class="relative z-10 min-h-screen flex items-center pt-40 md:pt-52 pb-20 md:pb-32">
        <div class="max-w-7xl mx-auto px-4 md:px-8 w-full">
            <div class="grid lg:grid-cols-2 gap-12 lg:gap-16 items-center">
                <div>
                    <div class="anim-in anim-delay-2 flex items-center gap-3 mb-8">
                        <div class="h-px w-8 bg-purple-500/50"></div>
                        <span class="text-[10px] uppercase tracking-widest text-neutral-400">Advanced GPR Processing Platform</span>
                    </div>
                    <h1 class="anim-in anim-delay-3 font-heading font-semibold text-5xl md:text-7xl lg:text-8xl tracking-tighter uppercase leading-[0.9] mb-8">
                        <span class="gradient-text">GPR</span><br>
                        <span class="text-white">VORTEX</span>
                    </h1>
                    <p class="anim-in anim-delay-4 text-lg md:text-2xl font-light text-neutral-400 leading-relaxed max-w-lg mb-10">
                        Advanced Ground Penetrating Radar (GPR) Processing and Interpretation Software
                    </p>
                    <div class="anim-in anim-delay-5 flex flex-wrap gap-4">
                        <a href="#about" class="btn-brand inline-flex items-center gap-2 px-6 py-3 rounded-full text-xs font-semibold uppercase tracking-widest text-white">
                            Explore
                            <iconify-icon icon="mdi:arrow-down" width="16"></iconify-icon>
                        </a>
                        <a href="#features" class="inline-flex items-center gap-2 px-6 py-3 rounded-full text-xs font-semibold uppercase tracking-widest text-neutral-400 border border-white/10 hover:border-purple-500/30 hover:text-white transition-all duration-300">
                            Features
                            <iconify-icon icon="mdi:arrow-right" width="16"></iconify-icon>
                        </a>
                    </div>
                    <div class="anim-in anim-delay-7 mt-14 flex gap-10 md:gap-14">
                        <div>
                            <div class="stat-number font-heading text-3xl md:text-4xl font-light tracking-tighter text-white">2D/3D</div>
                            <div class="text-[10px] uppercase tracking-widest text-neutral-500 mt-1">Visualization</div>
                        </div>
                        <div>
                            <div class="stat-number font-heading text-3xl md:text-4xl font-light tracking-tighter text-white">HD</div>
                            <div class="text-[10px] uppercase tracking-widest text-neutral-500 mt-1">Processing</div>
                        </div>
                        <div>
                            <div class="stat-number font-heading text-3xl md:text-4xl font-light tracking-tighter text-white">Fast</div>
                            <div class="text-[10px] uppercase tracking-widest text-neutral-500 mt-1">Ready</div>
                        </div>
                    </div>
                </div>
                <div class="anim-in anim-delay-6 flex justify-center lg:justify-end">
                    <div class="radar-visual float-anim">
                        <svg viewBox="0 0 360 360" fill="none" xmlns="http://www.w3.org/2000/svg" class="w-full h-full">
                            <circle cx="180" cy="180" r="170" stroke="rgba(255,255,255,0.04)" stroke-width="1"/>
                            <circle cx="180" cy="180" r="135" stroke="rgba(255,255,255,0.04)" stroke-width="1"/>
                            <circle cx="180" cy="180" r="100" stroke="rgba(255,255,255,0.05)" stroke-width="1"/>
                            <circle cx="180" cy="180" r="65" stroke="rgba(255,255,255,0.05)" stroke-width="1"/>
                            <circle cx="180" cy="180" r="30" stroke="rgba(255,255,255,0.06)" stroke-width="1"/>
                            <line x1="180" y1="10" x2="180" y2="350" stroke="rgba(255,255,255,0.03)" stroke-width="1"/>
                            <line x1="10" y1="180" x2="350" y2="180" stroke="rgba(255,255,255,0.03)" stroke-width="1"/>
                            <line x1="55" y1="55" x2="305" y2="305" stroke="rgba(255,255,255,0.02)" stroke-width="1"/>
                            <line x1="305" y1="55" x2="55" y2="305" stroke="rgba(255,255,255,0.02)" stroke-width="1"/>
                            <defs>
                                <linearGradient id="sweepGrad" x1="180" y1="180" x2="180" y2="10" gradientUnits="userSpaceOnUse">
                                    <stop offset="0%" stop-color="rgba(168,85,247,0)"/>
                                    <stop offset="100%" stop-color="rgba(168,85,247,0.3)"/>
                                </linearGradient>
                            </defs>
                            <path d="M180,180 L180,10 A170,170 0 0,1 325,105 Z" fill="url(#sweepGrad)" class="radar-sweep" style="transform-origin:180px 180px;"/>
                            <line x1="180" y1="180" x2="180" y2="10" stroke="rgba(168,85,247,0.6)" stroke-width="1.5" class="radar-sweep" style="transform-origin:180px 180px;"/>
                            <circle cx="140" cy="120" r="4" fill="#a855f7" opacity="0.8" class="radar-pulse"/>
                            <circle cx="220" cy="150" r="3" fill="#60a5fa" opacity="0.7" class="radar-pulse" style="animation-delay:0.5s;"/>
                            <circle cx="160" cy="230" r="3.5" fill="#a855f7" opacity="0.6" class="radar-pulse" style="animation-delay:1s;"/>
                            <circle cx="240" cy="200" r="2.5" fill="#c084fc" opacity="0.7" class="radar-pulse" style="animation-delay:1.5s;"/>
                            <circle cx="120" cy="190" r="2" fill="#60a5fa" opacity="0.5" class="radar-pulse" style="animation-delay:2s;"/>
                            <circle cx="200" cy="100" r="2.5" fill="#a855f7" opacity="0.6" class="radar-pulse" style="animation-delay:0.8s;"/>
                            <circle cx="180" cy="180" r="5" fill="#a855f7" opacity="0.9"/>
                            <circle cx="180" cy="180" r="10" stroke="rgba(168,85,247,0.3)" stroke-width="1" class="radar-pulse"/>
                        </svg>
                        <div class="absolute inset-0 flex items-center justify-center pointer-events-none">
                            <div class="w-20 h-20 rounded-full border border-purple-500/20 radar-ring"></div>
                            <div class="absolute w-20 h-20 rounded-full border border-purple-500/20 radar-ring-delay"></div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        <div class="absolute bottom-8 left-1/2 -translate-x-1/2 anim-in anim-delay-10 flex flex-col items-center gap-2">
            <span class="text-[10px] uppercase tracking-widest text-neutral-600">Scroll</span>
            <div class="w-px h-8 bg-gradient-to-b from-neutral-600 to-transparent"></div>
        </div>
    </section>

    <!-- Waveform Divider -->
    <div class="relative z-10 overflow-hidden">
        <div class="waveform-container">
            <div class="waveform" id="waveform1"></div>
        </div>
    </div>

    <!-- ==================== ABOUT ==================== -->
    <section id="about" class="relative z-10 py-24 md:py-36">
        <div class="max-w-6xl mx-auto px-4 md:px-8">
            <div class="reveal-section mb-16 md:mb-20">
                <div class="flex items-center gap-3 mb-6">
                    <div class="sep-dot"></div>
                    <span class="text-[10px] uppercase tracking-widest text-neutral-500">About</span>
                    <div class="h-px flex-1 bg-white/5"></div>
                </div>
                <h2 class="font-heading font-light text-4xl md:text-6xl lg:text-7xl tracking-tighter uppercase leading-[0.9]">
                    About <span class="gradient-text-purple">GPR VORTEX</span>
                </h2>
            </div>

            <div class="grid lg:grid-cols-12 gap-8 lg:gap-12">
                <!-- Main Description -->
                <div class="lg:col-span-7 reveal-section">
                    <div class="glass rounded-2xl p-8 md:p-10 glass-hover relative overflow-hidden">
                        <div class="absolute -top-24 -right-24 w-48 h-48 rounded-full bg-purple-500/5 blur-[80px] pointer-events-none"></div>
                        <div class="relative z-10">
                            <div class="flex items-center gap-3 mb-6">
                                <div class="w-10 h-10 rounded-xl bg-purple-500/10 border border-purple-500/20 flex items-center justify-center">
                                    <iconify-icon icon="mdi:radar" width="20" style="color:#a855f7;"></iconify-icon>
                                </div>
                                <h3 class="font-heading text-lg tracking-tight">What is GPR VORTEX?</h3>
                            </div>
                            <p class="text-neutral-400 font-light leading-relaxed mb-5">
                                GPR VORTEX is a professional software solution designed for processing, visualization, and interpretation of Ground Penetrating Radar (GPR) data. The software provides researchers, engineers, geophysicists, and industry professionals with advanced tools for subsurface investigation and analysis.
                            </p>
                            <p class="text-neutral-400 font-light leading-relaxed">
                                GPR VORTEX integrates data processing, filtering, interpretation, and visualization capabilities into a unified platform, enabling users to efficiently analyze GPR datasets and generate high-quality results.
                            </p>
                        </div>
                    </div>
                </div>

                <!-- Side Cards -->
                <div class="lg:col-span-5 flex flex-col gap-6 reveal-section">
                    <!-- Target Users -->
                    <div class="glass rounded-2xl p-8 glass-hover">
                        <div class="flex items-center gap-3 mb-5">
                            <div class="w-10 h-10 rounded-xl bg-blue-500/10 border border-blue-500/20 flex items-center justify-center">
                                <iconify-icon icon="mdi:account-group-outline" width="20" style="color:#60a5fa;"></iconify-icon>
                            </div>
                            <h3 class="font-heading text-lg tracking-tight">Target Users</h3>
                        </div>
                        <div class="flex flex-wrap gap-2">
                            <span class="px-3 py-1.5 rounded-full text-[11px] uppercase tracking-wider text-neutral-300 border border-white/10 bg-white/[0.02]">Researchers</span>
                            <span class="px-3 py-1.5 rounded-full text-[11px] uppercase tracking-wider text-neutral-300 border border-white/10 bg-white/[0.02]">Engineers</span>
                            <span class="px-3 py-1.5 rounded-full text-[11px] uppercase tracking-wider text-neutral-300 border border-white/10 bg-white/[0.02]">Geophysicists</span>
                            <span class="px-3 py-1.5 rounded-full text-[11px] uppercase tracking-wider text-neutral-300 border border-white/10 bg-white/[0.02]">Industry Professionals</span>
                        </div>
                    </div>

                    <!-- Core Capabilities -->
                    <div class="glass rounded-2xl p-8 glass-hover">
                        <div class="flex items-center gap-3 mb-5">
                            <div class="w-10 h-10 rounded-xl bg-green-500/10 border border-green-500/20 flex items-center justify-center">
                                <iconify-icon icon="mdi:cog-outline" width="20" style="color:#4ade80;"></iconify-icon>
                            </div>
                            <h3 class="font-heading text-lg tracking-tight">Core Capabilities</h3>
                        </div>
                        <ul class="space-y-3">
                            <li class="flex items-center gap-3 text-sm text-neutral-400 font-light">
                                <iconify-icon icon="mdi:check-circle-outline" width="16" style="color:#4ade80;"></iconify-icon>
                                Data Processing & Enhancement
                            </li>
                            <li class="flex items-center gap-3 text-sm text-neutral-400 font-light">
                                <iconify-icon icon="mdi:check-circle-outline" width="16" style="color:#4ade80;"></iconify-icon>
                                Signal Filtering & Noise Reduction
                            </li>
                            <li class="flex items-center gap-3 text-sm text-neutral-400 font-light">
                                <iconify-icon icon="mdi:check-circle-outline" width="16" style="color:#4ade80;"></iconify-icon>
                                2D & 3D Visualization
                            </li>
                            <li class="flex items-center gap-3 text-sm text-neutral-400 font-light">
                                <iconify-icon icon="mdi:check-circle-outline" width="16" style="color:#4ade80;"></iconify-icon>
                                Interpretation & Analysis
                            </li>
                        </ul>
                    </div>
                </div>
            </div>
        </div>
    </section>

    <!-- Divider -->
    <div class="max-w-6xl mx-auto px-4 md:px-8"><div class="divider"></div></div>

    <!-- ==================== FEATURES ==================== -->
    <section id="features" class="relative z-10 py-24 md:py-36">
        <div class="max-w-6xl mx-auto px-4 md:px-8">
            <div class="reveal-section mb-16 md:mb-20">
                <div class="flex items-center gap-3 mb-6">
                    <div class="sep-dot"></div>
                    <span class="text-[10px] uppercase tracking-widest text-neutral-500">Capabilities</span>
                    <div class="h-px flex-1 bg-white/5"></div>
                </div>
                <h2 class="font-heading font-light text-4xl md:text-6xl lg:text-7xl tracking-tighter uppercase leading-[0.9]">
                    Key <span class="gradient-text-purple">Features</span>
                </h2>
            </div>

            <div class="grid md:grid-cols-2 gap-4">
                <div class="feature-card glass rounded-2xl p-8 glass-hover reveal-section">
                    <div class="w-12 h-12 rounded-xl bg-purple-500/10 border border-purple-500/20 flex items-center justify-center mb-6 transition-transform duration-300 hover:scale-110">
                        <iconify-icon icon="mdi:lightning-bolt" width="24" style="color:#a855f7;"></iconify-icon>
                    </div>
                    <h3 class="font-heading text-xl tracking-tight mb-3">Advanced Data Processing</h3>
                    <p class="text-neutral-500 font-light text-sm leading-relaxed">Advanced GPR data processing and enhancement algorithms for optimal signal quality.</p>
                </div>

                <div class="feature-card glass rounded-2xl p-8 glass-hover reveal-section">
                    <div class="w-12 h-12 rounded-xl bg-blue-500/10 border border-blue-500/20 flex items-center justify-center mb-6 transition-transform duration-300 hover:scale-110">
                        <iconify-icon icon="mdi:filter-variant" width="24" style="color:#60a5fa;"></iconify-icon>
                    </div>
                    <h3 class="font-heading text-xl tracking-tight mb-3">Signal Filtering & Noise Reduction</h3>
                    <p class="text-neutral-500 font-light text-sm leading-relaxed">Comprehensive signal filtering and noise reduction tools to clean and enhance radar data.</p>
                </div>

                <div class="feature-card glass rounded-2xl p-8 glass-hover reveal-section">
                    <div class="w-12 h-12 rounded-xl bg-green-500/10 border border-green-500/20 flex items-center justify-center mb-6 transition-transform duration-300 hover:scale-110">
                        <iconify-icon icon="mdi:chart-line" width="24" style="color:#4ade80;"></iconify-icon>
                    </div>
                    <h3 class="font-heading text-xl tracking-tight mb-3">High-Resolution 2D Profiles</h3>
                    <p class="text-neutral-500 font-light text-sm leading-relaxed">High-resolution 2D profile visualization with customizable color maps and annotation tools.</p>
                </div>

                <div class="feature-card glass rounded-2xl p-8 glass-hover reveal-section">
                    <div class="w-12 h-12 rounded-xl bg-amber-500/10 border border-amber-500/20 flex items-center justify-center mb-6 transition-transform duration-300 hover:scale-110">
                        <iconify-icon icon="mdi:cube-scan" width="24" style="color:#fbbf24;"></iconify-icon>
                    </div>
                    <h3 class="font-heading text-xl tracking-tight mb-3">Interactive 3D Subsurface Imaging</h3>
                    <p class="text-neutral-500 font-light text-sm leading-relaxed">Interactive 3D subsurface imaging with rotation, slicing, and volume exploration capabilities.</p>
                </div>

                <div class="feature-card glass rounded-2xl p-8 glass-hover reveal-section">
                    <div class="w-12 h-12 rounded-xl bg-pink-500/10 border border-pink-500/20 flex items-center justify-center mb-6 transition-transform duration-300 hover:scale-110">
                        <iconify-icon icon="mdi:cursor-default-click-outline" width="24" style="color:#f472b6;"></iconify-icon>
                    </div>
                    <h3 class="font-heading text-xl tracking-tight mb-3">User-Friendly Interface</h3>
                    <p class="text-neutral-500 font-light text-sm leading-relaxed">User-friendly graphical interface designed for efficiency and ease of use.</p>
                </div>

                <div class="feature-card glass rounded-2xl p-8 glass-hover reveal-section">
                    <div class="w-12 h-12 rounded-xl bg-cyan-500/10 border border-cyan-500/20 flex items-center justify-center mb-6 transition-transform duration-300 hover:scale-110">
                        <iconify-icon icon="mdi:flask-outline" width="24" style="color:#22d3ee;"></iconify-icon>
                    </div>
                    <h3 class="font-heading text-xl tracking-tight mb-3">Data Interpretation & Analysis</h3>
                    <p class="text-neutral-500 font-light text-sm leading-relaxed">Data interpretation and analysis tools for scientific investigation and decision-making.</p>
                </div>

                <div class="feature-card glass rounded-2xl p-8 glass-hover reveal-section">
                    <div class="w-12 h-12 rounded-xl bg-violet-500/10 border border-violet-500/20 flex items-center justify-center mb-6 transition-transform duration-300 hover:scale-110">
                        <iconify-icon icon="mdi:file-export-outline" width="24" style="color:#8b5cf6;"></iconify-icon>
                    </div>
                    <h3 class="font-heading text-xl tracking-tight mb-3">Export of Figures & Results</h3>
                    <p class="text-neutral-500 font-light text-sm leading-relaxed">Export of figures and processed results in multiple professional formats for reporting.</p>
                </div>

                <div class="feature-card glass rounded-2xl p-8 glass-hover reveal-section">
                    <div class="w-12 h-12 rounded-xl bg-rose-500/10 border border-rose-500/20 flex items-center justify-center mb-6 transition-transform duration-300 hover:scale-110">
                        <iconify-icon icon="mdi:database-speed" width="24" style="color:#fb7185;"></iconify-icon>
                    </div>
                    <h3 class="font-heading text-xl tracking-tight mb-3">Efficient Large Dataset Handling</h3>
                    <p class="text-neutral-500 font-light text-sm leading-relaxed">Efficient handling of large datasets with optimized memory management and fast rendering.</p>
                </div>
            </div>
        </div>
    </section>

    <!-- Divider -->
    <div class="max-w-6xl mx-auto px-4 md:px-8"><div class="divider"></div></div>

    <!-- ==================== APPLICATIONS ==================== -->
    <section id="applications" class="relative z-10 py-24 md:py-36">
        <div class="max-w-6xl mx-auto px-4 md:px-8">
            <div class="grid lg:grid-cols-12 gap-12 lg:gap-16 items-start">
                <!-- Left Header -->
                <div class="lg:col-span-4 reveal-section">
                    <div class="flex items-center gap-3 mb-6">
                        <div class="sep-dot"></div>
                        <span class="text-[10px] uppercase tracking-widest text-neutral-500">Use Cases</span>
                    </div>
                    <h2 class="font-heading font-light text-4xl md:text-5xl lg:text-6xl tracking-tighter uppercase leading-[0.9] mb-6">
                        Applic<span class="gradient-text-purple">ations</span>
                    </h2>
                    <p class="text-neutral-500 font-light text-sm leading-relaxed">
                        GPR VORTEX serves a wide range of professional fields requiring accurate subsurface investigation and analysis.
                    </p>
                </div>

                <!-- Right Grid -->
                <div class="lg:col-span-8">
                    <div class="grid sm:grid-cols-2 gap-4">
                        <div class="app-card glass rounded-2xl p-7 reveal-section">
                            <div class="w-11 h-11 rounded-xl bg-purple-500/10 border border-purple-500/20 flex items-center justify-center mb-5">
                                <iconify-icon icon="mdi:earth" width="22" style="color:#a855f7;"></iconify-icon>
                            </div>
                            <h3 class="font-heading text-base tracking-tight mb-2">Geophysical Investigations</h3>
                            <p class="text-neutral-500 font-light text-sm leading-relaxed">Subsurface mapping, stratigraphic analysis, and geological structure identification.</p>
                        </div>

                        <div class="app-card glass rounded-2xl p-7 reveal-section">
                            <div class="w-11 h-11 rounded-xl bg-blue-500/10 border border-blue-500/20 flex items-center justify-center mb-5">
                                <iconify-icon icon="mdi:bridge" width="22" style="color:#60a5fa;"></iconify-icon>
                            </div>
                            <h3 class="font-heading text-base tracking-tight mb-2">Geotechnical Engineering</h3>
                            <p class="text-neutral-500 font-light text-sm leading-relaxed">Soil characterization, bedrock profiling, and foundation assessment for engineering projects.</p>
                        </div>

                        <div class="app-card glass rounded-2xl p-7 reveal-section">
                            <div class="w-11 h-11 rounded-xl bg-amber-500/10 border border-amber-500/20 flex items-center justify-center mb-5">
                                <iconify-icon icon="mdi:pyramid" width="22" style="color:#fbbf24;"></iconify-icon>
                            </div>
                            <h3 class="font-heading text-base tracking-tight mb-2">Archaeological Surveys</h3>
                            <p class="text-neutral-500 font-light text-sm leading-relaxed">Non-destructive detection of buried structures, artifacts, and archaeological features.</p>
                        </div>

                        <div class="app-card glass rounded-2xl p-7 reveal-section">
                            <div class="w-11 h-11 rounded-xl bg-green-500/10 border border-green-500/20 flex items-center justify-center mb-5">
                                <iconify-icon icon="mdi:pipe" width="22" style="color:#4ade80;"></iconify-icon>
                            </div>
                            <h3 class="font-heading text-base tracking-tight mb-2">Utility & Infrastructure Detection</h3>
                            <p class="text-neutral-500 font-light text-sm leading-relaxed">Locating underground pipes, cables, and infrastructure elements for safe excavation.</p>
                        </div>

                        <div class="app-card glass rounded-2xl p-7 reveal-section">
                            <div class="w-11 h-11 rounded-xl bg-cyan-500/10 border border-cyan-500/20 flex items-center justify-center mb-5">
                                <iconify-icon icon="mdi:water" width="22" style="color:#22d3ee;"></iconify-icon>
                            </div>
                            <h3 class="font-heading text-base tracking-tight mb-2">Environmental & Groundwater Studies</h3>
                            <p class="text-neutral-500 font-light text-sm leading-relaxed">Water table mapping, contaminant plume detection, and environmental site assessment.</p>
                        </div>

                        <div class="app-card glass rounded-2xl p-7 reveal-section">
                            <div class="w-11 h-11 rounded-xl bg-rose-500/10 border border-rose-500/20 flex items-center justify-center mb-5">
                                <iconify-icon icon="mdi:pickaxe" width="22" style="color:#fb7185;"></iconify-icon>
                            </div>
                            <h3 class="font-heading text-base tracking-tight mb-2">Mining & Exploration</h3>
                            <p class="text-neutral-500 font-light text-sm leading-relaxed">Ore body delineation, void detection, and subsurface mapping for mining operations.</p>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </section>

    <!-- Divider -->
    <div class="max-w-6xl mx-auto px-4 md:px-8"><div class="divider"></div></div>

    <!-- ==================== DEVELOPER ==================== -->
    <section id="developer" class="relative z-10 py-24 md:py-36">
        <div class="max-w-6xl mx-auto px-4 md:px-8">
            <div class="reveal-section mb-16 md:mb-20">
                <div class="flex items-center gap-3 mb-6">
                    <div class="sep-dot"></div>
                    <span class="text-[10px] uppercase tracking-widest text-neutral-500">Developer</span>
                    <div class="h-px flex-1 bg-white/5"></div>
                </div>
                <h2 class="font-heading font-light text-4xl md:text-6xl lg:text-7xl tracking-tighter uppercase leading-[0.9]">
                    Developer <span class="gradient-text-purple">Information</span>
                </h2>
            </div>

            <div class="grid lg:grid-cols-2 gap-8">
                <!-- Developer Card -->
                <div class="reveal-section">
                    <div class="glass rounded-2xl p-8 md:p-10 glass-hover h-full relative overflow-hidden">
                        <div class="absolute -top-20 -right-20 w-60 h-60 rounded-full bg-purple-500/5 blur-[80px] pointer-events-none"></div>
                        <div class="relative z-10">
                            <div class="flex items-center gap-4 mb-8">
                                <div class="w-16 h-16 rounded-2xl flex items-center justify-center" style="background:linear-gradient(135deg,#7c3aed,#2563eb);">
                                    <iconify-icon icon="mdi:account" width="32" style="color:white;"></iconify-icon>
                                </div>
                                <div>
                                    <div class="text-[10px] uppercase tracking-widest text-neutral-500 mb-1">Designed, Developed & Maintained by</div>
                                    <h3 class="font-heading text-2xl tracking-tight">Mohammad Hassan Soleimani</h3>
                                </div>
                            </div>
                            <p class="text-neutral-400 font-light leading-relaxed mb-6">
                                GPR VORTEX is continuously developed to provide modern, reliable, and scientifically robust solutions for Ground Penetrating Radar data analysis. Future versions will incorporate advanced automation, artificial intelligence, and enhanced visualization capabilities.
                            </p>
                            <div class="flex items-center gap-3 pt-4 border-t border-white/5">
                                <iconify-icon icon="mdi:code-tags" width="16" style="color:#a855f7;"></iconify-icon>
                                <span class="text-xs text-neutral-500 uppercase tracking-wider">Full-Stack Development</span>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Future Roadmap -->
                <div class="reveal-section">
                    <div class="glass rounded-2xl p-8 md:p-10 glass-hover h-full relative overflow-hidden">
                        <div class="absolute -bottom-20 -left-20 w-60 h-60 rounded-full bg-blue-500/5 blur-[80px] pointer-events-none"></div>
                        <div class="relative z-10">
                            <div class="flex items-center gap-4 mb-8">
                                <div class="w-16 h-16 rounded-2xl bg-white/[0.03] border border-white/10 flex items-center justify-center">
                                    <iconify-icon icon="mdi:rocket-launch-outline" width="32" style="color:#fbbf24;"></iconify-icon>
                                </div>
                                <div>
                                    <div class="text-[10px] uppercase tracking-widest text-neutral-500 mb-1">Looking Ahead</div>
                                    <h3 class="font-heading text-2xl tracking-tight">Future Roadmap</h3>
                                </div>
                            </div>
                            <div class="space-y-5">
                                <div class="flex gap-4">
                                    <div class="flex-shrink-0 w-8 h-8 rounded-lg bg-purple-500/10 flex items-center justify-center mt-0.5">
                                        <iconify-icon icon="mdi:brain" width="16" style="color:#a855f7;"></iconify-icon>
                                    </div>
                                    <div>
                                        <div class="text-sm font-medium mb-1">Artificial Intelligence</div>
                                        <div class="text-xs text-neutral-500 font-light leading-relaxed">AI-powered interpretation and automated feature recognition.</div>
                                    </div>
                                </div>
                                <div class="flex gap-4">
                                    <div class="flex-shrink-0 w-8 h-8 rounded-lg bg-blue-500/10 flex items-center justify-center mt-0.5">
                                        <iconify-icon icon="mdi:auto-fix" width="16" style="color:#60a5fa;"></iconify-icon>
                                    </div>
                                    <div>
                                        <div class="text-sm font-medium mb-1">Advanced Automation</div>
                                        <div class="text-xs text-neutral-500 font-light leading-relaxed">Streamlined processing pipelines with minimal manual intervention.</div>
                                    </div>
                                </div>
                                <div class="flex gap-4">
                                    <div class="flex-shrink-0 w-8 h-8 rounded-lg bg-green-500/10 flex items-center justify-center mt-0.5">
                                        <iconify-icon icon="mdi:cube-outline" width="16" style="color:#4ade80;"></iconify-icon>
                                    </div>
                                    <div>
                                        <div class="text-sm font-medium mb-1">Enhanced Visualization</div>
                                        <div class="text-xs text-neutral-500 font-light leading-relaxed">Next-generation 3D rendering and real-time data exploration.</div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </section>

    <!-- ==================== FOOTER ==================== -->
    <footer class="relative z-10 border-t border-white/5">
        <div class="max-w-6xl mx-auto px-4 md:px-8 py-12 md:py-16">
            <div class="grid md:grid-cols-3 gap-10 md:gap-16">
                <div>
                    <div class="flex items-center gap-2.5 mb-4">
                        <div class="w-8 h-8 rounded-lg flex items-center justify-center" style="background:linear-gradient(135deg,#7c3aed,#2563eb);">
                            <iconify-icon icon="mdi:radar" width="18" style="color:white;"></iconify-icon>
                        </div>
                        <span class="font-heading font-semibold text-sm tracking-wider uppercase">GPR VORTEX</span>
                    </div>
                    <p class="text-neutral-500 text-sm font-light leading-relaxed">
                        Advanced Ground Penetrating Radar (GPR) Processing and Interpretation Software
                    </p>
                </div>
                <div>
                    <h4 class="text-[10px] uppercase tracking-widest text-neutral-500 mb-4">Navigation</h4>
                    <div class="flex flex-col gap-2">
                        <a href="#about" class="text-sm text-neutral-400 hover:text-white transition-colors duration-150 font-light">About</a>
                        <a href="#features" class="text-sm text-neutral-400 hover:text-white transition-colors duration-150 font-light">Features</a>
                        <a href="#applications" class="text-sm text-neutral-400 hover:text-white transition-colors duration-150 font-light">Applications</a>
                        <a href="#developer" class="text-sm text-neutral-400 hover:text-white transition-colors duration-150 font-light">Developer</a>
                    </div>
                </div>
                <div>
                    <h4 class="text-[10px] uppercase tracking-widest text-neutral-500 mb-4">Information</h4>
                    <p class="text-neutral-500 text-sm font-light leading-relaxed mb-2">© 2026 GPR VORTEX. All Rights Reserved.</p>
                    <p class="text-neutral-500 text-sm font-light leading-relaxed">Developed by Mohammad Hassan Soleimani</p>
                </div>
            </div>
            <div class="mt-12 pt-8 border-t border-white/5 flex flex-col md:flex-row items-center justify-between gap-4">
                <div class="flex items-center gap-2">
                    <div class="w-1.5 h-1.5 rounded-full bg-green-500"></div>
                    <span class="text-[10px] uppercase tracking-widest text-neutral-600">GPR VORTEX — Operational</span>
                </div>
                <div class="text-[10px] uppercase tracking-widest text-neutral-600">Precision Subsurface Intelligence</div>
            </div>
        </div>
    </footer>

    <script>
        // Waveform bars
        document.addEventListener('DOMContentLoaded', () => {
            const wf = document.getElementById('waveform1');
            if (wf) {
                for (let i = 0; i < 400; i++) {
                    const bar = document.createElement('div');
                    bar.className = 'wave-bar';
                    const h = Math.sin(i * 0.05) * 18 + Math.sin(i * 0.02) * 12 + Math.random() * 8 + 5;
                    bar.style.height = h + 'px';
                    bar.style.opacity = (0.15 + Math.random() * 0.2).toString();
                    wf.appendChild(bar);
                }
            }
        });

        // Scroll Reveal
        const revealObserver = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) entry.target.classList.add('visible');
            });
        }, { threshold: 0.1, rootMargin: '0px 0px -50px 0px' });
        document.querySelectorAll('.reveal-section').forEach(el => revealObserver.observe(el));

        // Mobile Menu
        const mobileMenuBtn = document.getElementById('mobileMenuBtn');
        const mobileMenu = document.getElementById('mobileMenu');
        const mobileMenuClose = document.getElementById('mobileMenuClose');
        mobileMenuBtn.addEventListener('click', () => { mobileMenu.classList.remove('hidden'); document.body.style.overflow = 'hidden'; });
        mobileMenuClose.addEventListener('click', () => { mobileMenu.classList.add('hidden'); document.body.style.overflow = ''; });
        document.querySelectorAll('.mobile-link').forEach(link => {
            link.addEventListener('click', () => { mobileMenu.classList.add('hidden'); document.body.style.overflow = ''; });
        });

        // Smooth scroll
        document.querySelectorAll('a[href^="#"]').forEach(anchor => {
            anchor.addEventListener('click', function (e) {
                e.preventDefault();
                const target = document.querySelector(this.getAttribute('href'));
                if (target) target.scrollIntoView({ behavior: 'smooth', block: 'start' });
            });
        });

        // Nav background on scroll
        const nav = document.querySelector('nav');
        window.addEventListener('scroll', () => {
            nav.style.background = window.scrollY > 100 ? 'rgba(3,3,3,0.8)' : 'rgba(255,255,255,0.03)';
        });
    </script>

</body>
</html>
        """
        with open("about.html", "w", encoding="utf-8") as file:
            file.write(html_content)
        webbrowser.open("about.html")
        
    def open_grid_tool(self):
        if self.data is None:
            messagebox.showerror("Error", "No data loaded. Please open a profile first.")
            return
            
        radar_data = self.filtered_data if self.filtered_data is not None else self.data
            
        grid_window = GridWindow(
            parent=self.root,
            distance_interval=self.distance_interval,
            time_interval=self.time_interval,
            s_v=self.s_v,
            profile_name=self.profile_name,
            radar_data=radar_data
        )
        
    def open_filter_window(self):
        filter_window = Toplevel(self.root)
        filter_window.title("Filter Options")
        filter_window.geometry("400x540")
        self.create_filter_options(filter_window)
        
    def create_filter_options(self, parent):
        filter_frame = tk.Frame(parent)
        filter_frame.pack(fill=tk.X, pady=10)
        
        filter_label = tk.Label(filter_frame, text="Filter Options", font=('Helvetica', 14, 'bold'))
        filter_label.pack(pady=5)
        
        self.filter_vars = {}
        self.filter_params = {}
        self.filter_param_entries = {}
        
        self.filter_options = [
            ('Background Subtraction', {'window_size': 5}),
            ('Gaussian', {'sigma': 0.3}),
            ('Sobel Transform', {}),
            ('Absolute Value', {}),
            ('Remove Air-wave Reflections', {'threshold': 0.1}),
            ('Prewitt', {}),
            ('Laplace', {}),
            ('Shift', {'Shift_level': 5}),
            ('Binomial Filter', {'Binomial_size': 5}),
            ('Moving Average', {'Moving_size': 5}),
            ('Max Filter', {'Max_size': 3}),
            ('Min Filter', {'Min_size': 3}),
            ('Adaptive Thresholding', {'block_size': 11, 'C': 2}),
            ('Local Entropy Thresholding', {'Entropy_window_size': 1}),
            ('Scharr Transform', {}),  
        ]
        
        for filter_name, params in self.filter_options:
            row_frame = tk.Frame(filter_frame)
            row_frame.pack(fill=tk.X, pady=2)
            
            var = tk.BooleanVar()
            chk = tk.Checkbutton(row_frame, text=filter_name, variable=var)
            chk.pack(side=tk.LEFT, padx=5)
            
            self.filter_vars[filter_name] = var
            self.filter_params[filter_name] = params
            
            if params:
                for param_name, param_value in params.items():
                    param_subframe = tk.Frame(row_frame)
                    param_subframe.pack(side=tk.LEFT, padx=10)
                    
                    label = tk.Label(param_subframe, text=f"{param_name}:", fg='blue')
                    label.pack(side=tk.LEFT)
                    
                    entry = tk.Entry(param_subframe, width=8, fg='red')
                    entry.insert(0, str(param_value))
                    entry.pack(side=tk.LEFT)
                    
                    self.filter_param_entries[(filter_name, param_name)] = entry
                    
        apply_filter_button = tk.Button(filter_frame, text="Apply Filter", command=self.apply_filter)
        apply_filter_button.pack(pady=10)
        
    def apply_filter(self):
        if self.data is None:
            messagebox.showerror("Error", "No data loaded")
            return
            
        selected_filters = []
        for filter_name, var in self.filter_vars.items():
            if var.get():
                params = {}
                if filter_name == 'Remove Air-wave Reflections':
                    params['threshold'] = float(self.filter_param_entries[(filter_name, 'threshold')].get())
                elif filter_name == 'Background Subtraction':
                    params['window_size'] = int(self.filter_param_entries[(filter_name, 'window_size')].get())
                elif filter_name == 'Gaussian':
                    params['sigma'] = float(self.filter_param_entries[(filter_name, 'sigma')].get())
                elif filter_name == 'Shift':
                    params['Shift_level'] = int(self.filter_param_entries[(filter_name, 'Shift_level')].get())
                elif filter_name == 'Binomial Filter':
                    params['Binomial_size'] = int(self.filter_param_entries[(filter_name, 'Binomial_size')].get())
                elif filter_name == 'Moving Average':
                    params['Moving_size'] = int(self.filter_param_entries[(filter_name, 'Moving_size')].get())
                elif filter_name == 'Max Filter':
                    params['Max_size'] = int(self.filter_param_entries[(filter_name, 'Max_size')].get())
                elif filter_name == 'Min Filter':
                    params['Min_size'] = int(self.filter_param_entries[(filter_name, 'Min_size')].get())
                elif filter_name == 'Adaptive Thresholding':
                    params['block_size'] = int(self.filter_param_entries[(filter_name, 'block_size')].get())
                    params['C'] = int(self.filter_param_entries[(filter_name, 'C')].get())
                elif filter_name == 'Local Entropy Thresholding':
                    params['Entropy_window_size'] = int(self.filter_param_entries[(filter_name, 'Entropy_window_size')].get())
                selected_filters.append((filter_name, params))
                
        if not selected_filters:
            messagebox.showwarning("Warning", "No filters selected.")
            return
            
        self.filtered_data = apply_filters(self.data.copy(), selected_filters)
        self.update_filtered_gain(self.filtered_gain_slider.get())
        
    def create_gain_sliders(self, parent):
        original_gain_label = tk.Label(parent, text="Gain (Original Data)", font=('Helvetica', 10, 'bold'))
        original_gain_label.pack(pady=5)
        
        self.original_gain_slider = tk.Scale(parent, from_=0, to=10, resolution=0.1, orient=tk.HORIZONTAL, command=self.update_original_gain)
        self.original_gain_slider.set(1)
        self.original_gain_slider.pack(pady=5)
        
        filtered_gain_label = tk.Label(parent, text="Gain (Filtered Data)", font=('Helvetica', 10, 'bold'))
        filtered_gain_label.pack(pady=3)
        
        self.filtered_gain_slider = tk.Scale(parent, from_=0, to=10, resolution=0.1, orient=tk.HORIZONTAL, command=self.update_filtered_gain)
        self.filtered_gain_slider.set(1)
        self.filtered_gain_slider.pack(pady=3)
        
    def create_cmap_selection(self, parent):
        cmap_label = tk.Label(parent, text="Select Colormap for 2D Plot", font=('Helvetica', 12, 'bold'))
        cmap_label.pack(pady=5)
        
        self.cmap_var = tk.StringVar()
        self.cmap_var.set("gray")
        
        colormaps = ['gray', 'viridis', 'plasma', 'inferno', 'magma', 'cividis',
                      'hot', 'spring', 'summer', 'autumn', 'winter', 'bone',
                      'copper', 'twilight', 'twilight_shifted', 'ocean', 'gist_earth']
        
        colormap_menu = tk.OptionMenu(parent, self.cmap_var, *colormaps)
        colormap_menu.pack(pady=5)
        
    def browse_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("IPRH Files", "*.iprh"), ("All Files", "*.*")])
        if file_path:
            self.file_path = file_path
            self.file_entry.delete(0, tk.END)
            self.file_entry.insert(0, file_path)
            
            self.header_info = read_iprh_file(file_path)
            if self.header_info:
                self.show_header_info()
                print(self.header_info)
                
                self.num_traces = int(self.header_info.get("LAST TRACE", 0))
                self.num_samples = int(self.header_info.get("SAMPLES", 0))
                self.distance_interval = float(self.header_info.get("STOP POSITION", 0))
                self.time_interval = float(self.header_info.get("TIMEWINDOW", 0))
                self.s_v = float(self.header_info.get("SOIL VELOCITY", 0)) * 0.001
                
                base_name = os.path.basename(file_path)
                file_name = os.path.splitext(base_name)[0]
                self.profile_name = file_name
                
                iprb_file_path = filedialog.askopenfilename(filetypes=[("IPRB Files", "*.iprb"), ("All Files", "*.*")])
                if iprb_file_path:
                    self.data = read_iprb_file(iprb_file_path, self.num_traces, self.num_samples)
                    if self.data is not None:
                        self.display_data(self.data)
                        
    def show_header_info(self):
        header_window = Toplevel(self.root)
        header_window.title("Header Information")
        header_window.geometry("400x300")
        
        text_area = scrolledtext.ScrolledText(header_window, wrap=tk.WORD, width=48, height=15)
        text_area.pack(padx=10, pady=10)
        
        for key, value in self.header_info.items():
            text_area.insert(tk.END, f"{key}: {value}\n")
            
        text_area.configure(state=tk.DISABLED)
        
    def display_data(self, data):
        self.ax.clear()
        num_traces = data.shape[0]
        for i in range(num_traces):
            self.ax.plot(data[i], color='gray', alpha=0.5)
            
        title = f"GPR Data: Runs: {self.num_traces}, Samples: {self.num_samples}, Antenna: {self.header_info.get('ANTENNA', 'Unknown')}"
        self.ax.set_title(title)
        self.ax.set_xlabel("Sample Number")
        self.ax.set_ylabel("Amplitude")
        self.ax.grid(True)
        self.canvas.draw()
        
    def update_filtered_gain(self, value):
        if self.filtered_data is not None:
            gain_value = float(value)
            gain_applied_data = self.filtered_data * gain_value
            self.display_filtered_2d(gain_applied_data)
            
    def update_original_gain(self, value):
        if self.data is not None:
            gain_value = float(value)
            gain_applied_data = self.data * gain_value
            self.display_data(gain_applied_data)
            
    def plot_2d_data(self):
        if self.data is None or self.num_traces == 0 or self.num_samples == 0:
            messagebox.showerror("Error", "No data loaded.")
            return
            
        plot_2d_window = Toplevel(self.root)
        plot_2d_window.title("2D GPR Data Plot")
        self.create_2d_plot_widgets(plot_2d_window)
        self.display_filtered_2d(self.data)
        
    def create_2d_plot_widgets(self, parent):
        # Add profile name label at the top
        profile_label_frame = tk.Frame(parent, bg="#f0f0f0", height=40)
        profile_label_frame.pack(fill=tk.X, padx=10, pady=(10, 0))
        
        if self.profile_name:
            profile_label = tk.Label(profile_label_frame, 
                                    text=f"Profile: {self.profile_name}", 
                                    font=('Helvetica', 14, 'bold'), 
                                    bg="#f0f0f0")
            profile_label.pack(pady=5)
        
        # Create figure with single subplot
        self.fig_2d, self.ax_2d = plt.subplots(figsize=(8, 8))
        
        # Calculate depth values
        time_values = np.linspace(0, self.time_interval, self.num_samples)
        depth = (self.s_v * time_values) / 2
        
        # Set up extent with distance in km and depth in m
        extent = [0, self.distance_interval, depth[-1], 0]
        
        # Create the initial plot
        rotated_data = np.rot90(self.data, k=1)
        self.im = self.ax_2d.imshow(rotated_data, aspect='auto', extent=extent, cmap=self.cmap_var.get(), origin='lower')
        
        # Set title and labels with distance in km
        self.ax_2d.set_title(f"2D GPR Data: Runs: {self.num_traces}, Samples: {self.num_samples}, Antenna: {self.header_info.get('ANTENNA', 'Unknown')}")
        self.ax_2d.set_xlabel("Distance (m)")
        self.ax_2d.set_ylabel("Depth (m)")
        
        # Add light gray grid
        self.ax_2d.grid(True, linestyle='--', alpha=0.5, color='lightgray')
        
        # Add red dashed horizontal line at the bottom (y=0)
        self.ax_2d.axhline(y=0, color='red', linestyle='--', linewidth=1.5)
        
        control_frame = tk.Frame(parent)
        control_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)
        
        self.create_marker_controls(control_frame)
        self.create_cmap_selection(control_frame)
        
        # Store the current displayed data
        self.current_2d_data = self.data.copy()
        
        self.gain_slider_2d = tk.Scale(control_frame, from_=0, to=10, resolution=0.1, orient=tk.HORIZONTAL, command=self.update_gain_2d)
        self.gain_slider_2d.set(1)
        self.gain_slider_2d.pack(pady=5)
        
        self.filter_type_2d = tk.StringVar()
        self.filter_type_2d.set("None")
        self.filter_options_2d = ['None', 'Background Subtraction', 'Gaussian', 'Sobel Transform']
        filter_menu_2d = tk.OptionMenu(control_frame, self.filter_type_2d, *self.filter_options_2d)
        filter_menu_2d.pack(pady=5)
        
        apply_2d_filter_button = tk.Button(control_frame, text="Apply Filter", command=self.apply_filter_to_2d)
        apply_2d_filter_button.pack(pady=5)
        
        save_2d_button = tk.Button(control_frame, text="Save 2D Plot", command=lambda: self.save_2d_plot(self.fig_2d))
        save_2d_button.pack(pady=5)
        
        # NEW: Add distance line button
        distance_button = tk.Button(control_frame, text="Draw Distance Line", 
                                   command=self.open_distance_dialog,
                                   bg="#9C27B0", fg="white")
        distance_button.pack(pady=5)
        
        self.fig_2d.canvas.mpl_connect('button_press_event', self.add_marker_2d)
        
        self.canvas_2d = FigureCanvasTkAgg(self.fig_2d, master=parent)
        self.canvas_2d.draw()
        self.canvas_2d.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def create_marker_controls(self, parent):
        shape_label = tk.Label(parent, text="Select Marker Shape", font=('Helvetica', 14, 'bold'))
        shape_label.pack(pady=5)
        
        self.marker_shape_var = tk.StringVar()
        self.marker_shape_var.set('o')
        self.marker_shapes = ['o', 's', '^', 'D', '*', 'p', 'h']
        self.marker_shape_menu = tk.OptionMenu(parent, self.marker_shape_var, *self.marker_shapes, command=self.update_marker_shape)
        self.marker_shape_menu.pack(pady=5)
        
        color_label = tk.Label(parent, text="Select Marker Color", font=('Helvetica', 14, 'bold'))
        color_label.pack(pady=5)
        
        self.color_button = tk.Button(parent, text="Choose Color", command=self.choose_color)
        self.color_button.pack(pady=5)
        
    def choose_color(self):
        color = colorchooser.askcolor(title="Choose Marker Color")
        if color[1]:
            self.marker_color = color[1]
            
    def update_marker_shape(self, shape):
        self.marker_shape = shape
        
    def add_marker_2d(self, event):
        if event.inaxes != self.ax_2d:
            return
            
        if event.xdata is not None and event.ydata is not None:
            self.markers.append((event.xdata, event.ydata))
            self.ax_2d.scatter(event.xdata, event.ydata, marker=self.marker_shape_var.get(), color=self.marker_color)
            self.ax_2d.annotate(f"({event.xdata:.1f}, {event.ydata:.1f})", xy=(event.xdata, event.ydata),
                       xytext=(5, 5), textcoords="offset points", arrowprops=dict(arrowstyle="->", lw=1), fontsize=8)
            self.canvas_2d.draw()
            
    def update_gain_2d(self, value):
        gain_value = float(value)
        if self.data is not None:
            gain_applied_data = self.data * gain_value
            self.current_2d_data = gain_applied_data.copy()
            self.display_filtered_2d(gain_applied_data)
            
    def apply_filter_to_2d(self):
        if self.data is None:
            messagebox.showerror("Error", "No data loaded")
            return
            
        filter_type = self.filter_type_2d.get()
        if filter_type == "None":
            filtered_data = self.data
        else:
            filtered_data = apply_filters(self.data, [(filter_type, {})])
            
        gain_value = self.gain_slider_2d.get()
        gain_applied_data = filtered_data * gain_value
        self.current_2d_data = gain_applied_data.copy()
        self.display_filtered_2d(gain_applied_data)
            
    def display_filtered_2d(self, data):
        self.ax_2d.clear()
        
        time_values = np.linspace(0, self.time_interval, self.num_samples)
        depth = (self.s_v * time_values) / 2
        
        extent = [0, self.distance_interval, depth[-1], 0]
        
        rotated_data = np.rot90(data, k=1)
        self.ax_2d.imshow(rotated_data, aspect='auto', extent=extent, cmap=self.cmap_var.get(), origin='lower')
        
        self.ax_2d.set_title("2D GPR Data (Filtered)")
        self.ax_2d.set_xlabel("Distance (m)")
        self.ax_2d.set_ylabel("Depth (m)")
        
        self.ax_2d.grid(True, linestyle='--', alpha=0.5, color='lightgray')
        self.ax_2d.axhline(y=0, color='red', linestyle='--', linewidth=1.5)
        
        self.canvas_2d.draw()
        
    def save_plot(self):
        if self.filtered_data is None and self.data is None:
            messagebox.showerror("Error", "No data to save")
            return
            
        image_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png"), ("All files", "*.*")])
        if image_path:
            try:
                self.save_current_plot(image_path)
                messagebox.showinfo("Success", f"Plot successfully saved to {image_path}")
                self.show_image(image_path)
            except Exception as e:
                messagebox.showerror("Error", f"Error saving file: {e}")
                
    def save_current_plot(self, image_path):
        self.ax.clear()
        if self.data is not None:
            for i in range(self.data.shape[0]):
                self.ax.plot(self.data[i], color='gray', alpha=0.5, label='Original Data')
                
        if self.filtered_data is not None:
            for i in range(self.filtered_data.shape[0]):
                self.ax.plot(self.filtered_data[i], color='blue', alpha=0.5, label='Filtered Data')
                
        for marker in self.markers:
            self.ax.scatter(marker[0], marker[1], marker=self.marker_shape, color=self.marker_color)
            self.ax.annotate(f"({marker[0]:.1f}, {marker[1]:.1f}) ", xy=(marker[0], marker[1]),
                              xytext=(5, 5), textcoords="offset points", arrowprops=dict(arrowstyle="->", lw=1), fontsize=8)
                              
        title = f"GPR Data: Runs: {self.num_traces}, Samples: {self.num_samples}, Antenna: {self.header_info.get('ANTENNA', 'Unknown')}"
        self.ax.set_title(title)
        self.ax.set_xlabel("Sample Number")
        self.ax.set_ylabel("Amplitude")
        self.ax.legend()
        self.ax.grid(True)
        self.figure.savefig(image_path, dpi=300)
        
    def show_image(self, image_path):
        img_window = Toplevel(self.root)
        img_window.title("Saved Plot")
        img = Image.open(image_path).resize((1000, 600), Image.LANCZOS)
        img_tk = ImageTk.PhotoImage(img)
        label = tk.Label(img_window, image=img_tk)
        label.image = img_tk
        label.pack()
        
    def save_2d_plot(self, fig):
        if self.file_path:
            base_name = os.path.basename(self.file_path)
            file_name = os.path.splitext(base_name)[0]
            profile_name = file_name
        else:
            profile_name = f"Profile_{len(self.saved_2d_plots)+1}"
            
        plot_info = {
            'figure': fig,
            'data': self.current_2d_data.copy(),
            'header_info': self.header_info.copy(),
            'num_traces': self.num_traces,
            'num_samples': self.num_samples,
            'distance_interval': self.distance_interval,
            'time_interval': self.time_interval,
            's_v': self.s_v,
            'profile_name': profile_name,
            'direction': 'vertical',
            'rotation': 0
        }
        self.saved_2d_plots.append(plot_info)
        messagebox.showinfo("Success", f"2D plot saved for 3D slice interpretation with name: {profile_name}")
        
    def open_slice_interpretation_window(self):
        if not self.saved_2d_plots:
            messagebox.showerror("Error", "No 2D plots have been saved yet.")
            return
            
        slice_window = Toplevel(self.root)
        slice_window.title("3D Slice Interpretation")
        slice_window.geometry("1400x900")
        
        main_container = tk.Frame(slice_window)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        left_panel = tk.Frame(main_container, width=400, bg="#f0f0f0")
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        right_panel = tk.Frame(main_container, width=600, bg="#f9f9f9")
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0))
        
        profile_label = tk.Label(left_panel, text="Saved Profiles", font=('Helvetica', 14, 'bold'), bg="#f0f0f0")
        profile_label.pack(pady=(0, 10))
        
        columns = ('Profile Name', 'Distance (m)', 'Depth (m)', 'Y Position (m)', 'X Offset (m)', 'Direction', 'Rotation (°)')
        tree = ttk.Treeview(left_panel, columns=columns, show='headings', height=20)
        
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=120)
        
        scrollbar = ttk.Scrollbar(left_panel, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscroll=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        tree.pack(fill=tk.BOTH, expand=True)
        
        y_positions = {}
        x_offsets = {}
        profile_directions = {}
        profile_rotations = {}
        
        for idx, plot_info in enumerate(self.saved_2d_plots):
            depth = (plot_info['s_v'] * plot_info['time_interval']) / 2
            distance = plot_info['distance_interval']
            direction = plot_info.get('direction', 'vertical')
            rotation = plot_info.get('rotation', 0)
            
            tree.insert('', 'end', iid=f"profile_{idx}", 
                        values=(plot_info['profile_name'], 
                                f"{distance:.2f}", 
                                f"{depth:.2f}", 
                                "0.0", 
                                "0.0", 
                                direction.capitalize(), 
                                f"{rotation:.1f}"))
            
            y_positions[idx] = 0.0
            x_offsets[idx] = 0.0
            profile_directions[idx] = direction
            profile_rotations[idx] = rotation
        
        control_notebook = ttk.Notebook(right_panel)
        control_notebook.pack(fill=tk.BOTH, expand=True)
        
        position_tab = tk.Frame(control_notebook)
        control_notebook.add(position_tab, text="Position")
        
        position_frame = tk.LabelFrame(position_tab, text="Position Controls", padx=10, pady=10)
        position_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        tk.Label(position_frame, text="Select Profile:", font=('Helvetica', 12, 'bold')).pack(anchor=tk.W, pady=(0, 5))
        
        profile_var = tk.StringVar()
        profile_menu = ttk.Combobox(position_frame, textvariable=profile_var, state="readonly", font=('Helvetica', 10))
        profile_menu['values'] = [f"{plot_info['profile_name']}" for plot_info in self.saved_2d_plots]
        profile_menu.pack(fill=tk.X, pady=5)
        
        tk.Label(position_frame, text="Y Position (m):", font=('Helvetica', 12, 'bold')).pack(anchor=tk.W, pady=(10, 5))
        y_entry = tk.Entry(position_frame, font=('Helvetica', 12))
        y_entry.pack(fill=tk.X, pady=5)
        
        tk.Label(position_frame, text="X Offset (m):", font=('Helvetica', 12, 'bold')).pack(anchor=tk.W, pady=(10, 5))
        x_offset_entry = tk.Entry(position_frame, font=('Helvetica', 12))
        x_offset_entry.pack(fill=tk.X, pady=5)
        
        def set_y_position():
            selected = profile_menu.current()
            if selected == -1:
                messagebox.showwarning("Warning", "Please select a profile")
                return
                
            try:
                y_pos = float(y_entry.get())
                x_offset = float(x_offset_entry.get()) if x_offset_entry.get() else 0.0
                
                y_positions[selected] = y_pos
                x_offsets[selected] = x_offset
                
                # Update tree
                tree.item(f"profile_{selected}", values=(
                    self.saved_2d_plots[selected]['profile_name'],
                    f"{self.saved_2d_plots[selected]['distance_interval']:.2f}",
                    f"{(self.saved_2d_plots[selected]['s_v'] * self.saved_2d_plots[selected]['time_interval']) / 2:.2f}",
                    f"{y_pos:.2f}",
                    f"{x_offset:.2f}",
                    profile_directions[selected].capitalize(),
                    f"{profile_rotations[selected]:.1f}"
                ))
                
                update_2d_map()
                
                messagebox.showinfo("Success", f"Y position set to {y_pos:.2f} m and X offset set to {x_offset:.2f} m")
            except ValueError:
                messagebox.showerror("Error", "Please enter valid numbers")
        
        set_y_button = tk.Button(position_frame, text="Set Position", command=set_y_position, 
                                 bg="#4CAF50", fg="white", font=('Helvetica', 12, 'bold'))
        set_y_button.pack(fill=tk.X, pady=10)
        
        direction_tab = tk.Frame(control_notebook)
        control_notebook.add(direction_tab, text="Direction")
        
        direction_frame = tk.LabelFrame(direction_tab, text="Profile Direction", padx=10, pady=10)
        direction_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        tk.Label(direction_frame, text="Select Profile:", font=('Helvetica', 12, 'bold')).pack(anchor=tk.W, pady=(0, 5))
        
        profile_dir_var = tk.StringVar()
        profile_dir_menu = ttk.Combobox(direction_frame, textvariable=profile_dir_var, state="readonly", font=('Helvetica', 10))
        profile_dir_menu['values'] = [f"{plot_info['profile_name']}" for plot_info in self.saved_2d_plots]
        profile_dir_menu.pack(fill=tk.X, pady=5)
        
        tk.Label(direction_frame, text="Select Direction:", font=('Helvetica', 12, 'bold')).pack(anchor=tk.W, pady=(10, 5))
        direction_var = tk.StringVar()
        direction_var.set("vertical")
        direction_options = ["vertical", "horizontal"]
        direction_menu = ttk.Combobox(direction_frame, textvariable=direction_var, values=direction_options, state="readonly", font=('Helvetica', 10))
        direction_menu.pack(fill=tk.X, pady=5)
        
        def set_direction():
            selected = profile_dir_menu.current()
            if selected == -1:
                messagebox.showwarning("Warning", "Please select a profile")
                return
                
            direction = direction_var.get()
            profile_directions[selected] = direction
            self.saved_2d_plots[selected]['direction'] = direction
            
            tree.item(f"profile_{selected}", values=(
                self.saved_2d_plots[selected]['profile_name'],
                f"{self.saved_2d_plots[selected]['distance_interval']:.2f}",
                f"{(self.saved_2d_plots[selected]['s_v'] * self.saved_2d_plots[selected]['time_interval']) / 2:.2f}",
                f"{y_positions[selected]:.2f}",
                f"{x_offsets[selected]:.2f}",
                direction.capitalize(),
                f"{profile_rotations[selected]:.1f}"
            ))
            
            update_2d_map()
            
            messagebox.showinfo("Success", f"Direction set to {direction}")
        
        set_dir_button = tk.Button(direction_frame, text="Set Direction", command=set_direction,
                                  bg="#2196F3", fg="white", font=('Helvetica', 12, 'bold'))
        set_dir_button.pack(fill=tk.X, pady=10)
        
        rotation_tab = tk.Frame(control_notebook)
        control_notebook.add(rotation_tab, text="Rotation")
        
        rotation_frame = tk.LabelFrame(rotation_tab, text="Profile Rotation", padx=10, pady=10)
        rotation_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        tk.Label(rotation_frame, text="Select Profile:", font=('Helvetica', 12, 'bold')).pack(anchor=tk.W, pady=(0, 5))
        
        profile_rot_var = tk.StringVar()
        profile_rot_menu = ttk.Combobox(rotation_frame, textvariable=profile_rot_var, state="readonly", font=('Helvetica', 10))
        profile_rot_menu['values'] = [f"{plot_info['profile_name']}" for plot_info in self.saved_2d_plots]
        profile_rot_menu.pack(fill=tk.X, pady=5)
        
        tk.Label(rotation_frame, text="Rotation Angle (degrees):", font=('Helvetica', 12, 'bold')).pack(anchor=tk.W, pady=(10, 5))
        rotation_entry = tk.Entry(rotation_frame, font=('Helvetica', 12))
        rotation_entry.pack(fill=tk.X, pady=5)
        
        tk.Label(rotation_frame, text="Or use slider:", font=('Helvetica', 12, 'bold')).pack(anchor=tk.W, pady=(10, 5))
        rotation_slider = tk.Scale(rotation_frame, from_=-180, to=180, resolution=1, orient=tk.HORIZONTAL, 
                                  command=lambda v: rotation_entry.delete(0, tk.END) or rotation_entry.insert(0, v))
        rotation_slider.pack(fill=tk.X, pady=5)
        
        def set_rotation():
            selected = profile_rot_menu.current()
            if selected == -1:
                messagebox.showwarning("Warning", "Please select a profile")
                return
                
            try:
                rotation = float(rotation_entry.get())
                profile_rotations[selected] = rotation
                self.saved_2d_plots[selected]['rotation'] = rotation
                
                tree.item(f"profile_{selected}", values=(
                    self.saved_2d_plots[selected]['profile_name'],
                    f"{self.saved_2d_plots[selected]['distance_interval']:.2f}",
                    f"{(self.saved_2d_plots[selected]['s_v'] * self.saved_2d_plots[selected]['time_interval']) / 2:.2f}",
                    f"{y_positions[selected]:.2f}",
                    f"{x_offsets[selected]:.2f}",
                    profile_directions[selected].capitalize(),
                    f"{rotation:.1f}"
                ))
                
                update_2d_map()
                
                messagebox.showinfo("Success", f"Rotation set to {rotation} degrees")
            except ValueError:
                messagebox.showerror("Error", "Please enter a valid number")
        
        set_rot_button = tk.Button(rotation_frame, text="Set Rotation", command=set_rotation,
                                  bg="#FF9800", fg="white", font=('Helvetica', 12, 'bold'))
        set_rot_button.pack(fill=tk.X, pady=10)
        
        map_tab = tk.Frame(control_notebook)
        control_notebook.add(map_tab, text="2D Map View")
        
        map_frame = tk.Frame(map_tab)
        map_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        fig_map, ax_map = plt.subplots(figsize=(5, 4))
        canvas_map = FigureCanvasTkAgg(fig_map, master=map_frame)
        canvas_map.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        def update_2d_map():
            ax_map.clear()
            
            ax_map.set_xlabel('X Position (m)')
            ax_map.set_ylabel('Y Position (m)')
            ax_map.set_title('Profile Positions (Top View)')
            ax_map.grid(True, linestyle='--', alpha=0.7)
            
            min_x = 0
            max_x = 0
            min_y = float('inf')
            max_y = float('-inf')
            
            for idx, plot_info in enumerate(self.saved_2d_plots):
                distance = plot_info['distance_interval']
                y_pos = y_positions.get(idx, 0.0)
                x_offset = x_offsets.get(idx, 0.0)
                rotation = profile_rotations.get(idx, 0)
                
                rotation_rad = np.radians(rotation)
                projected_length = distance * abs(np.cos(rotation_rad))
                
                x_pos = x_offset
                
                max_x = max(max_x, x_pos + projected_length)
                min_y = min(min_y, y_pos)
                max_y = max(max_y, y_pos)
            
            x_padding = max_x * 0.1 if max_x > 0 else 1
            y_padding = (max_y - min_y) * 0.1 if max_y > min_y else 1
            
            ax_map.set_xlim(min_x - x_padding, max_x + x_padding)
            ax_map.set_ylim(min_y - y_padding, max_y + y_padding)
            
            for idx, plot_info in enumerate(self.saved_2d_plots):
                distance = plot_info['distance_interval']
                y_pos = y_positions.get(idx, 0.0)
                x_offset = x_offsets.get(idx, 0.0)
                rotation = profile_rotations.get(idx, 0)
                direction = profile_directions.get(idx, 'vertical')
                
                rotation_rad = np.radians(rotation)
                projected_length = distance * abs(np.cos(rotation_rad))
                
                color = 'blue' if direction == 'vertical' else 'green'
                
                rect = Rectangle((x_offset, y_pos - 0.5), projected_length, 1, 
                                linewidth=1, edgecolor='black', facecolor=color, alpha=0.7)
                ax_map.add_patch(rect)
                
                ax_map.text(x_offset + projected_length/2, y_pos, plot_info['profile_name'], 
                           ha='center', va='center', fontsize=8, rotation=rotation)
            
            canvas_map.draw()
        
        display_tab = tk.Frame(control_notebook)
        control_notebook.add(display_tab, text="Display")
        
        cmap_frame = tk.LabelFrame(display_tab, text="Colormap", padx=10, pady=10)
        cmap_frame.pack(fill=tk.X, padx=10, pady=10)
        
        tk.Label(cmap_frame, text="Select Colormap:", font=('Helvetica', 12, 'bold')).pack(anchor=tk.W, pady=(0, 5))
        
        self.slice_cmap_var = tk.StringVar()
        self.slice_cmap_var.set("viridis")
        
        colormaps = ['gray', 'viridis', 'plasma', 'inferno', 'magma', 'cividis']
        cmap_menu = ttk.Combobox(cmap_frame, textvariable=self.slice_cmap_var, values=colormaps, state="readonly", font=('Helvetica', 10))
        cmap_menu.pack(fill=tk.X, pady=5)
        
        aspect_frame = tk.LabelFrame(display_tab, text="Aspect Mode", padx=10, pady=10)
        aspect_frame.pack(fill=tk.X, padx=10, pady=10)
        
        tk.Label(aspect_frame, text="Select Aspect Mode:", font=('Helvetica', 12, 'bold')).pack(anchor=tk.W, pady=(0, 5))
        
        self.aspect_mode_var = tk.StringVar()
        self.aspect_mode_var.set("cube")
        
        aspect_modes = [
            ("Cube (Equal Ratio)", "cube"),
            ("Data (Based on Range)", "data"),
        ]
        
        aspect_menu = ttk.Combobox(aspect_frame, textvariable=self.aspect_mode_var, 
                                  values=[mode[0] for mode in aspect_modes], state="readonly", font=('Helvetica', 10))
        aspect_menu.pack(fill=tk.X, pady=5)
        
        self.aspect_mode_values = {mode[0]: mode[1] for mode in aspect_modes}
        
        instructions_tab = tk.Frame(control_notebook)
        control_notebook.add(instructions_tab, text="Instructions")
        
        instructions = scrolledtext.ScrolledText(instructions_tab, wrap=tk.WORD, width=60, height=20, font=('Helvetica', 10))
        instructions.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        instructions.insert(tk.END, 
            "Instructions:\n\n"
            "1. Select a profile from the dropdown in the Position tab\n"
            "2. Enter its Y position and X offset in meters\n"
            "3. Click 'Set Position'\n"
            "4. A red point will appear on the 2D plot showing the Y position\n"
            "5. Repeat for all profiles\n\n"
            "6. In the Direction tab, select a profile and set its direction (vertical or horizontal)\n\n"
            "7. In the Rotation tab, select a profile and set its rotation angle (degrees)\n"
            "   - You can enter the value directly or use the slider\n"
            "   - Positive values rotate counter-clockwise\n"
            "   - Negative values rotate clockwise\n\n"
            "8. In the 2D Map View tab, you can see a top-down view of all profile positions\n"
            "   - Blue rectangles represent vertical profiles\n"
            "   - Green rectangles represent horizontal profiles\n"
            "   - The width of the rectangle shows the projected length after rotation\n\n"
            "9. In the Display tab, select a colormap and aspect mode\n\n"
            "10. Click 'Generate 3D Slice' at the bottom\n\n"
            "Aspect Mode Options:\n"
            "- Cube (Equal Ratio): Forces the axes to be drawn as a cube with equal aspect ratios\n"
            "- Data (Based on Range): Uses the data's aspect ratio\n\n"
            "Direction Options:\n"
            "- Vertical: Depth is along Y-axis, distance along X-axis\n"
            "- Horizontal: Depth is along Z-axis, distance along X-axis\n\n"
            "Note: Y positions define where each profile is placed along the Y-axis in the 3D visualization.\n"
            "X offsets define the horizontal displacement of each profile in the X direction.\n"
            "The 2D Map View helps visualize the relative positions of all profiles before generating the 3D slice."
        )
        instructions.config(state=tk.DISABLED)
        
        generate_button = tk.Button(
            right_panel,
            text="Generate 3D Slice",
            font=("Helvetica", 14, 'bold'),
            command=lambda: self.generate_cube_plotly(slice_window, y_positions, profile_directions, profile_rotations, x_offsets),
            bg="#E91E63",
            fg="white",
            height=2
        )
        generate_button.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)
        
        update_2d_map()
        
    def generate_cube_plotly(self, parent_window, y_positions, profile_directions, profile_rotations, x_offsets):
        if not self.saved_2d_plots:
            messagebox.showerror("Error", "No 2D plots have been saved yet.")
            return
            
        fig = go.Figure()
        selected_cmap = self.slice_cmap_var.get()
        selected_aspect_mode = self.aspect_mode_var.get()
        aspect_mode = self.aspect_mode_values.get(selected_aspect_mode, "cube")
        
        for idx, plot_info in enumerate(self.saved_2d_plots):
            fig_2d = plot_info['figure']
            ax_2d = fig_2d.gca()
            
            image_data = ax_2d.images[0].get_array() if ax_2d.images else np.zeros((100, 100))
            
            data_min = image_data.min()
            data_max = image_data.max()
            if data_max - data_min > 0:
                normalized_data = (image_data - data_min) / (data_max - data_min)
            else:
                normalized_data = np.zeros_like(image_data)
            
            distance_interval = plot_info['distance_interval']
            time_interval = plot_info['time_interval']
            s_v = plot_info['s_v']
            max_depth = (s_v * time_interval) / 2
            
            direction = profile_directions.get(idx, 'vertical')
            rotation = profile_rotations.get(idx, 0)
            rotation_rad = np.radians(rotation)
            
            y_pos = y_positions.get(idx, 0.0)
            x_offset = x_offsets.get(idx, 0.0)
            
            if direction == 'vertical':
                x = np.linspace(0, distance_interval, image_data.shape[1])
                y = np.linspace(max_depth, 0, image_data.shape[0])
                x, y = np.meshgrid(x, y)
                
                z = np.full_like(x, y_pos)
                
                x = x + x_offset
                
                x_rot = x * np.cos(rotation_rad) + z * np.sin(rotation_rad)
                z_rot = -x * np.sin(rotation_rad) + z * np.cos(rotation_rad)
                
                fig.add_trace(go.Surface(
                    x=x_rot,
                    y=y,
                    z=z_rot,
                    surfacecolor=normalized_data,
                    colorscale=selected_cmap,
                    name=plot_info['profile_name'],
                    hovertemplate=(
                        f"<b>{plot_info['profile_name']}</b><br>"
                        "Distance: %{x:.2f} m<br>"
                        "Depth: %{y:.2f} m<br>"
                        "Y Position: %{z:.2f} m<br>"
                        "<extra></extra>"
                    )
                ))
            else:
                x = np.linspace(0, distance_interval, image_data.shape[1])
                z = np.linspace(0, max_depth, image_data.shape[0])
                x, z = np.meshgrid(x, z)
                
                y = np.full_like(x, y_pos)
                
                x = x + x_offset
                
                x_rot = x * np.cos(rotation_rad) + z * np.sin(rotation_rad)
                z_rot = -x * np.sin(rotation_rad) + z * np.cos(rotation_rad)
                
                fig.add_trace(go.Surface(
                    x=x_rot,
                    y=y,
                    z=z_rot,
                    surfacecolor=normalized_data,
                    colorscale=selected_cmap,
                    showscale=False,
                    name=plot_info['profile_name'],
                    hovertemplate=(
                        f"<b>{plot_info['profile_name']}</b><br>"
                        "Distance: %{x:.2f} m<br>"
                        "Y Position: %{y:.2f} m<br>"
                        "Depth: %{z:.2f} m<br>"
                        "<extra></extra>"
                    )
                ))
        
        fig.update_layout(
            title="3D GPR Slice Interpretation",
            scene=dict(
                xaxis=dict(title='', showticklabels=False),
                yaxis=dict(title='', showticklabels=False),
                zaxis=dict(title='', showticklabels=False, autorange="reversed"),
                aspectmode=aspect_mode
            ),
            margin=dict(l=0, r=0, b=0, t=30),
            legend=dict(orientation="h", yanchor="bottom", y=0.02, xanchor="right", x=0.95)
        )
        
        html_path = "slice_interpretation.html"
        fig.write_html(html_path, auto_open=False)
        webbrowser.open(html_path)
        
        messagebox.showinfo("Success", f"3D slice generated with {selected_aspect_mode} aspect mode and saved to {html_path}")
    
    def open_distance_dialog(self):
        dialog = Toplevel(self.root)
        dialog.title="Calculate Distance"
        dialog.geometry("300x200")
        
        input_frame = tk.Frame(dialog, padx=10, pady=10)
        input_frame.pack(fill=tk.BOTH, expand=True)
        
        tk.Label(input_frame, text="Start Latitude:").grid(row=0, column=0, sticky=tk.W, pady=5)
        start_lat = tk.Entry(input_frame)
        start_lat.grid(row=0, column=1, pady=5)
        
        tk.Label(input_frame, text="Start Longitude:").grid(row=1, column=0, sticky=tk.W, pady=5)
        start_lon = tk.Entry(input_frame)
        start_lon.grid(row=1, column=1, pady=5)
        
        tk.Label(input_frame, text="End Latitude:").grid(row=2, column=0, sticky=tk.W, pady=5)
        end_lat = tk.Entry(input_frame)
        end_lat.grid(row=2, column=1, pady=5)
        
        tk.Label(input_frame, text="End Longitude:").grid(row=3, column=0, sticky=tk.W, pady=5)
        end_lon = tk.Entry(input_frame)
        end_lon.grid(row=3, column=1, pady=5)
        
        def calculate_and_draw():
            try:
                lat1 = float(start_lat.get())
                lon1 = float(start_lon.get())
                lat2 = float(end_lat.get())
                lon2 = float(end_lon.get())
                
                R = 6371.0  # Earth radius in kilometers
                phi1 = radians(lat1)
                phi2 = radians(lat2)
                delta_phi = radians(lat2 - lat1)
                delta_lambda = radians(lon2 - lon1)
                a = sin(delta_phi / 2)**2 + cos(phi1) * cos(phi2) * sin(delta_lambda / 2)**2
                c = 2 * atan2(sqrt(a), sqrt(1 - a))
                distance_km = R * c
                distance_m = distance_km * 1000  # Convert to meters
                
                self.draw_distance_line(lat1, lon1, lat2, lon2, distance_m)
                
                dialog.destroy()
                messagebox.showinfo("Distance Calculated", f"Distance: {distance_m:.2f} meters")
                
            except ValueError:
                messagebox.showerror("Error", "Invalid coordinates entered!")
        
        calculate_button = tk.Button(dialog, text="Calculate & Draw", command=calculate_and_draw)
        calculate_button.pack(pady=10)
    
    def draw_distance_line(self, lat1, lon1, lat2, lon2, distance_m):
        # Remove existing distance line if present
        if hasattr(self, 'distance_line') and self.distance_line:
            self.distance_line.remove()
            for text in self.ax_2d.texts:
                if 'Distance:' in text.get_text():
                    text.remove()
        
        x_start = 0
        y_start = 0  # Surface level
        x_end = min(distance_m, self.distance_interval)  # Limit to plot width
        y_end = 0    # Surface level
        
        # Draw the line connecting the two points (SOLID line, not dashed)
        self.distance_line = self.ax_2d.plot(
            [x_start, x_end], 
            [y_start, y_end], 
            color='red', 
            linestyle='-', 
            linewidth=2,
            label=f'Distance: {distance_m:.1f} m'
        )[0]
        
        self.ax_2d.scatter(x_start, y_start, color='red', s=50, zorder=5)
        self.ax_2d.scatter(x_end, y_end, color='red', s=50, zorder=5)
        
        self.ax_2d.text(x_start, y_start - 0.1, f"Start: ({lat1:.4f}, {lon1:.4f})", 
                       ha='center', color='red', fontsize=8)
        self.ax_2d.text(x_end, y_end - 0.1, f"End: ({lat2:.4f}, {lon2:.4f})", 
                       ha='center', color='red', fontsize=8)
        
        self.fig_2d.text(0.5, 0.02, f"Distance: {distance_m:.1f} m", 
                         ha='center', color='red', fontweight='bold',
                         bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
        
        # Update the plot
        self.canvas_2d.draw()

if __name__ == "__main__":
    splash_screen()
    root = tk.Tk()
    app = GPRImageReader(root)
    root.mainloop()