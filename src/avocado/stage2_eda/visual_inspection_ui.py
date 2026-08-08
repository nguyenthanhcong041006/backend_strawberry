import os
from pathlib import Path
import cv2
import pandas as pd
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib

matplotlib.use("TkAgg")

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def load_data():
    csv_path = PROJECT_ROOT / "data" / "02_processed" / "avocado" / "eda" / "features" / "avocado_features.csv"
    if not os.path.exists(csv_path):
        return None
    df = pd.read_csv(csv_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

class AvocadoApp:
    def __init__(self, root, df):
        self.root = root
        self.root.title("🥑 Avocado Visual Inspection Interface")
        self.root.geometry("1400x900")
        
        self.df = df
        self.fruit_ids = self.df['fruit_id'].unique()
        self.current_fruit = self.fruit_ids[0]
        self.df_fruit = self.get_fruit_data(self.current_fruit)
        self.frame_idx = 0
        self.is_playing = False
        self.threshold_l = 35
        
        self.setup_ui()
        self.update_ui()
        
    def get_fruit_data(self, fruit_id):
        df_f = self.df[self.df['fruit_id'] == fruit_id].sort_values('timestamp').reset_index(drop=True)
        start_time = df_f['timestamp'].min()
        df_f['elapsed_hours'] = (df_f['timestamp'] - start_time).dt.total_seconds() / 3600.0
        return df_f
        
    def setup_ui(self):
        # --- Left Panel (Controls) ---
        left_frame = ttk.Frame(self.root, padding=10, width=300)
        left_frame.pack(side=tk.LEFT, fill=tk.Y)
        
        ttk.Label(left_frame, text="Controls", font=('Arial', 14, 'bold')).pack(pady=10)
        
        ttk.Label(left_frame, text="Select Fruit:").pack(anchor=tk.W)
        self.fruit_combo = ttk.Combobox(left_frame, values=list(self.fruit_ids), state="readonly")
        self.fruit_combo.set(self.current_fruit)
        self.fruit_combo.bind("<<ComboboxSelected>>", self.on_fruit_change)
        self.fruit_combo.pack(fill=tk.X, pady=5)
        
        self.play_btn = ttk.Button(left_frame, text="▶️ Play Time-lapse", command=self.toggle_play)
        self.play_btn.pack(fill=tk.X, pady=10)
        
        ttk.Label(left_frame, text="Timeline Frame:").pack(anchor=tk.W)
        self.slider = ttk.Scale(left_frame, from_=0, to=len(self.df_fruit)-1, orient=tk.HORIZONTAL, command=self.on_slider_change)
        self.slider.pack(fill=tk.X, pady=5)
        
        self.timestamp_lbl = ttk.Label(left_frame, text="Timestamp:")
        self.timestamp_lbl.pack(anchor=tk.W, pady=5)
        
        ttk.Label(left_frame, text="Darkness Threshold (L*):").pack(anchor=tk.W, pady=(20, 0))
        self.thresh_slider = ttk.Scale(left_frame, from_=0, to=100, orient=tk.HORIZONTAL, command=self.on_thresh_change)
        self.thresh_lbl = ttk.Label(left_frame, text=f"Value: {self.threshold_l}")
        self.thresh_slider.set(self.threshold_l)
        self.thresh_slider.pack(fill=tk.X, pady=5)
        self.thresh_lbl.pack(anchor=tk.W)
        
        ttk.Button(left_frame, text="🚩 Flag Bad Segmentation", command=self.flag_frame).pack(fill=tk.X, pady=30)
        
        # --- Right Panel (Visuals) ---
        right_frame = ttk.Frame(self.root, padding=10)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Metrics row
        metrics_frame = ttk.Frame(right_frame)
        metrics_frame.pack(fill=tk.X, pady=5)
        
        self.metric_vars = {
            "Dark Coverage": tk.StringVar(),
            "Green Ratio": tk.StringVar(),
            "Darkening Speed": tk.StringVar(),
            "Rapid Onset": tk.StringVar()
        }
        
        for k, var in self.metric_vars.items():
            f = ttk.LabelFrame(metrics_frame, text=k)
            f.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
            ttk.Label(f, textvariable=var, font=('Arial', 14, 'bold')).pack(pady=10)
            
        # Images row
        img_frame = ttk.Frame(right_frame)
        img_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        self.img_lbl_1 = ttk.Label(img_frame)
        self.img_lbl_1.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.img_lbl_2 = ttk.Label(img_frame)
        self.img_lbl_2.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Graphs row
        graphs_frame = ttk.Frame(right_frame)
        graphs_frame.pack(fill=tk.BOTH, expand=True)
        
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(10, 3))
        self.canvas = FigureCanvasTkAgg(self.fig, master=graphs_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def on_fruit_change(self, event):
        self.current_fruit = self.fruit_combo.get()
        self.df_fruit = self.get_fruit_data(self.current_fruit)
        self.slider.configure(to=len(self.df_fruit)-1)
        self.frame_idx = 0
        self.slider.set(0)
        self.update_ui()
        
    def on_slider_change(self, val):
        if not self.is_playing:
            self.frame_idx = int(float(val))
            if hasattr(self, '_slider_timer'):
                self.root.after_cancel(self._slider_timer)
            self._slider_timer = self.root.after(100, self.update_ui)
            
    def on_thresh_change(self, val):
        self.threshold_l = int(float(val))
        if hasattr(self, 'thresh_lbl'):
            self.thresh_lbl.config(text=f"Value: {self.threshold_l}")
        if hasattr(self, '_thresh_timer'):
            self.root.after_cancel(self._thresh_timer)
        self._thresh_timer = self.root.after(100, self.update_images)
        
    def toggle_play(self):
        self.is_playing = not self.is_playing
        if self.is_playing:
            self.play_btn.config(text="⏸️ Stop Time-lapse")
            if self.frame_idx >= len(self.df_fruit) - 1:
                self.frame_idx = 0
            self.play_loop()
        else:
            self.play_btn.config(text="▶️ Play Time-lapse")
            
    def play_loop(self):
        if self.is_playing:
            if self.frame_idx < len(self.df_fruit) - 1:
                self.frame_idx += 1
                self.slider.set(self.frame_idx)
                self.update_ui()
                self.root.after(100, self.play_loop) # 100ms delay
            else:
                self.is_playing = False
                self.play_btn.config(text="▶️ Play Time-lapse")
                
    def apply_dark_threshold(self, img_path):
        if not os.path.exists(img_path):
            return None, None
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img is None or img.shape[2] != 4:
            return None, None
            
        b, g, r, a = cv2.split(img)
        mask = a > 0
        bgr = cv2.merge([b, g, r])
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        l_chan, _, _ = cv2.split(lab)
        
        dark_mask = (l_chan < self.threshold_l) & mask
        
        overlay = bgr.copy()
        overlay[dark_mask] = [0, 0, 255]
        
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        
        # Transparent BG using RGBA
        rgba = cv2.merge([rgb[:,:,0], rgb[:,:,1], rgb[:,:,2], a])
        overlay_rgba = cv2.merge([overlay_rgb[:,:,0], overlay_rgb[:,:,1], overlay_rgb[:,:,2], a])
        return rgba, overlay_rgba
        
    def update_images(self):
        row = self.df_fruit.iloc[self.frame_idx]
        rgba, overlay_rgba = self.apply_dark_threshold(row['image_path'])
        
        if rgba is not None:
            # Resize for UI
            h, w = rgba.shape[:2]
            scale = 400 / h if h > 0 else 1
            new_w, new_h = int(w * scale), int(h * scale)
            
            rgba = cv2.resize(rgba, (new_w, new_h))
            overlay_rgba = cv2.resize(overlay_rgba, (new_w, new_h))
            
            img1 = ImageTk.PhotoImage(image=Image.fromarray(rgba))
            img2 = ImageTk.PhotoImage(image=Image.fromarray(overlay_rgba))
            
            self.img_lbl_1.config(image=img1)
            self.img_lbl_1.image = img1
            self.img_lbl_2.config(image=img2)
            self.img_lbl_2.image = img2
            
    def update_graphs(self, current_hours):
        self.ax1.clear()
        self.ax2.clear()
        
        # Dark Coverage
        self.ax1.plot(self.df_fruit['elapsed_hours'], self.df_fruit['dark_coverage'], color='tab:blue')
        self.ax1.axvline(x=current_hours, color='red', linestyle='--')
        self.ax1.set_title("Dark Coverage Over Time")
        self.ax1.set_xlabel("Elapsed Hours")
        self.ax1.set_ylabel("Coverage Fraction")
        
        # Darkening Speed
        self.ax2.plot(self.df_fruit['elapsed_hours'], self.df_fruit['darkening_speed'], color='tab:orange')
        self.ax2.axvline(x=current_hours, color='red', linestyle='--')
        self.ax2.set_title("Darkening Speed Over Time")
        self.ax2.set_xlabel("Elapsed Hours")
        self.ax2.set_ylabel("Speed (/hr)")
        
        self.fig.tight_layout()
        self.canvas.draw()
        
    def update_ui(self):
        row = self.df_fruit.iloc[self.frame_idx]
        
        # Update metrics
        self.timestamp_lbl.config(text=f"Timestamp: {row['timestamp'].strftime('%Y-%m-%d %H:%M')}")
        self.metric_vars["Dark Coverage"].set(f"{row['dark_coverage']:.1%}")
        self.metric_vars["Green Ratio"].set(f"{row['green_ratio']:.3f}")
        self.metric_vars["Darkening Speed"].set(f"{row['darkening_speed']:.3f} /hr")
        self.metric_vars["Rapid Onset"].set("Yes" if row['rapid_darkening_onset_flag'] else "No")
        
        self.update_images()
        self.update_graphs(row['elapsed_hours'])
        
    def flag_frame(self):
        row = self.df_fruit.iloc[self.frame_idx]
        flag_file = PROJECT_ROOT / "data" / "02_processed" / "avocado" / "eda" / "flags" / "bad_segmentation_flags.csv"
        flag_file.parent.mkdir(parents=True, exist_ok=True)
        flag_data = pd.DataFrame([{
            "fruit_id": self.current_fruit,
            "timestamp": row['timestamp'],
            "image_path": row['image_path'],
            "threshold_l": self.threshold_l
        }])
        if os.path.exists(flag_file):
            flag_data.to_csv(flag_file, mode='a', header=False, index=False)
        else:
            flag_data.to_csv(flag_file, index=False)
        messagebox.showinfo("Success", "Frame flagged for bad segmentation!")

if __name__ == "__main__":
    df = load_data()
    if df is not None:
        root = tk.Tk()
        app = AvocadoApp(root, df)
        root.mainloop()
    else:
        print("Data not found. Run phase 3.2 first.")
