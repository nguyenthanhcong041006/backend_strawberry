import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[3]

def create_output_dir(base_dir, subdir):
    path = os.path.join(base_dir, subdir)
    os.makedirs(path, exist_ok=True)
    return path

def save_plot(fig, output_dir, filename, df_subset=None):
    """Saves plot in PNG and SVG, and saves subset data"""
    png_path = os.path.join(output_dir, f"{filename}.png")
    svg_path = os.path.join(output_dir, f"{filename}.svg")
    csv_path = os.path.join(output_dir, f"{filename}_data.csv")
    
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    fig.savefig(svg_path, format='svg', bbox_inches='tight')
    
    if df_subset is not None:
        df_subset.to_csv(csv_path, index=False)
        
    plt.close(fig)

def plot_time_series(df, fruit_id, metric, y_label, title, output_dir, filename):
    fig, ax = plt.subplots(figsize=(10, 5))
    df_fruit = df[df['fruit_id'] == fruit_id].copy()
    
    if len(df_fruit) == 0:
        return
        
    # X-axis will be elapsed hours from first frame
    start_time = df_fruit['timestamp'].min()
    df_fruit['elapsed_hours'] = (df_fruit['timestamp'] - start_time).dt.total_seconds() / 3600.0
    
    sns.lineplot(data=df_fruit, x='elapsed_hours', y=metric, ax=ax, label=metric)
    
    # EOL Marker (assumed last frame)
    eol_hour = df_fruit['elapsed_hours'].max()
    ax.axvline(x=eol_hour, color='red', linestyle='--', label='Assumed EOL')
    
    ax.set_title(title)
    ax.set_xlabel('Elapsed Time (Hours)')
    ax.set_ylabel(y_label)
    ax.legend()
    
    save_plot(fig, output_dir, f"{fruit_id}_{filename}", df_fruit[['timestamp', 'elapsed_hours', metric]])

def plot_dark_coverage_comparison(df, fruit_id, output_dir):
    fig, ax = plt.subplots(figsize=(10, 5))
    df_fruit = df[df['fruit_id'] == fruit_id].copy()
    
    if len(df_fruit) == 0:
        return
        
    start_time = df_fruit['timestamp'].min()
    df_fruit['elapsed_hours'] = (df_fruit['timestamp'] - start_time).dt.total_seconds() / 3600.0
    
    sns.lineplot(data=df_fruit, x='elapsed_hours', y='dark_coverage', ax=ax, label='Raw Dark Coverage', alpha=0.5)
    sns.lineplot(data=df_fruit, x='elapsed_hours', y='smoothed_dark_coverage', ax=ax, label='Smoothed Dark Coverage', linewidth=2)
    
    eol_hour = df_fruit['elapsed_hours'].max()
    ax.axvline(x=eol_hour, color='red', linestyle='--', label='Assumed EOL')
    
    ax.set_title(f"Dark Coverage vs Time for {fruit_id}")
    ax.set_xlabel('Elapsed Time (Hours)')
    ax.set_ylabel('Dark Coverage Fraction')
    ax.legend()
    
    save_plot(fig, output_dir, f"{fruit_id}_dark_coverage_comparison", df_fruit[['elapsed_hours', 'dark_coverage', 'smoothed_dark_coverage']])

def create_image_montage(df, fruit_id, output_dir):
    """Creates a montage of early, mid, near-EOL, and EOL stages"""
    df_fruit = df[df['fruit_id'] == fruit_id].sort_values('timestamp').reset_index(drop=True)
    if len(df_fruit) < 4:
        return
        
    indices = [
        0, # Early
        len(df_fruit) // 3, # Mid
        int(len(df_fruit) * 0.8), # Near-EOL
        len(df_fruit) - 1 # EOL
    ]
    
    labels = ["Early", "Middle", "Near-EOL", "EOL"]
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    for idx, ax, label in zip(indices, axes, labels):
        row = df_fruit.iloc[idx]
        img_path = row['image_path']
        if os.path.exists(img_path):
            # Load alpha mask correctly
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            if img.shape[2] == 4:
                # Convert BGRA to RGBA for matplotlib
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
            ax.imshow(img)
            ax.set_title(f"{label}\n{row['timestamp'].strftime('%Y-%m-%d %H:%M')}")
        ax.axis('off')
        
    plt.suptitle(f"Visual Deterioration Stages: {fruit_id}")
    
    png_path = os.path.join(output_dir, f"{fruit_id}_montage.png")
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def main():
    print("--- Avocado Stage 2 EDA: Graphing ---")
    
    csv_path = PROJECT_ROOT / "data" / "02_processed" / "avocado" / "eda" / "features" / "avocado_features.csv"
    
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found. Run avocado stage2 EDA feature extraction first.")
        return
        
    df = pd.read_csv(csv_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Create output directories
    graphs_dir = PROJECT_ROOT / "data" / "02_processed" / "avocado" / "eda" / "graphs"
    time_series_dir = create_output_dir(graphs_dir, "time_series")
    montage_dir = create_output_dir(graphs_dir, "montages")
    comparison_dir = create_output_dir(graphs_dir, "comparisons")
    
    fruit_ids = df['fruit_id'].unique()
    
    sns.set_theme(style="whitegrid")
    
    print(f"Generating graphs for {len(fruit_ids)} fruits...")
    
    for fruit in fruit_ids:
        print(f"  Processing {fruit}...")
        
        # 1. Dark coverage vs time (Smoothed vs Raw)
        plot_dark_coverage_comparison(df, fruit, time_series_dir)
        
        # 2. Darkening speed vs time
        plot_time_series(df, fruit, 'darkening_speed', 'Darkening Speed (fraction/hour)', 
                         f'Darkening Speed vs Time for {fruit}', time_series_dir, 'darkening_speed')
                         
        # 3. Lightness vs time
        plot_time_series(df, fruit, 'mean_lab_l', 'Mean Lightness (L*)', 
                         f'Mean Lightness vs Time for {fruit}', time_series_dir, 'lightness')
                         
        # 4. Excess green vs time
        plot_time_series(df, fruit, 'excess_green', 'Excess Green', 
                         f'Excess Green vs Time for {fruit}', time_series_dir, 'excess_green')
                         
        # 5. Largest dark spot vs time
        plot_time_series(df, fruit, 'largest_dark_spot_growth', 'Largest Dark Spot Growth', 
                         f'Largest Spot Growth vs Time for {fruit}', time_series_dir, 'largest_spot_growth')
                         
        # 6. Mask area vs time (QA)
        plot_time_series(df, fruit, 'mask_area', 'Mask Area (pixels)', 
                         f'Mask Area vs Time (Segmentation QA) for {fruit}', time_series_dir, 'mask_area')
                         
        # 7. Image Montage
        create_image_montage(df, fruit, montage_dir)
        
    # Cross-fruit EOL Comparison
    print("Generating cross-fruit comparison...")
    eol_rows = df.sort_values('timestamp').groupby('fruit_id').tail(1)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=eol_rows, x='fruit_id', y='dark_coverage', ax=ax, palette='viridis')
    ax.set_title("EOL Dark Coverage Comparison Across Fruits")
    ax.set_ylabel("Final Dark Coverage Fraction")
    save_plot(fig, comparison_dir, "cross_fruit_eol_dark_coverage", eol_rows[['fruit_id', 'dark_coverage']])
    
    print(f"EDA Graphing completed! Output saved to: {graphs_dir}")

if __name__ == "__main__":
    main()
