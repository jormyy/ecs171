import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
from tqdm import tqdm

# Directory setup
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, 'raw')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'output')
PLOT_DIR = os.path.join(OUTPUT_DIR, 'plots')
STATS_DIR = os.path.join(OUTPUT_DIR, 'stats')
CATEGORIES_DIR = os.path.join(PLOT_DIR, 'categories')
COMPARISON_DIR = os.path.join(PLOT_DIR, 'comparisons')

def create_directories():
    """Create necessary directories for output."""
    directories = [OUTPUT_DIR, PLOT_DIR, STATS_DIR, CATEGORIES_DIR, COMPARISON_DIR]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")

def load_ndjson(filepath, nrows=None):
    """Load NDJSON file into a DataFrame."""
    data = []
    with open(filepath, 'r') as f:
        for i, line in enumerate(f):
            if nrows is not None and i >= nrows:
                break
            data.append(json.loads(line))
    return pd.DataFrame(data)

def extract_drawing_features(drawing):
    """Extract features from drawing array."""
    n_strokes = len(drawing)
    total_points = sum(len(stroke[0]) for stroke in drawing)
    time_taken = max(stroke[2][-1] for stroke in drawing)
    
    return {
        'n_strokes': n_strokes,
        'total_points': total_points,
        'time_taken': time_taken,
    }

def analyze_single_category(filepath, category_name):
    """Analyze a single category and save its plots and statistics."""
    print(f"\nAnalyzing {category_name}...")
    
    # Load and process data
    df = load_ndjson(filepath)
    
    # Extract features
    print("Extracting features...")
    features = []
    for drawing in tqdm(df['drawing'], desc=f"Processing {category_name}"):
        features.append(extract_drawing_features(drawing))
    
    features_df = pd.DataFrame(features)
    df = pd.concat([df, features_df], axis=1)
    
    # Create category directory
    category_dir = os.path.join(CATEGORIES_DIR, category_name)
    os.makedirs(category_dir, exist_ok=True)
    
    # Generate plots
    print(f"Generating plots for {category_name}...")
    
    # 1. Strokes distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='n_strokes', bins=20)
    plt.title(f'Distribution of Stroke Count in {category_name.title()} Drawings')
    plt.xlabel('Number of strokes')
    plt.ylabel('Count')
    plt.savefig(os.path.join(category_dir, f'{category_name}_strokes_dist.png'))
    plt.close()
    
    # 2. Time distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='time_taken', bins=30)
    plt.title(f'Distribution of Drawing Times for {category_name.title()}')
    plt.xlabel('Time taken (ms)')
    plt.ylabel('Count')
    plt.savefig(os.path.join(category_dir, f'{category_name}_time_dist.png'))
    plt.close()
    
    # 3. Points distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='total_points', bins=30)
    plt.title(f'Distribution of Total Points in {category_name.title()} Drawings')
    plt.xlabel('Number of points')
    plt.ylabel('Count')
    plt.savefig(os.path.join(category_dir, f'{category_name}_points_dist.png'))
    plt.close()
    
    # Calculate and save statistics
    stats = {
        'sample_size': len(df),
        'strokes': {
            'mean': float(df['n_strokes'].mean()),
            'median': float(df['n_strokes'].median()),
            'std': float(df['n_strokes'].std())
        },
        'time': {
            'mean': float(df['time_taken'].mean()),
            'median': float(df['time_taken'].median()),
            'std': float(df['time_taken'].std())
        },
        'points': {
            'mean': float(df['total_points'].mean()),
            'median': float(df['total_points'].median()),
            'std': float(df['total_points'].std())
        },
        'recognition_rate': float(df['recognized'].mean() * 100)
    }
    
    with open(os.path.join(STATS_DIR, f'{category_name}_stats.json'), 'w') as f:
        json.dump(stats, f, indent=4)
    
    return stats

def create_comparison_plots(all_stats):
    """Create comparison plots across all categories."""
    print("\nGenerating comparison plots...")
    
    # Convert stats to DataFrame for easier plotting
    comparison_data = {
        category: {
            'mean_strokes': stats['strokes']['mean'],
            'mean_time': stats['time']['mean'],
            'mean_points': stats['points']['mean'],
            'recognition_rate': stats['recognition_rate']
        }
        for category, stats in all_stats.items()
    }
    df_comparison = pd.DataFrame(comparison_data).T
    
    # Plot comparisons
    metrics = [
        ('mean_time', 'Average Drawing Time (ms)'),
        ('mean_strokes', 'Average Number of Strokes'),
        ('mean_points', 'Average Number of Points'),
        ('recognition_rate', 'Recognition Rate (%)')
    ]
    
    for metric, title in metrics:
        plt.figure(figsize=(12, 6))
        df_comparison[metric].plot(kind='bar')
        plt.title(title + ' by Category')
        plt.xlabel('Category')
        plt.ylabel(title)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(COMPARISON_DIR, f'comparison_{metric}.png'))
        plt.close()
    
    # Save comparison statistics
    with open(os.path.join(STATS_DIR, 'comparison_stats.json'), 'w') as f:
        json.dump(comparison_data, f, indent=4)

def main():
    # Create directory structure
    create_directories()
    
    # Get all NDJSON files
    ndjson_files = [f for f in os.listdir(RAW_DATA_DIR) if f.endswith('.ndjson')]
    total_files = len(ndjson_files)
    
    print(f"\nStarting analysis of {total_files} NDJSON files...")
    
    # Analyze each category
    all_stats = {}
    for index, filename in enumerate(ndjson_files, 1):
        print(f"\nProcessing file {index}/{total_files}: {filename}")
        filepath = os.path.join(RAW_DATA_DIR, filename)
        category_name = os.path.splitext(filename)[0]
        
        try:
            stats = analyze_single_category(filepath, category_name)
            all_stats[category_name] = stats
            print(f"✓ Successfully processed {filename}")
            print(f"  Category: {category_name}")
            print(f"  Progress: {index}/{total_files} ({(index/total_files)*100:.1f}%)")
            
        except Exception as e:
            print(f"✗ Error processing {filename}: {str(e)}")
    
    print("\nCreating comparison plots...")
    create_comparison_plots(all_stats)
    
    print("\nAnalysis complete!")
    print(f"- Individual category analyses saved in: {CATEGORIES_DIR}")
    print(f"- Comparison plots saved in: {COMPARISON_DIR}")
    print(f"- Statistics saved in: {STATS_DIR}")

if __name__ == "__main__":
    main()
