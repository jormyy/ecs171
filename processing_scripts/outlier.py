import os
import json
import numpy as np
import pandas as pd

# Directory structure
RAW_DIR = "raw"
STATS_DIR = os.path.join("eda", "output", "stats")
PROCESSED_DIR = "processed_data"
OUTLIERS_DIR = os.path.join(PROCESSED_DIR, "outliers")
INLIERS_DIR = os.path.join(PROCESSED_DIR, "inliers")

def create_directories():
    """Create necessary directories if they don't exist."""
    for directory in [PROCESSED_DIR, OUTLIERS_DIR, INLIERS_DIR]:
        os.makedirs(directory, exist_ok=True)

def load_stats(category_name):
    """Load statistics for a given category from the stats directory."""
    stats_file = os.path.join(STATS_DIR, f"{category_name}_stats.json")
    try:
        with open(stats_file, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: No stats file found for {category_name}")
        return None

def get_drawing_time(drawing):
    """Calculate total drawing time from the timestamp arrays."""
    total_time = 0
    for stroke in drawing:
        if len(stroke[2]) > 0:  # Check if there are timestamps
            total_time = max(total_time, stroke[2][-1])  # Get the last timestamp
    return total_time

def is_outlier(drawing_data, stats):
    """
    Determine if a drawing is an outlier based on multiple criteria.
    Returns True if the drawing is an outlier.
    """
    # Get statistics for outlier detection
    stroke_mean = stats['strokes']['mean']
    stroke_std = stats['strokes']['std']
    time_mean = stats['time']['mean']
    time_std = stats['time']['std']
    
    # Calculate z-scores
    # Number of strokes is the length of the drawing array
    stroke_z = abs((len(drawing_data) - stroke_mean) / stroke_std)
    
    # Get total drawing time
    total_time = get_drawing_time(drawing_data)
    time_z = abs((total_time - time_mean) / time_std)
    
    # Define outlier thresholds (adjust these as needed)
    STROKE_THRESHOLD = 3  # 3 standard deviations
    TIME_THRESHOLD = 3
    
    return (stroke_z > STROKE_THRESHOLD) or (time_z > TIME_THRESHOLD)

def process_category(filename):
    """Process a single category file and split into outliers and inliers."""
    category_name = os.path.splitext(filename)[0]
    print(f"\nProcessing {category_name}...")
    
    # Load statistics
    stats = load_stats(category_name)
    if not stats:
        print(f"Skipping {category_name} due to missing statistics")
        return
    
    # Initialize output files
    inlier_file = os.path.join(INLIERS_DIR, filename)
    outlier_file = os.path.join(OUTLIERS_DIR, filename)
    
    # Process the file
    outlier_count = 0
    inlier_count = 0
    
    with open(os.path.join(RAW_DIR, filename), 'r') as f_in, \
         open(inlier_file, 'w') as f_inlier, \
         open(outlier_file, 'w') as f_outlier:
        
        for line in f_in:
            data = json.loads(line.strip())
            drawing = data['drawing']
            
            if is_outlier(drawing, stats):
                f_outlier.write(line)
                outlier_count += 1
            else:
                f_inlier.write(line)
                inlier_count += 1
    
    print(f"✓ {category_name} processed:")
    print(f"  - Inliers: {inlier_count}")
    print(f"  - Outliers: {outlier_count}")
    print(f"  - Outlier percentage: {(outlier_count/(outlier_count+inlier_count)*100):.1f}%")
    
    return {
        'category': category_name,
        'inliers': inlier_count,
        'outliers': outlier_count
    }

def main():
    # Create directory structure
    create_directories()
    
    # Get all NDJSON files
    ndjson_files = [f for f in os.listdir(RAW_DIR) if f.endswith('.ndjson')]
    total_files = len(ndjson_files)
    
    print(f"Starting outlier detection for {total_files} categories...")
    
    # Process each category
    results = []
    for index, filename in enumerate(ndjson_files, 1):
        print(f"\nFile {index}/{total_files}")
        result = process_category(filename)
        if result:
            results.append(result)
    
    # Save summary
    summary_file = os.path.join(PROCESSED_DIR, 'processing_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nProcessing complete!")
    print(f"- Inlier datasets saved in: {INLIERS_DIR}")
    print(f"- Outlier datasets saved in: {OUTLIERS_DIR}")
    print(f"- Summary saved in: {os.path.join(PROCESSED_DIR, 'processing_summary.json')}")

if __name__ == "__main__":
    main()
