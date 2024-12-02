import os
import json
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

# Directory structure
PROCESSED_DIR = "processed_data"
INLIERS_DIR = os.path.join(PROCESSED_DIR, "inliers")
OUTLIERS_DIR = os.path.join(PROCESSED_DIR, "outliers")
IMAGE_DIR = os.path.join(PROCESSED_DIR, "images")

# Image parameters
CANVAS_SIZE = 256
STROKE_WIDTH = 2
BG_COLOR = 255
FG_COLOR = 0

# Skip these categories that are already processed
SKIP_CATEGORIES = {}

def create_directories():
    """Create necessary directories if they don't exist."""
    for split in ["inliers", "outliers"]:
        base_dir = os.path.join(IMAGE_DIR, split)
        os.makedirs(base_dir, exist_ok=True)

def normalize_strokes(drawing):
    """Normalize stroke coordinates to fit within canvas while maintaining aspect ratio."""
    all_coords = []
    for stroke in drawing:
        for x, y in zip(stroke[0], stroke[1]):
            all_coords.append((x, y))
    
    all_coords = np.array(all_coords)
    min_x, min_y = all_coords.min(axis=0)
    max_x, max_y = all_coords.max(axis=0)
    
    width = max_x - min_x
    height = max_y - min_y
    
    if width == 0 and height == 0:
        return drawing
    
    scale = (CANVAS_SIZE - STROKE_WIDTH * 4) / max(width, height)
    
    center_x = (max_x + min_x) / 2
    center_y = (max_y + min_y) / 2
    
    normalized_drawing = []
    for stroke in drawing:
        x_coords = np.array(stroke[0])
        y_coords = np.array(stroke[1])
        
        x_norm = (x_coords - center_x) * scale + CANVAS_SIZE / 2
        y_norm = (y_coords - center_y) * scale + CANVAS_SIZE / 2
        
        normalized_drawing.append([x_norm.tolist(), y_norm.tolist(), stroke[2]])
    
    return normalized_drawing

def draw_strokes(drawing):
    """Convert strokes to an image."""
    image = Image.new('L', (CANVAS_SIZE, CANVAS_SIZE), color=BG_COLOR)
    draw = ImageDraw.Draw(image)
    
    drawing = normalize_strokes(drawing)
    
    for stroke in drawing:
        points = list(zip(stroke[0], stroke[1]))
        if len(points) > 1:
            draw.line(points, fill=FG_COLOR, width=STROKE_WIDTH, joint="curve")
    
    return image

def process_file(filepath, category, is_inlier=True):
    """Process a single NDJSON file and convert drawings to images."""
    if category in SKIP_CATEGORIES:
        print(f"Skipping {category} as it's already processed")
        return 0
        
    split = "inliers" if is_inlier else "outliers"
    output_dir = os.path.join(IMAGE_DIR, split, category)
    os.makedirs(output_dir, exist_ok=True)
    
    count = 0
    dataset_type = "inlier" if is_inlier else "outlier"
    print(f"\nProcessing {category} ({dataset_type}s)...")
    
    with open(filepath, 'r') as f:
        total_lines = sum(1 for _ in f)
    
    with open(filepath, 'r') as f:
        for line in tqdm(f, total=total_lines, desc=f"{category} ({dataset_type}s)"):
            data = json.loads(line)
            drawing = data['drawing']
            
            image = draw_strokes(drawing)
            image_path = os.path.join(output_dir, f"{category}_{count}.png")
            image.save(image_path)
            count += 1
    
    return count

def main():
    create_directories()
    
    inlier_total = 0
    outlier_total = 0
    
    # Process inliers
    inlier_files = [f for f in os.listdir(INLIERS_DIR) if f.endswith('.ndjson')]
    for filename in inlier_files:
        category = os.path.splitext(filename)[0]
        filepath = os.path.join(INLIERS_DIR, filename)
        count = process_file(filepath, category, is_inlier=True)
        inlier_total += count
        if count > 0:
            print(f"Created {count} inlier images for {category}")
    
    # Process outliers
    outlier_files = [f for f in os.listdir(OUTLIERS_DIR) if f.endswith('.ndjson')]
    for filename in outlier_files:
        category = os.path.splitext(filename)[0]
        filepath = os.path.join(OUTLIERS_DIR, filename)
        count = process_file(filepath, category, is_inlier=False)
        outlier_total += count
        if count > 0:
            print(f"Created {count} outlier images for {category}")
    
    print("\nConversion complete!")
    print(f"Total new inlier images created: {inlier_total}")
    print(f"Total new outlier images created: {outlier_total}")
    print(f"Total new images created: {inlier_total + outlier_total}")
    print(f"Images saved in: {IMAGE_DIR}")

if __name__ == "__main__":
    main()
