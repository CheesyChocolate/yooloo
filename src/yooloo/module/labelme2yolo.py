import os
import json
import shutil
import random
import numpy as np
from glob import glob
from typing import List, Dict, Any, Optional


def convert_labelme_to_yolov8(
    labelme_dir: str,
    output_dir: str,
    categories: Optional[List[str]] = None,
    train_ratio: float = 0.8,
    copy_images: bool = True,
    random_seed: int = 42,
) -> Dict[str, Any]:
    """
    Convert LabelMe annotation format to YOLOv8 format

    Args:
        labelme_dir (str): Directory containing LabelMe JSON files
        output_dir (str): Directory where YOLOv8 format dataset will be saved
        categories (list, optional): List of category names. If None, categories will be extracted from LabelMe files
        train_ratio (float): Ratio of images to use for training (0.0 to 1.0)
        copy_images (bool): Whether to copy images to the output directory
        random_seed (int): Random seed for reproducibility

    Returns:
        Dict[str, Any]: Statistics about the conversion
    """
    random.seed(random_seed)
    np.random.seed(random_seed)

    # Create YOLOv8 directory structure
    os.makedirs(os.path.join(output_dir, "images", "train"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "images", "val"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "labels", "train"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "labels", "val"), exist_ok=True)

    # Find all labelme JSON files
    labelme_json_files = sorted(
        glob(os.path.join(labelme_dir, "**/*.json"), recursive=True))
    if len(labelme_json_files) == 0:
        print(f"No JSON files found in {labelme_dir}")
        return {"error": "No JSON files found"}

    # If no categories provided, collect them from the files
    if not categories:
        categories_set = set()
        for labelme_file in labelme_json_files:
            with open(labelme_file, "r") as f:
                labelme_data = json.load(f)
                for shape in labelme_data.get("shapes", []):
                    categories_set.add(shape["label"])
        categories = sorted(list(categories_set))

    # Create category ID map (YOLO format uses indices starting from 0)
    category_id_map = {cat: i for i, cat in enumerate(categories)}

    # Create yaml config file for YOLOv8
    yaml_path = os.path.join(output_dir, "dataset.yaml")
    with open(yaml_path, "w") as f:
        yaml_content = {
            "path": os.path.abspath(output_dir),
            "train": "images/train",
            "val": "images/val",
            "nc": len(categories),
            "names": categories,
        }
        f.write("# YOLOv8 Dataset Configuration\n")
        f.write(f"path: {yaml_content['path']}\n")
        f.write(f"train: {yaml_content['train']}\n")
        f.write(f"val: {yaml_content['val']}\n")
        f.write(f"nc: {yaml_content['nc']}\n")
        f.write("names:\n")
        for cat in categories:
            f.write(f"  - '{cat}'\n")

    # Randomly split files into train and val sets
    random.shuffle(labelme_json_files)
    split_idx = int(len(labelme_json_files) * train_ratio)
    train_files = labelme_json_files[:split_idx]
    val_files = labelme_json_files[split_idx:]

    # Statistics for tracking
    stats = {
        "total_files": len(labelme_json_files),
        "train_files": len(train_files),
        "val_files": len(val_files),
        "categories": categories,
        "annotations_count": 0,
        "skipped_files": 0,
        "skipped_annotations": 0,
    }

    # Process each set
    for set_name, file_list in [("train", train_files), ("val", val_files)]:
        for labelme_file in file_list:
            with open(labelme_file, "r") as f:
                try:
                    labelme_data = json.load(f)
                except json.JSONDecodeError:
                    print(f"Error: Could not parse JSON file: {labelme_file}")
                    stats["skipped_files"] += 1
                    continue

            # Get image info
            if "imagePath" in labelme_data and labelme_data["imagePath"]:
                image_path = labelme_data["imagePath"]
                # Handle relative paths
                if not os.path.isabs(image_path):
                    # Try to find relative to the JSON file
                    image_path = os.path.join(os.path.dirname(labelme_file),
                                              image_path)

                if not os.path.exists(image_path):
                    # Try to find the image with different extensions
                    base_name = os.path.splitext(image_path)[0]
                    for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
                        if os.path.exists(base_name + ext):
                            image_path = base_name + ext
                            break
            else:
                # Try to derive image path from JSON path
                potential_image_base = os.path.splitext(labelme_file)[0]
                for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
                    if os.path.exists(potential_image_base + ext):
                        image_path = potential_image_base + ext
                        break
                else:
                    print(f"Warning: Could not find image for {labelme_file}")
                    stats["skipped_files"] += 1
                    continue

            # Get image dimensions
            image_width = labelme_data.get("imageWidth", 0)
            image_height = labelme_data.get("imageHeight", 0)

            if image_width <= 0 or image_height <= 0:
                print(f"Warning: Invalid image dimensions for {labelme_file}")
                stats["skipped_files"] += 1
                continue

            # Get image filename for output
            image_filename = os.path.basename(image_path)
            output_image_path = os.path.join(output_dir, "images", set_name,
                                             image_filename)
            output_label_path = os.path.join(
                output_dir,
                "labels",
                set_name,
                os.path.splitext(image_filename)[0] + ".txt",
            )

            # Copy or link the image
            if copy_images:
                try:
                    shutil.copy2(image_path, output_image_path)
                except Exception as e:
                    print(f"Error copying image {image_path}: {str(e)}")
                    stats["skipped_files"] += 1
                    continue
            else:
                try:
                    rel_path = os.path.relpath(
                        image_path, os.path.dirname(output_image_path))
                    if os.path.exists(output_image_path):
                        os.remove(output_image_path)
                    os.symlink(rel_path, output_image_path)
                except Exception as e:
                    print(f"Error linking image {image_path}: {str(e)}")
                    try:
                        shutil.copy2(image_path, output_image_path)
                    except Exception as e:
                        print(f"Error copying image {image_path}: {str(e)}")
                        stats["skipped_files"] += 1
                        continue

            # Process annotations and write to YOLO format
            yolo_lines = []

            for shape in labelme_data.get("shapes", []):
                label = shape.get("label", "")
                shape_type = shape.get("shape_type", "")
                points = shape.get("points", [])

                # Skip if the category is not in our list
                if label not in category_id_map:
                    print(
                        f"Warning: Label '{label}' not found in categories list. Skipping..."
                    )
                    stats["skipped_annotations"] += 1
                    continue

                # Get category ID (YOLO class index)
                class_id = category_id_map[label]

                # Convert coordinates based on shape type
                if shape_type == "rectangle":
                    # Rectangle: points contains [top-left, bottom-right]
                    x1, y1 = points[0]
                    x2, y2 = points[1]

                    # Ensure correct order
                    x_min, x_max = min(x1, x2), max(x1, x2)
                    y_min, y_max = min(y1, y2), max(y1, y2)

                    # Convert to YOLO format (center_x, center_y, width, height)
                    width = x_max - x_min
                    height = y_max - y_min
                    center_x = x_min + width / 2
                    center_y = y_min + height / 2

                    # Normalize coordinates
                    center_x /= image_width
                    center_y /= image_height
                    width /= image_width
                    height /= image_height

                    # Add to YOLO annotations
                    yolo_lines.append(
                        f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}"
                    )
                    stats["annotations_count"] += 1

                elif shape_type == "polygon":
                    # For polygons, compute bounding box
                    x_coords = [p[0] for p in points]
                    y_coords = [p[1] for p in points]

                    x_min, x_max = min(x_coords), max(x_coords)
                    y_min, y_max = min(y_coords), max(y_coords)

                    # Convert to YOLO format (center_x, center_y, width, height)
                    width = x_max - x_min
                    height = y_max - y_min
                    center_x = x_min + width / 2
                    center_y = y_min + height / 2

                    # Normalize coordinates
                    center_x /= image_width
                    center_y /= image_height
                    width /= image_width
                    height /= image_height

                    # Add to YOLO annotations
                    yolo_lines.append(
                        f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}"
                    )
                    stats["annotations_count"] += 1

                elif shape_type == "circle":
                    # Circle: points contains [center, point on perimeter]
                    center_x, center_y = points[0]
                    radius_point_x, radius_point_y = points[1]

                    # Calculate radius
                    radius = np.sqrt((center_x - radius_point_x)**2 +
                                     (center_y - radius_point_y)**2)

                    # Convert to bounding box
                    x_min = center_x - radius
                    y_min = center_y - radius
                    width = 2 * radius
                    height = 2 * radius

                    # Normalize coordinates
                    center_x /= image_width
                    center_y /= image_height
                    width /= image_width
                    height /= image_height

                    # Add to YOLO annotations
                    yolo_lines.append(
                        f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}"
                    )
                    stats["annotations_count"] += 1

                else:
                    # Skip unsupported shape types
                    print(
                        f"Warning: Shape type '{shape_type}' not supported. Skipping..."
                    )
                    stats["skipped_annotations"] += 1
                    continue

            # Write annotations to file
            with open(output_label_path, "w") as f:
                for line in yolo_lines:
                    f.write(line + "\n")

    # Print summary
    print(
        f"Conversion complete: {len(labelme_json_files)} LabelMe files processed"
    )
    print(f"Train set: {stats['train_files']} images")
    print(f"Validation set: {stats['val_files']} images")
    print(f"Total annotations: {stats['annotations_count']}")
    print(f"Categories: {', '.join(categories)}")

    return stats
