import os
import json
import glob
import shutil
import cv2
import numpy as np
from tqdm import tqdm
import random
from typing import Dict, List, Tuple
from datetime import datetime


def create_coco_directory_structure(output_dir: str) -> Dict[str, str]:
    """
    Create the basic COCO directory structure.

    Args:
        output_dir: base directory for the COCO dataset

    Returns:
        Dictionary with paths to directories
    """

    annotations_dir = os.path.join(output_dir, "annotations")
    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "val")

    os.makedirs(annotations_dir, exist_ok=True)
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    paths = {
        "annotations_dir": annotations_dir,
        "train_dir": train_dir,
        "val_dir": val_dir,
    }

    return paths


def create_base_coco_structure() -> Dict:
    """Create base COCO JSON structure."""
    return {
        "info": {
            "description": "Dataset converted from labelme format to COCO format",
            "url": "",
            "version": "1.0",
            "year": datetime.now().year,
            "contributor": "",
            "date_created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        },
        "licenses": [{"id": 1, "name": "Unknown License", "url": ""}],
        "images": [],
        "annotations": [],
        "categories": [],
    }


def collect_categories(
        json_files: List[str]) -> Tuple[Dict[str, int], List[Dict]]:
    """
    Collect all category names from labelme files and assign IDs.

    Args:
        json_files: List of paths to labelme JSON files

    Returns:
        Tuple of (category_id_map, categories_list)
    """
    category_id_map = {}
    categories = []
    category_id = 1

    print("Collecting categories...")
    for json_file in tqdm(json_files):
        with open(json_file, "r") as f:
            labelme_data = json.load(f)

        for shape in labelme_data.get("shapes", []):
            label = shape.get("label")
            if label and label not in category_id_map:
                category_id_map[label] = category_id
                categories.append({
                    "supercategory": "none",
                    "id": category_id,
                    "name": label,
                })
                category_id += 1

    return category_id_map, categories


def calculate_polygon_area(points):
    """Calculate the area of a polygon using the Shoelace formula."""
    x = [p[0] for p in points]
    y = [p[1] for p in points]

    return 0.5 * abs(
        sum(x[i] * y[i + 1] - x[i + 1] * y[i]
            for i in range(len(points) - 1)) + x[-1] * y[0] - x[0] * y[-1])


def process_json_files(
    json_files: List[str],
    labelme_dir: str,
    image_dir: str,     # TODO: pass output dir instead, so it can be use for file_name
    category_id_map: Dict[str, int],
    start_ann_id: int = 1,
) -> Dict:
    """
    Process JSON files and copy images to the output directory.

    Args:
        json_files: List of JSON files to process
        labelme_dir: Directory containing labelme files
        image_dir: Directory to save images (train or val)
        category_id_map: Mapping of category names to IDs
        start_ann_id: Starting annotation ID

    Returns:
        COCO data dictionary with images and annotations
    """

    coco_data = create_base_coco_structure()
    coco_data["categories"] = [{
        "supercategory": "none",    # TODO: correctly support supercategories
        "id": cat_id,
        "name": cat_name
    } for cat_name, cat_id in category_id_map.items()]

    category_coco_data = {
        cat_name: create_base_coco_structure()
        for cat_name in category_id_map.keys()
    }

    for cat_name, cat_id in category_id_map.items():
        category_coco_data[cat_name]["categories"] = [{
            "supercategory": "none",
            "id": cat_id,
            "name": cat_name
        }]

    ann_id = start_ann_id

    category_images = {cat_name: set() for cat_name in category_id_map.keys()}

    print(f"Processing {len(json_files)} files...")
    for img_id, json_file in enumerate(tqdm(json_files), 1):
        with open(json_file, "r") as f:
            labelme_data = json.load(f)

        image_path = labelme_data.get("imagePath", "")
        if not image_path:
            base_filename = os.path.splitext(os.path.basename(json_file))[0]
            for ext in [".jpg", ".jpeg", ".png"]:
                potential_path = os.path.join(labelme_dir, base_filename + ext)
                if os.path.exists(potential_path):
                    image_path = base_filename + ext
                    break

        source_image = os.path.join(labelme_dir, image_path)
        if not os.path.exists(source_image):
            print(f"Warning: Image file not found for {json_file}")
            continue

        original_filename = os.path.basename(image_path)
        target_image = os.path.join(image_dir, original_filename)

        shutil.copy(source_image, target_image)

        relative_path = os.path.join(os.path.basename(image_dir),
                                     original_filename)

        if "imageWidth" in labelme_data and "imageHeight" in labelme_data:
            img_width = labelme_data["imageWidth"]
            img_height = labelme_data["imageHeight"]
        else:
            img = cv2.imread(source_image)
            img_height, img_width = img.shape[:2]

        image_info = {
            "id": img_id,
            "width": img_width,
            "height": img_height,
            "file_name": relative_path,
            "license": 1,
            "flickr_url": "",
            "coco_url": "",
            "date_captured": "",
        }

        coco_data["images"].append(image_info)

        image_categories = set()

        for shape in labelme_data.get("shapes", []):
            label = shape.get("label")
            shape_type = shape.get("shape_type")
            points = shape.get("points", [])

            if label not in category_id_map:
                continue

            image_categories.add(label)

            if shape_type == "polygon":
                segmentation = [coord for point in points for coord in point]

                x_coords = [point[0] for point in points]
                y_coords = [point[1] for point in points]
                x_min, y_min = min(x_coords), min(y_coords)
                width = max(x_coords) - x_min
                height = max(y_coords) - y_min

                area = calculate_polygon_area(points)

            elif shape_type == "rectangle":
                x1, y1 = points[0]
                x2, y2 = points[1]
                x_min, y_min = min(x1, x2), min(y1, y2)
                width, height = abs(x2 - x1), abs(y2 - y1)

                area = width * height

                segmentation = [
                    x_min,
                    y_min,
                    x_min + width,
                    y_min,
                    x_min + width,
                    y_min + height,
                    x_min,
                    y_min + height,
                ]

            elif shape_type == "circle":
                center_x, center_y = points[0]
                radius_point_x, radius_point_y = points[1]
                radius = np.sqrt((radius_point_x - center_x)**2 +
                                 (radius_point_y - center_y)**2)

                num_points = 36
                polygon_points = []
                for i in range(num_points):
                    angle = 2 * np.pi * i / num_points
                    x = center_x + radius * np.cos(angle)
                    y = center_y + radius * np.sin(angle)
                    polygon_points.extend([x, y])

                x_min, y_min = center_x - radius, center_y - radius
                width, height = 2 * radius, 2 * radius

                area = np.pi * (radius**2)
                segmentation = polygon_points

            else:
                # Skip unsupported shape types
                print(f"Warning: Shape type '{shape_type}' not supported. Skipping...")
                continue

            annotation = {
                "id": ann_id,
                "image_id": img_id,
                "category_id": category_id_map[label],
                "segmentation": [segmentation],
                "area": area,
                "bbox": [x_min, y_min, width, height],
                "iscrowd": 0,
            }

            coco_data["annotations"].append(annotation)

            category_coco_data[label]["annotations"].append(annotation.copy())

            ann_id += 1

        for cat_name in image_categories:
            category_images[cat_name].add(img_id)
            category_coco_data[cat_name]["images"].append(image_info.copy())

    return coco_data, category_coco_data, category_images


def convert_labelme_to_coco(labelme_dir: str,
                            output_dir: str,
                            train_val_ratio: float = 0.8) -> None:
    """
    Convert labelme annotations to COCO format with train/val split.

    Args:
        labelme_dir: Directory containing labelme JSON files
        output_dir: Directory to save the COCO dataset
        train_val_ratio: Ratio of data to use for training (0.0-1.0)
    """
    if train_val_ratio <= 0.0 or train_val_ratio >= 1.0:
        raise ValueError("train_val_ratio must be between 0.0 and 1.0")

    paths = create_coco_directory_structure(output_dir)

    json_files = glob.glob(os.path.join(labelme_dir, "*.json"))
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in {labelme_dir}")

    print(f"Found {len(json_files)} labelme JSON files")

    category_id_map, categories = collect_categories(json_files)
    print(
        f"Found {len(category_id_map)} categories: {', '.join(category_id_map.keys())}"
    )

    random.shuffle(json_files)
    split_index = int(len(json_files) * train_val_ratio)
    train_files = json_files[:split_index]
    val_files = json_files[split_index:]

    print("Processing training set...")
    train_data, train_category_data, train_category_images = process_json_files(
        train_files, labelme_dir, paths["train_dir"], category_id_map)

    print("Processing validation set...")
    val_data, val_category_data, val_category_images = process_json_files(
        val_files,
        labelme_dir,
        paths["val_dir"],
        category_id_map,
        start_ann_id=len(train_data["annotations"]) + 1,
    )

    annotations_dir = paths["annotations_dir"]

    print("Saving annotation files...")

    with open(os.path.join(annotations_dir, "instances_train.json"), "w") as f:
        json.dump(train_data, f, indent=2)

    with open(os.path.join(annotations_dir, "instances_val.json"), "w") as f:
        json.dump(val_data, f, indent=2)

    for category in category_id_map.keys():
        clean_cat = category.replace(" ", "_").lower()

        train_cat_file = os.path.join(annotations_dir,
                                      f"{clean_cat}_train.json")
        with open(train_cat_file, "w") as f:
            json.dump(train_category_data[category], f, indent=2)

        val_cat_file = os.path.join(annotations_dir, f"{clean_cat}_val.json")
        with open(val_cat_file, "w") as f:
            json.dump(val_category_data[category], f, indent=2)

    print(f"Conversion complete!")
    print(
        f"Train set: {len(train_data['images'])} images, {len(train_data['annotations'])} annotations"
    )
    print(
        f"Val set: {len(val_data['images'])} images, {len(val_data['annotations'])} annotations"
    )

    print("\nCategory statistics:")
    for cat_name in category_id_map.keys():
        train_count = len(train_category_data[cat_name]["annotations"])
        val_count = len(val_category_data[cat_name]["annotations"])
        train_img_count = len(train_category_images[cat_name])
        val_img_count = len(val_category_images[cat_name])
        print(
            f"  - {cat_name}: {train_count} train annotations in {train_img_count} images, "
            f"{val_count} val annotations in {val_img_count} images")

    print(f"\nCOCO dataset created at {output_dir}")
