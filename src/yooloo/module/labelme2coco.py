import os
import json
import numpy as np
from glob import glob
from datetime import datetime
from typing import List, Optional, Dict, Any


def convert_labelme_to_coco(
    labelme_dir: str, coco_dir: str, categories: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Convert LabelMe annotation format to COCO annotation format

    Args:
        labelme_dir (str): Directory containing LabelMe JSON files
        coco_dir (str): Directory where COCO format JSON will be saved
        categories (list, optional): List of category names. If None, categories will be extracted from LabelMe files

    Returns:
        Dict[str, Any]: The COCO format data structure that was saved
    """
    # Create coco directory if it doesn't exist
    os.makedirs(coco_dir, exist_ok=True)

    # Initialize COCO format structure
    coco_format = {
        "info": {
            "description": "Dataset converted from LabelMe format to COCO format",
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

    # Find all labelme JSON files
    labelme_json_files = sorted(glob(os.path.join(labelme_dir, "*.json")))
    if len(labelme_json_files) == 0:
        print(f"No JSON files found in {labelme_dir}")
        return coco_format

    # If no categories provided, collect them from the files
    if not categories:
        categories_set = set()
        for labelme_file in labelme_json_files:
            with open(labelme_file, "r") as f:
                labelme_data = json.load(f)
                for shape in labelme_data.get("shapes", []):
                    categories_set.add(shape["label"])
        categories = sorted(list(categories_set))

    # Create category entries for COCO format
    for i, category_name in enumerate(categories, 1):
        coco_format["categories"].append(
            {"id": i, "name": category_name, "supercategory": "none"}
        )

    # Create category ID lookup
    category_id_map = {cat["name"]: cat["id"] for cat in coco_format["categories"]}

    # Process each LabelMe file
    image_id = 1
    annotation_id = 1

    for labelme_file in labelme_json_files:
        with open(labelme_file, "r") as f:
            labelme_data = json.load(f)

        # Get image info
        image_filename = os.path.basename(
            labelme_data.get("imagePath", labelme_file.replace(".json", ".jpg"))
        )
        image_filename = os.path.join(labelme_dir, image_filename)
        image_width = labelme_data.get("imageWidth", 0)
        image_height = labelme_data.get("imageHeight", 0)

        # Check if image dimensions are available directly
        if image_width == 0 or image_height == 0:
            if "imageData" in labelme_data and labelme_data["imageData"] is not None:
                # Could extract dimensions from imageData if needed
                pass
            else:
                print(
                    f"Warning: Could not determine image dimensions for {image_filename}"
                )

        # Create image entry
        coco_format["images"].append(
            {
                "id": image_id,
                "file_name": image_filename,
                "width": image_width,
                "height": image_height,
                "date_captured": "",
                "license": 1,
                "coco_url": "",
                "flickr_url": "",
            }
        )

        # Process annotations
        for shape in labelme_data.get("shapes", []):
            label = shape.get("label", "")
            shape_type = shape.get("shape_type", "polygon")
            points = shape.get("points", [])

            # Skip if the category is not in our list
            if label not in category_id_map:
                print(
                    f"Warning: Label '{label}' not found in categories list. Skipping..."
                )
                continue

            # Convert to COCO format
            if shape_type == "polygon":
                # Flatten points for COCO format
                segmentation = [np.array(points).flatten().tolist()]

                # Calculate bounding box
                x_coords = [p[0] for p in points]
                y_coords = [p[1] for p in points]
                x_min = min(x_coords)
                y_min = min(y_coords)
                width = max(x_coords) - x_min
                height = max(y_coords) - y_min

                # Calculate area using shoelace formula (polygon area)
                area = 0.0
                for i in range(len(points)):
                    j = (i + 1) % len(points)
                    area += points[i][0] * points[j][1]
                    area -= points[j][0] * points[i][1]
                area = abs(area) / 2.0

            elif shape_type == "rectangle":
                # Rectangle: points contains [top-left, bottom-right]
                x1, y1 = points[0]
                x2, y2 = points[1]
                # Ensure correct order
                x_min, x_max = min(x1, x2), max(x1, x2)
                y_min, y_max = min(y1, y2), max(y1, y2)
                width = x_max - x_min
                height = y_max - y_min

                # For COCO, convert rectangle to polygon
                segmentation = [
                    [x_min, y_min, x_max, y_min, x_max, y_max, x_min, y_max]
                ]

                # Calculate area
                area = width * height

            elif shape_type == "circle":
                # Circle: points contains [center, point on perimeter]
                center_x, center_y = points[0]
                radius_point_x, radius_point_y = points[1]
                radius = np.sqrt(
                    (center_x - radius_point_x) ** 2 + (center_y - radius_point_y) ** 2
                )

                # Approximate circle with 20-point polygon
                num_points = 20
                polygon_points = []
                for i in range(num_points):
                    angle = 2 * np.pi * i / num_points
                    x = center_x + radius * np.cos(angle)
                    y = center_y + radius * np.sin(angle)
                    polygon_points.extend([x, y])

                segmentation = [polygon_points]

                # Calculate bounding box
                x_min = center_x - radius
                y_min = center_y - radius
                width = 2 * radius
                height = 2 * radius

                # Calculate area
                area = np.pi * (radius**2)

            else:
                # Skip unsupported shape types
                print(f"Warning: Shape type '{shape_type}' not supported. Skipping...")
                continue

            # Create annotation entry
            coco_format["annotations"].append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": category_id_map[label],
                    "segmentation": segmentation,
                    "area": area,
                    "bbox": [x_min, y_min, width, height],
                    "iscrowd": 0,
                }
            )

            annotation_id += 1

        image_id += 1

    # Save COCO format JSON
    coco_json_path = os.path.join(coco_dir, "annotations.json")
    with open(coco_json_path, "w") as f:
        json.dump(coco_format, f, indent=2)

    print(
        f"Conversion complete: {len(labelme_json_files)} LabelMe files converted to COCO format"
    )
    print(f"COCO JSON file saved at: {coco_json_path}")

    return coco_format
