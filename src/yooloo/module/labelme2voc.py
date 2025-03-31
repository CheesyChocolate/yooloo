import os
import json
import numpy as np
import cv2
import xml.etree.ElementTree as ET
import glob
import shutil
from tqdm import tqdm
import random


def create_voc_structure(output_dir):
    """Create the standard Pascal VOC directory structure."""
    voc_dirs = {
        "annotations": os.path.join(output_dir, "Annotations"),
        "jpegimages": os.path.join(output_dir, "JPEGImages"),
        "imagesets": os.path.join(output_dir, "ImageSets"),
        "imagesets_main": os.path.join(output_dir, "ImageSets", "Main"),
        "imagesets_layout": os.path.join(output_dir, "ImageSets", "Layout"),
        "imagesets_segmentation": os.path.join(output_dir, "ImageSets", "Segmentation"),
        "segmentation_class": os.path.join(output_dir, "SegmentationClass"),
        "segmentation_object": os.path.join(output_dir, "SegmentationObject"),
    }

    for directory in voc_dirs.values():
        os.makedirs(directory, exist_ok=True)

    return voc_dirs


def load_labelme_json(json_file):
    """Load a labelme JSON file."""
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def create_class_color_map(class_names):
    """Create a color map for visualization of class segmentation masks."""

    color_map = {name: i + 1 for i, name in enumerate(class_names)}
    return color_map


def polygon_to_mask(img_shape, points, shape_type="polygon"):
    """Convert polygon or rectangle points to binary mask."""
    mask = np.zeros(img_shape[:2], dtype=np.uint8)

    if shape_type == "polygon":
        points_array = np.array(points, dtype=np.int32)

        points_array = points_array.reshape((-1, 1, 2))
        cv2.fillPoly(mask, [points_array], 1)

    elif shape_type == "rectangle":
        x1, y1 = int(points[0]), int(points[1])
        x2, y2 = int(points[2]), int(points[3])
        cv2.rectangle(mask, (x1, y1), (x2, y2), 1, -1)  # -1 means filled rectangle

    return mask


def create_segmentation_masks(labelme_data, class_color_map):
    """
    Create segmentation masks from labelme data.

    Returns:
        class_mask: Segmentation mask with class IDs
        instance_mask: Segmentation mask with instance IDs
    """
    img_height = labelme_data["imageHeight"]
    img_width = labelme_data["imageWidth"]

    class_mask = np.zeros((img_height, img_width), dtype=np.uint8)
    instance_mask = np.zeros((img_height, img_width), dtype=np.uint8)

    for i, shape in enumerate(labelme_data["shapes"]):
        label = shape["label"]
        shape_type = shape["shape_type"]
        points = shape["points"]

        if shape_type == "polygon":
            flat_points = [coord for point in points for coord in point]
            object_mask = polygon_to_mask(
                (img_height, img_width), flat_points, "polygon"
            )
        elif shape_type == "rectangle":
            x1, y1 = points[0]
            x2, y2 = points[1]
            object_mask = polygon_to_mask(
                (img_height, img_width), [x1, y1, x2, y2], "rectangle"
            )
        else:
            continue

        class_id = class_color_map.get(label, 0)
        class_mask[object_mask > 0] = class_id

        instance_mask[object_mask > 0] = i + 1

    return class_mask, instance_mask


def save_mask_as_png(mask, output_path, colormap=None):
    """Save a mask as PNG file, with optional colormap for visualization."""
    if colormap is not None:
        colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        for class_id, color in colormap.items():
            colored_mask[mask == class_id] = color
        cv2.imwrite(output_path, colored_mask)
    else:
        cv2.imwrite(output_path, mask)


def create_voc_xml_annotation(labelme_data, xml_file, image_filename):
    """Create VOC format XML annotation file from labelme data."""
    root = ET.Element("annotation")

    ET.SubElement(root, "folder").text = "VOC2012"
    ET.SubElement(root, "filename").text = image_filename

    size = ET.SubElement(root, "size")
    ET.SubElement(size, "width").text = str(labelme_data["imageWidth"])
    ET.SubElement(size, "height").text = str(labelme_data["imageHeight"])
    ET.SubElement(size, "depth").text = "3"  # Assuming RGB images

    for shape in labelme_data["shapes"]:
        obj = ET.SubElement(root, "object")
        ET.SubElement(obj, "name").text = shape["label"]
        ET.SubElement(obj, "difficult").text = "0"

        bndbox = ET.SubElement(obj, "bndbox")

        points = shape["points"]
        if shape["shape_type"] == "polygon":
            x_coords = [point[0] for point in points]
            y_coords = [point[1] for point in points]
            xmin, xmax = min(x_coords), max(x_coords)
            ymin, ymax = min(y_coords), max(y_coords)
        elif shape["shape_type"] == "rectangle":
            xmin, ymin = points[0]
            xmax, ymax = points[1]

        ET.SubElement(bndbox, "xmin").text = str(int(xmin))
        ET.SubElement(bndbox, "ymin").text = str(int(ymin))
        ET.SubElement(bndbox, "xmax").text = str(int(xmax))
        ET.SubElement(bndbox, "ymax").text = str(int(ymax))

    tree = ET.ElementTree(root)
    tree.write(xml_file)


def generate_imagesets_files(image_ids, voc_dirs, train_ratio=0.8):
    """Generate train/val split files for Main, Layout, and Segmentation."""

    random.shuffle(image_ids)

    split_idx = int(len(image_ids) * train_ratio)
    train_ids = image_ids[:split_idx]
    val_ids = image_ids[split_idx:]

    for dir_key in ["imagesets_main", "imagesets_layout", "imagesets_segmentation"]:
        with open(os.path.join(voc_dirs[dir_key], "train.txt"), "w") as f:
            f.write("\n".join(train_ids))

        with open(os.path.join(voc_dirs[dir_key], "val.txt"), "w") as f:
            f.write("\n".join(val_ids))

        with open(os.path.join(voc_dirs[dir_key], "trainval.txt"), "w") as f:
            f.write("\n".join(train_ids + val_ids))


def convert_labelme_to_voc(labelme_dir, output_dir, train_ratio=0.8):
    """
    Convert a directory of labelme JSON files to Pascal VOC format.

    Args:
        labelme_dir: Directory containing labelme JSON files and their corresponding images
        output_dir: Directory to save the Pascal VOC dataset
        train_ratio: Ratio of images to use for training (rest go to validation)
    """

    voc_dirs = create_voc_structure(output_dir)

    json_files = glob.glob(os.path.join(labelme_dir, "*.json"))

    if not json_files:
        print(f"No JSON files found in {labelme_dir}")
        return

    all_classes = set()
    for json_file in json_files:
        data = load_labelme_json(json_file)
        for shape in data["shapes"]:
            all_classes.add(shape["label"])

    class_names = sorted(list(all_classes))
    class_color_map = create_class_color_map(class_names)

    print(f"Found {len(class_names)} classes: {', '.join(class_names)}")

    vis_colormap = {
        i + 1: (r, g, b)
        for i, (r, g, b) in enumerate(
            [
                (255, 0, 0),
                (0, 255, 0),
                (0, 0, 255),
                (255, 255, 0),
                (255, 0, 255),
                (0, 255, 255),
                (128, 0, 0),
                (0, 128, 0),
                (0, 0, 128),
                (128, 128, 0),
                (128, 0, 128),
                (0, 128, 128),
            ][: len(class_names)]
        )
    }
    vis_colormap[0] = (0, 0, 0)  # Background is black

    image_ids = []

    for json_file in tqdm(json_files, desc="Converting labelme to VOC"):
        labelme_data = load_labelme_json(json_file)

        base_name = os.path.splitext(os.path.basename(json_file))[0]
        image_ids.append(base_name)

        img_file = os.path.join(labelme_dir, labelme_data["imagePath"])
        if not os.path.exists(img_file):
            for ext in [".jpg", ".jpeg", ".png"]:
                potential_img = os.path.join(labelme_dir, base_name + ext)
                if os.path.exists(potential_img):
                    img_file = potential_img
                    break

        if not os.path.exists(img_file):
            print(f"Warning: Image file not found for {json_file}")
            continue

        image_filename = base_name + os.path.splitext(img_file)[1]
        shutil.copy(img_file, os.path.join(voc_dirs["jpegimages"], image_filename))

        xml_file = os.path.join(voc_dirs["annotations"], base_name + ".xml")
        create_voc_xml_annotation(labelme_data, xml_file, image_filename)

        class_mask, instance_mask = create_segmentation_masks(
            labelme_data, class_color_map
        )

        save_mask_as_png(
            class_mask,
            os.path.join(voc_dirs["segmentation_class"], base_name + ".png"),
            vis_colormap,
        )
        save_mask_as_png(
            instance_mask,
            os.path.join(voc_dirs["segmentation_object"], base_name + ".png"),
            vis_colormap,
        )

    generate_imagesets_files(image_ids, voc_dirs, train_ratio)

    with open(os.path.join(output_dir, "class_names.txt"), "w") as f:
        for i, name in enumerate(class_names):
            f.write(f"{i + 1}: {name}\n")

    print(f"Conversion complete! VOC format dataset saved to {output_dir}")
    print(f"- {len(image_ids)} images processed")
    print(f"- {len(class_names)} classes identified")
