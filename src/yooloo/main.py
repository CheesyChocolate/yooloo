from yooloo.module.labelme2coco import convert_labelme_to_coco
from yooloo.module.labelme2yolo import convert_labelme_to_yolov8


def main_labelme_to_coco():
    # Convert all LabelMe files to COCO format
    result = convert_labelme_to_coco(
        labelme_dir="data/selected-rectangle/labelme",
        coco_dir="data/selected-rectangle/coco",
    )
    print(
        f"Converted {len(result['images'])} images with {len(result['annotations'])} annotations"
    )

    result = convert_labelme_to_coco(
        labelme_dir="data/selected-polygons/labelme",
        coco_dir="data/selected-polygons/coco",
    )
    print(
        f"Converted {len(result['images'])} images with {len(result['annotations'])} annotations"
    )


def main_labelme_to_yolo():
    # Convert all LabelMe files to YOLOv8 format
    stats = convert_labelme_to_yolov8(
        labelme_dir="data/selected-rectangle/labelme",
        output_dir="data/selected-rectangle/yolo",
        train_ratio=0.8,  # 80%/20%
        copy_images=True,  # copy/link
    )
    print(f"Conversion complete with {stats['annotations_count']} annotations")
    print(f"Train set: {stats['train_files']} images")
    print(f"Validation set: {stats['val_files']} images")

    stats = convert_labelme_to_yolov8(
        labelme_dir="data/selected-polygons/labelme",
        output_dir="data/selected-polygons/yolo",
        train_ratio=0.8,  # 80%/20%
        copy_images=True,  # copy/link
    )
    print(f"Conversion complete with {stats['annotations_count']} annotations")
    print(f"Train set: {stats['train_files']} images")
    print(f"Validation set: {stats['val_files']} images")


def main():
    # main_labelme_to_coco()
    main_labelme_to_yolo()


if __name__ == "__main__":
    main()
