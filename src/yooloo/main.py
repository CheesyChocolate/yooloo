from yooloo.module.labelme2coco import convert_labelme_to_coco


def main():
    # Convert all LabelMe files in 'foo' directory to COCO format in 'coco' directory
    result = convert_labelme_to_coco(
        labelme_dir="data/selected-rectangle/labelme",
        coco_dir="data/selected-rectangle/coco",
    )
    print(
        f"Converted {len(result['images'])} images with {len(result['annotations'])} annotations"
    )


if __name__ == "__main__":
    main()
