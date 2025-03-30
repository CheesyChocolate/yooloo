# Stolen from https://github.com/khanmhmdi/Ready-to-Use-Computer-Vision-Modules/blob/main/coco_validator.py

import argparse
import json
import sys
from typing import Dict, List, Set, Union

"""
This code is COCO JSON validation script. you only need to pass the json path to this code to find all the inconsistencies of your annotation file with the common COCO json format file.
"""


class COCOValidator:
    """
    A comprehensive validator for COCO (Common Objects in Context) JSON annotation files.

    This validator performs multiple checks on the structure and content of COCO JSON files,
    providing detailed error reporting and guidance.

    Key Features:
    - Validates top-level JSON structure
    - Checks integrity of images, categories, and annotations
    - Provides comprehensive error reporting
    - Supports command-line usage
    """

    @staticmethod
    def load_json(file_path: str) -> Dict:
        """
        Safely load a JSON file from the given file path.

        Args:
            file_path (str): Path to the JSON file to be loaded.

        Returns:
            Dict: Parsed JSON data.

        Raises:
            SystemExit: If file cannot be read or parsed.
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Error: File not found at {file_path}")
            sys.exit(1)
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"Unexpected error reading JSON file: {e}")
            sys.exit(1)

    @classmethod
    def validate_coco_structure(cls, coco_data: Dict) -> List[str]:
        """
        Comprehensive validation of COCO JSON structure.

        Args:
            coco_data (Dict): COCO JSON data to validate.

        Returns:
            List[str]: List of validation errors, empty if no errors found.
        """
        errors = []

        # Validate top-level keys
        errors.extend(cls._validate_top_level_keys(coco_data))

        # Validate images and capture their ids
        image_ids = cls._validate_images(coco_data, errors)

        # Validate categories and capture their ids
        category_ids = cls._validate_categories(coco_data, errors)

        # Validate annotations
        cls._validate_annotations(coco_data, errors, image_ids, category_ids)

        return errors

    @staticmethod
    def _validate_top_level_keys(coco_data: Dict) -> List[str]:
        """Validate presence of required top-level keys in COCO JSON."""
        errors = []
        required_keys = ["images", "annotations", "categories"]
        for key in required_keys:
            if key not in coco_data:
                errors.append(f"Missing required top-level key: '{key}'")
        return errors

    @classmethod
    def _validate_images(cls, coco_data: Dict, errors: List[str]) -> Set[int]:
        """
        Validate image entries for required keys and uniqueness.

        Args:
            coco_data (Dict): Full COCO JSON data
            errors (List[str]): List to append validation errors

        Returns:
            Set[int]: Set of valid image IDs
        """
        images = coco_data.get("images", [])
        image_ids = set()
        required_keys = ["id", "width", "height", "file_name"]

        for idx, image in enumerate(images):
            # Check required keys
            for key in required_keys:
                if key not in image:
                    errors.append(
                        f"Image entry at index {idx} missing required key: '{key}'"
                    )

            # Check image ID uniqueness
            image_id = image.get("id")
            if image_id in image_ids:
                errors.append(f"Duplicate image id found: {image_id}")
            else:
                image_ids.add(image_id)

            # Type and value checks
            cls._validate_image_types(image, image_id, errors)

        return image_ids

    @staticmethod
    def _validate_image_types(image: Dict, image_id: int, errors: List[str]):
        """Validate image entry data types and values."""
        if not isinstance(image.get("width", 0), int) or not isinstance(
            image.get("height", 0), int
        ):
            errors.append(f"Image id {image_id} has non-integer width/height.")
        if not isinstance(image.get("file_name", ""), str):
            errors.append(
                f"Image id {image_id} has invalid file_name (should be a string)."
            )

    @classmethod
    def _validate_categories(cls, coco_data: Dict, errors: List[str]) -> Set[int]:
        """
        Validate category entries for required keys and uniqueness.

        Args:
            coco_data (Dict): Full COCO JSON data
            errors (List[str]): List to append validation errors

        Returns:
            Set[int]: Set of valid category IDs
        """
        categories = coco_data.get("categories", [])
        category_ids = set()
        required_keys = ["id", "name", "supercategory"]

        for idx, category in enumerate(categories):
            # Check required keys
            for key in required_keys:
                if key not in category:
                    errors.append(
                        f"Category entry at index {idx} missing required key: '{key}'"
                    )

            # Check category ID uniqueness
            cat_id = category.get("id")
            if cat_id in category_ids:
                errors.append(f"Duplicate category id found: {cat_id}")
            else:
                category_ids.add(cat_id)

            # Type checks for name and supercategory
            if not isinstance(category.get("name", ""), str):
                errors.append(
                    f"Category id {cat_id} has invalid name (should be a string)."
                )
            if not isinstance(category.get("supercategory", ""), str):
                errors.append(
                    f"Category id {cat_id} has invalid supercategory (should be a string)."
                )

        return category_ids

    @classmethod
    def _validate_annotations(
        cls,
        coco_data: Dict,
        errors: List[str],
        valid_image_ids: Set[int],
        valid_category_ids: Set[int],
    ):
        """
        Validate annotation entries for required keys, references, and data integrity.

        Args:
            coco_data (Dict): Full COCO JSON data
            errors (List[str]): List to append validation errors
            valid_image_ids (Set[int]): Set of valid image IDs
            valid_category_ids (Set[int]): Set of valid category IDs
        """
        annotations = coco_data.get("annotations", [])
        required_keys = ["id", "image_id", "category_id", "bbox", "area", "iscrowd"]
        annotation_ids = set()

        for idx, ann in enumerate(annotations):
            # Check required keys
            for key in required_keys:
                if key not in ann:
                    errors.append(
                        f"Annotation at index {idx} missing required key: '{key}'"
                    )

            # Validate ID uniqueness
            ann_id = ann.get("id")
            if ann_id in annotation_ids:
                errors.append(f"Duplicate annotation id found: {ann_id}")
            else:
                annotation_ids.add(ann_id)

            # Cross-reference validation
            cls._validate_annotation_references(
                ann, ann_id, valid_image_ids, valid_category_ids, errors
            )

            # Detailed annotation validations
            cls._validate_bbox(ann, ann_id, errors)
            cls._validate_area(ann, ann_id, errors)
            cls._validate_iscrowd(ann, ann_id, errors)
            cls._validate_segmentation(ann, ann_id, errors)

    @staticmethod
    def _validate_annotation_references(
        ann: Dict,
        ann_id: int,
        valid_image_ids: Set[int],
        valid_category_ids: Set[int],
        errors: List[str],
    ):
        """Validate image and category references in annotations."""
        image_id = ann.get("image_id")
        if image_id not in valid_image_ids:
            errors.append(
                f"Annotation id {ann_id} references non-existent image id: {image_id}"
            )

        category_id = ann.get("category_id")
        if category_id not in valid_category_ids:
            errors.append(
                f"Annotation id {ann_id} references non-existent category id: {category_id}"
            )

    @staticmethod
    def _validate_bbox(ann: Dict, ann_id: int, errors: List[str]):
        """Validate bounding box format and values."""
        bbox = ann.get("bbox")
        if not (isinstance(bbox, list) and len(bbox) == 4):
            errors.append(
                f"Annotation id {ann_id} has invalid bbox format. Expected list of 4 numbers."
            )
        else:
            if any(not isinstance(coord, (int, float)) for coord in bbox):
                errors.append(
                    f"Annotation id {ann_id} has bbox with non-numeric values: {bbox}"
                )
            elif any(coord < 0 for coord in bbox[0:2]):
                errors.append(
                    f"Annotation id {ann_id} has negative x or y in bbox: {bbox}"
                )

    @staticmethod
    def _validate_area(ann: Dict, ann_id: int, errors: List[str]):
        """Validate annotation area."""
        area = ann.get("area")
        if not isinstance(area, (int, float)):
            errors.append(f"Annotation id {ann_id} has area missing or not numeric.")
        elif area < 0:
            errors.append(f"Annotation id {ann_id} has negative area: {area}")

    @staticmethod
    def _validate_iscrowd(ann: Dict, ann_id: int, errors: List[str]):
        """Validate iscrowd flag."""
        iscrowd = ann.get("iscrowd")
        if not isinstance(iscrowd, (int, bool)):
            errors.append(
                f"Annotation id {ann_id} has invalid iscrowd value: {iscrowd}"
            )

    @staticmethod
    def _validate_segmentation(ann: Dict, ann_id: int, errors: List[str]):
        """Validate optional segmentation data."""
        if "segmentation" in ann:
            seg = ann["segmentation"]
            if not (isinstance(seg, list) or isinstance(seg, dict)):
                errors.append(
                    f"Annotation id {ann_id} has segmentation of invalid type. Expected list or dict."
                )

    @classmethod
    def validate_file(cls, file_path: str) -> bool:
        """
        Validate a COCO JSON file and print results.

        Args:
            file_path (str): Path to the COCO JSON file to validate.

        Returns:
            bool: True if validation passes, False otherwise.
        """
        try:
            # Load JSON data
            coco_data = cls.load_json(file_path)

            # Validate structure
            errors = cls.validate_coco_structure(coco_data)

            # Report results
            if errors:
                print(f"Found {len(errors)} inconsistencies in the COCO JSON format:")
                for err in errors:
                    print(f" - {err}")
                return False
            else:
                print("The COCO annotations file passed all basic format validations.")
                return True

        except Exception as e:
            print(f"Unexpected error during validation: {e}")
            return False


def main():
    """
    Command-line interface for COCO JSON validation.
    Supports validation of multiple files and provides clear output.
    """
    parser = argparse.ArgumentParser(
        description="Validate COCO JSON annotation files.",
        epilog="Example: python coco_validator.py annotations1.json annotations2.json",
    )
    parser.add_argument(
        "files", nargs="+", help="One or more COCO JSON files to validate"
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Suppress successful validation messages",
    )

    args = parser.parse_args()

    validation_results = []

    for file_path in args.files:
        print(f"Validating file: {file_path}")
        result = COCOValidator.validate_file(file_path)
        validation_results.append(result)
        print()  # Add a blank line between file validations

    # Overall result
    if all(validation_results):
        print("All specified files passed validation.")
        sys.exit(0)
    else:
        print("One or more files failed validation.")
        sys.exit(1)


if __name__ == "__main__":
    main()
