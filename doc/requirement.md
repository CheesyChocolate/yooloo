Pattern Recognition Course Spring Semester Midterm Report

# Introduction

In this project, students will manually annotate an object of their choice in
20 different images using the LabelMe tool, following instance-based and
semi-supervised methods. Subsequently, they will write Python scripts to
convert the annotated data into YOLO (various versions), COCO format, and
Pascal VOC format for YOLOX, adapting these conversions to their prepared
dataset.

This report covers the objective of the project, the methods used, the
implementation process, and the results obtained.

# Objective of the Project

The main objectives of this project are:

- To help students learn the data annotation process
- To enable them to perform instance-based and semi-supervised annotation using
the LabelMe tool
- To encourage students to understand different data formats and develop
conversion algorithms
- To help students comprehend the structure of YOLO (v2, v3, v4, v5, v8), COCO,
and Pascal VOC formats
- To allow students to experience the process of creating real-world datasets

# Methods Used

## Data Annotation with LabelMe

LabelMe is an open-source manual data annotation tool commonly used for object
detection. Students will annotate selected objects using the following two
methods:

- Instance-Based Annotation: Each object will be labeled individually,
assigning a unique identity to each one.
- Semi-Supervised Annotation: Students will first manually annotate a portion
of the dataset and then attempt to accelerate the process using automated
annotation methods.

## Data Formats and Conversion

The annotated data will be converted into the following formats:

- YOLO Format: A format that contains the class label and the normalized
coordinates of the bounding boxes.
- COCO Format: A JSON-based data structure that includes bounding box
information and class labels.
- Pascal VOC Format (for YOLOX): An XML-based format that contains object
class, bounding box, and image size information.

# Implementation Process

## Data Annotation

1. Using the LabelMe tool, each student will annotate 20 images following both
   instancebased and semi-supervised methods.
2. The annotated data will be saved in .json format.

## Data Conversion Algorithms

Students will write conversion algorithms for the following:

- Converting LabelMe to YOLO format
- Converting LabelMe to COCO format
- Converting LabelMe to Pascal VOC format

Each algorithm must be developed using the Python programming language.

# Deliverables

Upon completing the project, students must submit the following materials:

1. Code Files: Python scripts containing the data conversion algorithms.
2. Annotated Dataset: .json files created with LabelMe.
3. Converted Datasets: Data in YOLO, COCO, and Pascal VOC formats.
4. Report: A detailed report covering the project process, methods used, and
   results obtained.

Note: Code similarity detection will be performed. If students submit highly
similar code, the submission will not be evaluated.

6. Conclusion and Evaluation

This project aims to help students learn the data annotation process,
understand differences between data formats, and apply data conversion
processes in practice.

The submitted projects will be evaluated based on the following criteria:

- Annotation accuracy (30%)
- Correctness of data conversion scripts (40%)
- Content and clarity of the report (20%)
- File organization and completeness of submitted materials (10%)

Good luck!
