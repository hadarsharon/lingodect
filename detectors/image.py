"""
Module for defining image-based handwritten language detection models and their full implementation(s).
These models are used with Datasets from the datasets.py module, to perform language classification on text.
"""

from datasets.image.IAM.iam import IAM

characters = {}

if __name__ == "__main__":
    iam = IAM()
    labels = iam.get_labels(partition="train")
    vectorized = iam.vectorize_label(label=labels[0])
    _ = iam.prepare_dataset()
    print("Yes.")
