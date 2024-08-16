import pandas as pd
import slideflow as sf
from slideflow.util import is_project
from utils import (
    PROJECT_PATH,
    DATASET_PATH,
    SPLIT_PATH
)

"""
Pipeline Step 1: Create Project

-> Creates project structure and modifies annotations.csv to better reflect the dataset.
-> Adds 'category' and 'slide' columns to annotations.csv.
-> Adds 'dataset' column to annotations.csv annotating to which split each image belongs.
-> This script should be run only once.
"""

if __name__ == "__main__":

    if not is_project(PROJECT_PATH):
        # creates empty slideflow project (adds .json configs, folders etc.)
        # Won't work if files already exist
        project = sf.create_project(
            root=PROJECT_PATH,
            name="MIL-Image-Classification",
            slides="/share/UBC-OCEAN/"
        )

    # Load annotations
    annotations = pd.read_csv(PROJECT_PATH / "annotations.csv", index_col="patient")

    # Add label annotation ('label' column) to annotations.csv
    df = pd.read_csv(DATASET_PATH / "train.csv", index_col="image_id")
    df.index = df.index.astype(str)
    annotations.update(df["label"])

    # Add slide annotation to annotations.csv
    annotations["slide"] = annotations.index

    # Load test, train and validation splits
    test  = pd.read_csv(SPLIT_PATH / "test.csv",  index_col="image_id")
    train = pd.read_csv(SPLIT_PATH / "train.csv", index_col="image_id")
    val   = pd.read_csv(SPLIT_PATH / "validation.csv",   index_col="image_id")

    # Add dataset annotation to annotations.csv
    for idx in annotations.index:
        if idx in test.index:
            annotations.loc[idx, "dataset"] = "test"
        elif idx in train.index:
            annotations.loc[idx, "dataset"] = "train"
        elif idx in val.index:
            annotations.loc[idx, "dataset"] = "validation"
        else:
            continue

    # Save modified annotations.csv
    annotations.to_csv(PROJECT_PATH / "annotations.csv")