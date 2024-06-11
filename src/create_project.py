import pandas as pd
import slideflow as sf
from slideflow.util import is_project
from utils import PROJECT_PATH, DATASET_PATH

"""
Creates project structure and modifies annotations.csv
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

    # Add label annotation ('category' column) to annotations.csv
    df = pd.read_csv(DATASET_PATH / "train.csv", index_col="image_id")
    df.rename(columns={"label": "category"}, inplace=True)
    df.index = df.index.astype(str)
    # Adding identical thumbnail entries
    for index, row in df.iterrows():
        df.loc[f"{index}_thumbnail"] = row.copy()
    annotations.update(df["category"])

    # Save modified annotations.csv
    annotations.to_csv(PROJECT_PATH / "annotations.csv")