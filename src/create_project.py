import pandas as pd
import slideflow as sf
from slideflow.util import is_project
from utils import PROJECT_PATH

if __name__ == "__main__":

    if not is_project(PROJECT_PATH):
        # creates empty slideflow project (adds .json configs, folders etc.)
        # Won't work if files already exist
        project = sf.create_project(
            root=PROJECT_PATH,
            name="MIL-Image-Classification",
            slides="/share/UBC-OCEAN/"
        )

    # delete thumbnail entries from annotations.csv
    annotations = pd.read_csv(PROJECT_PATH / "annotations.csv", index_col="patient")
    indices = annotations.index.astype(str).str                         # Get indices as string list
    thumbnail_entries = annotations[indices.contains("_thumbnail")]
    annotations.drop(thumbnail_entries, inplace=True)                   
    annotations.to_csv(PROJECT_PATH / "annotations.csv")