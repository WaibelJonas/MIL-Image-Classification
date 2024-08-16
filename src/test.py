import pandas as pd
import slideflow as sf
from pathlib import Path
from utils import PROJECT_PATH, DATASET_PATH

SPLIT_PATH = Path("/share/praktikum2024/splits")


if __name__ == "__main__":

    ann = pd.read_csv("annotations.csv", index_col="patient")

    ann.rename(columns={"category": "label"}, inplace=True)

    ann.to_csv("annotations.csv")