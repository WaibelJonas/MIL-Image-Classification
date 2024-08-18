import os
import json
import warnings
import slideflow as sf
from pathlib import Path
from slideflow.mil import mil_config, train_mil
from sklearn.model_selection import KFold

from utils import (
    TILE_SIZE,
    TILE_UM,
    EPOCHS,
    LEARNING_RATE,
    BATCH_SIZE,
    PROJECT_PATH,
    SPLIT_PATH,
    BAGS_PATH,
    MODEL_PATH
)

"""

"""

def train(tile_px: int = TILE_SIZE, tile_um: str = TILE_UM,
          bags_path: Path = BAGS_PATH, split_path: Path = SPLIT_PATH, model_path: Path = MODEL_PATH,
          batch_size: int = BATCH_SIZE, epochs: int = EPOCHS, learning_rate: float = LEARNING_RATE):
    """
    Pipeline step 4: Train MIL model

    -> Trains a MIL model using the slideflow library.\n
    -> Utilizes the split information in annotations.csv to filter the dataset into train and validation.\n
    -> Ignoring warnings since slideflows train_mil function produces a lot of them. (feel free to remove)\n

    Parameters:
        tile_px: int
            Size of the thumbnail in pixels.
        tile_um: str
            Size of the thumbnail in micrometers.
        bags_path: Path
            Path to the bags.
        split_path: Path
            Path to the splits.
        model_path: Path
            Path to the model.
        batch_size: int
            Batch size for training.
        epochs: int
            Number of epochs for training.
    """

    # Load project
    project = sf.load_project(PROJECT_PATH)

    # Load dataset
    ubc = project.dataset(
        tile_px=tile_px,
        tile_um=tile_um,
        config=(PROJECT_PATH / "datasets.json").name,
        filters = {"dataset": ["train", "validation"]},
        sources="MIL-Image-Classification"
    )

    # Load splits
    with open(split_path, "r") as file_stream:
        splits = json.load(file_stream)

    # Define the model configuration
    config = mil_config(
        model      = "attention_mil",
        trainer    = "fastai",
        lr         = learning_rate,
        epochs     = epochs,
        batch_size = batch_size
    )

    # Iterate over folds
    for index, fold in enumerate(splits):
        print(f"--- Fold: {index} ---")

        # Extract train, validation, and test slides
        train_slides = list(fold['splits']['train'].keys())
        val_slides   = list(fold['splits']['val'].keys())

        train = ubc.filter({"slide": train_slides})
        val = ubc.filter({"slide": val_slides})

        # Train the model
        with warnings.catch_warnings():
            warnings.simplefilter("ignore") # Ignore warnings

            train_mil(
                config        = config,
                train_dataset = train,
                val_dataset   = val,
                outcomes = "label",
                project  = project,
                bags     = str(bags_path),
                outdir   = str(model_path)
            )

if __name__ == "__main__":
    train(
        tile_px = TILE_SIZE,
        tile_um = TILE_UM,
        bags_path = BAGS_PATH,
        split_path = SPLIT_PATH,
        model_path = MODEL_PATH,
        batch_size = BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        epochs = EPOCHS,
    )