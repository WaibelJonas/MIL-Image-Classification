import os
import json
import warnings
import slideflow as sf
from slideflow.mil import mil_config, train_mil
from slideflow.mil import TrainerConfigFastAI
from sklearn.model_selection import KFold

from utils import PROJECT_PATH, SHARE_PATH, BAGS_PATH, MODEL_PATH

"""
Pipeline step 4: Train MIL model

-> Trains a MIL model using the slideflow library.
-> Utilizes the split information in annotations.csv to filter the dataset into train and validation.
-> Ignoring warnings since slideflows train_mil function produces a lot of them. (feel free to remove)
"""

# Define number of folds for cross validation
N_FOLDS: int = 5

if __name__ == "__main__":

    os.environ["SF_LOGGING_LEVEL"] = "10"

    # Loading project and torch dataset
    project = sf.load_project(PROJECT_PATH)
    # Load torch dataset of thumbnails from project
    ubc = project.dataset(
        tile_px=512,
        tile_um="20x",
        config=(PROJECT_PATH / "datasets.json").name,
        filters = {"dataset": ["train", "validation"]},
        sources="MIL-Image-Classification"
    )

    # Define the model configuration
    config = TrainerConfigFastAI(
        model = "attention_mil",
        lr         = 1e-4,
        batch_size = 32,
        epochs     = 40
    )

    with open(SHARE_PATH / "splits.json", "r") as file_stream:
        splits = json.load(file_stream)

    
    current_fold = splits[0]

    # Extract train, validation, and test slides
    train_slides = list(current_fold['splits']['train'].keys())
    val_slides   = list(current_fold['splits']['val'].keys())

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
            bags     = str(BAGS_PATH),
            outdir   = str(MODEL_PATH)
        )