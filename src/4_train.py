import os
import warnings
import slideflow as sf
from slideflow.mil import mil_config, train_mil
from sklearn import KFold

from utils import PROJECT_PATH

"""
Pipeline for training a model
"""

if __name__ == "__main__":

    os.environ["SF_LOGGING_LEVEL"] = "10"

    # Loading project and torch dataset
    project = sf.load_project(PROJECT_PATH)
    # Load torch dataset of thumbnails from project
    ubc = project.dataset(
        tile_px=512,
        tile_um="20x",
        config=(PROJECT_PATH / "datasets.json").name,
        sources="MIL-Image-Classification"
    )

    # filter dataset into train and validation
    train = ubc.filter(dataset=["train"])
    val   = ubc.filter(dataset=["validation"])

    # Define the model configuration
    config = mil_config("attention_mil",
                        lr=1e-4,
                        fit_one_cycle=True)

    # Train the model
    with warnings.catch_warnings():
        warnings.simplefilter("ignore") # Ignore warnings
        
        train_mil(
            config=config,
            train_dataset=train,
            val_dataset=val,
            outcomes="category",
            project=project,
            bags=str(PROJECT_PATH / "bags"),
            outdir=str(PROJECT_PATH / "models"),
        )