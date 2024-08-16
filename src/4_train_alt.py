import os
import warnings
import slideflow as sf
from slideflow.mil import train_mil
from slideflow.mil import TrainerConfigFastAI


from utils import PROJECT_PATH, BAGS_PATH, MODEL_PATH


if __name__ == "__main__":

    os.environ["SF_LOGGING_LEVEL"] = "10"

    # Load the project
    project = sf.load_project(PROJECT_PATH)
    # Load torch dataset of thumbnails from project
    ubc = project.dataset(
        tile_px=512,
        tile_um="20x",
        config=(PROJECT_PATH / "datasets.json").name,
        filters = {"dataset": ["train", "validation"]},
        sources="MIL-Image-Classification"
    )

    # filter dataset into train and validation
    train = ubc.filter({"dataset": ["train"]})
    val   = ubc.filter({"dataset": ["validation"]})

    # Define the model configuration
    config = TrainerConfigFastAI(
        model = "attention_mil",
        lr         = 1e-4,
        batch_size = 64,
        epochs     = 100
    )

    # Train the model
    with warnings.catch_warnings():
        warnings.simplefilter("ignore") # Ignore warnings
        
        train_mil(
            config=config,
            train_dataset=train,
            val_dataset=val,
            outcomes="label",
            project=project,
            bags=str(BAGS_PATH),
            outdir=str(MODEL_PATH),
        )