import os
import warnings
import slideflow as sf
from slideflow.mil import eval_mil

from utils import PROJECT_PATH

if __name__ == "__main__":

    os.environ["SF_LOGGING_LEVEL"] = "10"

    # Loading project and torch dataset
    project = sf.load_project(PROJECT_PATH)
    # Load torch dataset of thumbnails from project
    test = project.dataset(
        tile_px=512,
        tile_um="20x",
        config=(PROJECT_PATH / "datasets.json").name,
        filters = {"dataset": ["test"]},
        sources="MIL-Image-Classification"
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore") # Ignore warnings

        evaluation = eval_mil(
            project=project,
            dataset=test,
            weights=str(PROJECT_PATH / "models" / "model.pth"),
            outcomes="category",
            bags=str(PROJECT_PATH / "bags"),
            models=str(PROJECT_PATH / "models"),
            outdir=str(PROJECT_PATH / "evaluation"),
        )
    
    print(evaluation)