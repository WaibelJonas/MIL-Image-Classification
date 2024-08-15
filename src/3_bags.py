import os
import slideflow as sf
from slideflow.model import build_feature_extractor

from utils import PROJECT_PATH, BAGS_PATH

"""
Pipeline for generating bags using histossl feature extractor
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
    
    plip = build_feature_extractor("plip", tile_px = 512)

    project.generate_feature_bags(
        model   = plip,
        dataset = ubc,
        outdir  = BAGS_PATH
    )
7
