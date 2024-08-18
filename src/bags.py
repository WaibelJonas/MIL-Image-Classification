import os
import slideflow as sf
from slideflow.model import build_feature_extractor

from utils import PROJECT_PATH, BAGS_PATH

"""
Pipeline step 3: Generate feature bags

-> Extracts features from tiles and saves them in the local project storage.
-> Uses the plip feature extractor.
-> Expect long runtime for large datasets.
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
    
    histo = build_feature_extractor("histossl", tile_px = 512)

    project.generate_feature_bags(
        model   = histo,
        dataset = ubc,
        outdir  = str(BAGS_PATH)
    )
7
