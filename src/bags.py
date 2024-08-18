import os
import slideflow as sf
from pathlib import Path
from slideflow.model import build_feature_extractor

from utils import TILE_SIZE, TILE_UM, FEATURE_EXTRACTOR, PROJECT_PATH, BAGS_PATH



def bags(tile_px: int = TILE_SIZE, tile_um: str = TILE_UM, feature_extractor: str = FEATURE_EXTRACTOR, outdir: Path = BAGS_PATH):
    """
    Pipeline step 3: Generate feature bags

    -> Extracts features from tiles and saves them in the local project storage.\n
    -> Uses the plip feature extractor.\n
    -> Expect long runtime for large datasets.\n

    Parameters:
        tile_px (int): Size of the tiles in pixels.
        tile_um (str): Size of the tiles in micrometers.
        feature_extractor (str): Name of the feature extractor.
        outdir (str): Path to save the extracted features
    """

    os.environ["SF_LOGGING_LEVEL"] = "10"

    # Loading project and torch dataset
    project = sf.load_project(PROJECT_PATH)
    # Load torch dataset of thumbnails from project
    ubc = project.dataset(
        tile_px=tile_px,
        tile_um=tile_um,
        config=(PROJECT_PATH / "datasets.json").name,
        sources="MIL-Image-Classification"
    )

    project.generate_feature_bags(
        model   = feature_extractor,
        dataset = ubc,
        outdir  = str(outdir)
    )


if __name__ == "__main__":
    bags(
        tile_px = TILE_SIZE,
        tile_um = TILE_UM,
        feature_extractor = FEATURE_EXTRACTOR,
        outdir = BAGS_PATH
    )


