import os
import slideflow as sf
from slideflow.slide import qc
from utils import PROJECT_PATH

if __name__ == "__main__":

    # Set backend to libvips (the default backend cucim does not support .png images)
    os.environ["SF_SLIDE_BACKEND"] = "libvips"
    
    # Initialize Stain Normalizer
    reinhard = sf.norm.autoselect('reinhard')

    # Loading the project & metadata
    project = sf.load_project(PROJECT_PATH)

    # Load thumbnail dataset object from project
    ubc = project.dataset(
        tile_px=224,
        tile_um="10x",
        config=(PROJECT_PATH / "datasets.json").name,
        sources="UBC-OCEAN-Thumbnails"
    )

    # extract tiles into datasets tile folder using normalizer
    ubc.extract_tiles(
        qc=qc.Otsu(),
        normalizer="reinhard_mask",
        save_tiles=True
    )