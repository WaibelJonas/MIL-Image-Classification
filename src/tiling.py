import os
import slideflow as sf
from utils import PROJECT_PATH

if __name__ == "__main__":

    # Set backend to libvips (the default backend cucim does not support .png images)
    os.environ["SF_SLIDE_BACKEND"] = "libvips"

    # Loading the project & metadata
    project = sf.load_project(PROJECT_PATH)

    # Load dataset object from project
    ubc = project.dataset(tile_px=224, tile_um="20x")

    # extract tiles into specified tiles folder
    ubc.extract_tiles()

    