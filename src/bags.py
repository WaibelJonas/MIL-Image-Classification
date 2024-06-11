import slideflow as sf
from slideflow.slide import qc
from utils import PROJECT_PATH
from slideflow.model import build_feature_extractor


if __name__ == "__main__":

    # Loading project and torch dataset
    project = sf.load_project(PROJECT_PATH)
    # Load torch dataset of thumbnails from project
    ubc = project.dataset(
        tile_px = 224,
        tile_um = "10x",
        config  = (PROJECT_PATH / "datasets.json").name,
        sources = "UBC-OCEAN-Thumbnails"
    ).balance(
        "category",
        strategy   = "category"     # Category level oversampling
    )

    histo = build_feature_extractor("histossl", tile_px = 224)

    project.generate_feature_bags(
        histo,
        ubc,
        output_dir = "/bags"
    )
        

