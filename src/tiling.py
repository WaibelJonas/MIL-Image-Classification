import os
from slideflow.slide import qc
import slideflow as sf
from pathlib import Path
from utils import (
    TILE_SIZE,
    TILE_UM,
    PROJECT_PATH,
    DATASET_PATH,
    TFRECORD_PATH,
    TIFF_BUFFER_SIZE,
    CLEAR_BUFFER,
    batch_generator,
    batch_conversion
)

TIFF_BUFFER_PATH  = PROJECT_PATH / "tiffs"   # Path to temorarily save the converted tiffs
CLEAR_BUFFER      = True                     # Whether to clear the tiff buffer after processing

def tiling(dataset_path: Path = DATASET_PATH, output_path: Path = TFRECORD_PATH,
           tile_px: int = TILE_SIZE, tile_um: str = TILE_UM,
           batch_size: int = TIFF_BUFFER_SIZE, clear_buffer: bool = CLEAR_BUFFER):
    """
    Pipeline step 1: Tiling

    -> Extracts tiles from Whole Slide Images (WSI) and saves them on the shared storage.

    -> The tiling is done in batches to avoid memory issues.

    -> Expect long runtime for large datasets.

    Parameters:
        dataset_path (Path): Path to the dataset containing the WSI.
        output_path (Path): Path to save the extracted tiles.
        tile_px (int): Size of the tiles in pixels.
        tile_um (str): Size of the tiles in micrometers.
        batch_size (int): Number of WSI to process in parallel.
        clear_buffer (bool): Whether to clear the tiff buffer after processing.
    """

    # Set backend to libvips (the default backend cucim does not support .png images)
    os.environ["SF_SLIDE_BACKEND"] = "libvips"

    # Extract all WSI from Dataset (without thumbnails)
    slide_list: list[Path] = []
    for root, _, file_names in os.walk(dataset_path / "train_images"):
        slide_list.extend([Path(root) / file_name for file_name in file_names if file_name.endswith('.png') and not '_thumbnail' in file_name])

    # Load the project & metadata
    project = sf.load_project(PROJECT_PATH)

    # Load dataset object from project
    ubc = project.dataset(
        tile_px=tile_px,
        tile_um=tile_um,
        config=(PROJECT_PATH / "datasets.json").name,
        sources="TIFF_BUFFER"
    )

    # Separate Slides into batches
    for batch_idx, batch in enumerate(batch_generator(slide_list, batch_size=batch_size)):

        print(f"--- Batch {batch_idx} ---")

        # Skipping fully completed batches
        if all([(output_path / f"{tile_px}px_{tile_um}" / f"{slide.stem}.tfrecords").exists() for slide in batch]):
            print("Slides in batch already processed. Skipping")
            continue

        # Convert batch to tiffs
        _ = batch_conversion(batch, TIFF_BUFFER_PATH, verbose=True)

        try:
            # extract tiles into datasets tile folder using normalizer
            ubc.extract_tiles(
                qc=qc.Otsu(),
                normalizer="reinhard_mask",
                report=False
            )

        except Exception as err:
            print(err)

        if clear_buffer:
            for tiff in [file for file in TIFF_BUFFER_PATH.iterdir() if file.suffix == ".tiff"]:
                os.remove(tiff)
            print("Cleared tiff buffer")


if __name__ == "__main__":
    tiling()




    
