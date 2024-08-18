import os
from slideflow.slide import qc
import slideflow as sf
from pathlib import Path
from utils import (
    PROJECT_PATH,
    DATASET_PATH,
    TFRECORD_PATH,
    batch_generator,
    batch_conversion
)

"""
Pipeline step 2: Tiling

-> Extracts tiles from Whole Slide Images (WSI) and saves them on the shared storage.
-> The tiling is done in batches to avoid memory issues.
-> Expect long runtime for large datasets.
"""

TIFF_BUFFER_PATH  = PROJECT_PATH / "tiffs"   # Path to temorarily save the converted tiffs
CLEAR_BUFFER      = True                     # Whether to clear the tiff buffer after processing

if __name__ == "__main__":

    # Set backend to libvips (the default backend cucim does not support .png images)
    os.environ["SF_SLIDE_BACKEND"] = "libvips"

    # Extract all WSI from Dataset (without thumbnails)
    slide_list: list[Path] = []
    for root, _, file_names in os.walk(DATASET_PATH / "train_images"):
        slide_list.extend([Path(root) / file_name for file_name in file_names if file_name.endswith('.png') and not '_thumbnail' in file_name])

    # Load the project & metadata
    project = sf.load_project(PROJECT_PATH)

    # Load dataset object from project
    ubc = project.dataset(
        tile_px=512,
        tile_um="20x",
        config=(PROJECT_PATH / "datasets.json").name,
        sources="TIFF_BUFFER"
    )

    # Separate Slides into batches
    for batch_idx, batch in enumerate(batch_generator(slide_list, batch_size=10)):

        print(f"--- Batch {batch_idx} ---")

        # Skipping fully completed batches
        if all([(TFRECORD_PATH / "512px_20x" / f"{slide.stem}.tfrecords").exists() for slide in batch]):
            print("Slides in batch already processed. Skipping")
            continue

        # Convert batch to tiffs
        tiff_list = batch_conversion(batch, TIFF_BUFFER_PATH, verbose=True)

        try:
            # extract tiles into datasets tile folder using normalizer
            ubc.extract_tiles(
                qc=qc.Otsu(),
                normalizer="reinhard_mask",
                report=False
            )

        except Exception as err:
            print(err)

        if CLEAR_BUFFER:
            for tiff in [file for file in TIFF_BUFFER_PATH.iterdir() if file.suffix == ".tiff"]:
                os.remove(tiff)
            print("Cleared tiff buffer")



    
