from tiling import tiling
from bags import bags
from train import train
from evaluation import evaluate
from click import command, option

from utils import (
    TILE_SIZE,
    TILE_UM,
    DATASET_PATH,
    TFRECORD_PATH,
    BAGS_PATH,
    SPLIT_PATH,
    MODEL_PATH,
    CLEAR_BUFFER,
    TIFF_BUFFER_SIZE,
    FEATURE_EXTRACTOR,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    EVALUATION_PATH
)

@command()
@option("--skip-tiling",      is_flag=True,  default=False,   help="Skip tiling",      show_default=True)
@option("--skip-bags",        is_flag=True,  default=False,   help="Skip bags",        show_default=True)
@option("--skip-train",       is_flag=True,  default=False,  help="Skip train",       show_default=True)
@option("--skip-evaluation",  is_flag=True,  default=False,  help="Skip evaluation",  show_default=True)
def main(skip_tiling: bool = True, skip_bags: bool = True, skip_train: bool = False, skip_evaluation: bool = False):
    """
    Complete pipeline for the project.
    """

    if not skip_tiling:
        tiling(
            DATASET_PATH,
            TFRECORD_PATH,
            tile_px=TILE_SIZE,
            tile_um=TILE_UM,
            batch_size=TIFF_BUFFER_SIZE,
            clear_buffer=CLEAR_BUFFER
        )

    if not skip_bags:
        bags(
            tile_px=TILE_SIZE,
            tile_um=TILE_UM,
            feature_extractor=FEATURE_EXTRACTOR,
            outdir=BAGS_PATH,
        )

    if not skip_train:
        train(
            tile_px=TILE_SIZE,
            tile_um=TILE_UM,
            bags_path=BAGS_PATH,
            split_path=SPLIT_PATH,
            model_path=MODEL_PATH,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            learning_rate=LEARNING_RATE
        )

    if not skip_evaluation:
        evaluate(
            tile_px=TILE_SIZE,
            tile_um=TILE_UM,
            splits_path=SPLIT_PATH,
            model_path=MODEL_PATH,
            bags_path=BAGS_PATH,
            outdir=EVALUATION_PATH
        )

if __name__ == "__main__":
    main()