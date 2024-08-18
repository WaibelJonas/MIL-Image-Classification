import os
import pyvips
from pathlib import Path
from time import time
from typing import (
    Generator,
    TypeVar,
    Union,
)

"""
Variables and utility methods used throughout the project.
"""

# ------------------------- #
# --- Project Variables --- #
# ------------------------- #

# --- General ---

# Random seed for reproducing results
RANDOM_SEED: int = 42

# --- Paths ---

# Root Project folder
PROJECT_PATH: Path = Path("/data/waibeljo/MIL-Image-Classification")

# Path to Images folder
IMAGES_PATH: Path = PROJECT_PATH / "images"

# Path to bags folder
BAGS_PATH: Path = PROJECT_PATH / "bags"

# Path to models folder
MODEL_PATH: Path = PROJECT_PATH / "models"

# Evaluation Path
EVALUATION_PATH: Path = PROJECT_PATH / "evaluation"

# Path to dataset
DATASET_PATH: Path = Path("/share/UBC-OCEAN")

# Path to share folder
SHARE_PATH: Path = Path("/share/praktikum2024")

# Path to TFRecord folder
TFRECORD_PATH: Path = SHARE_PATH / "tfrecords"

# Path to splits folder
SPLIT_PATH: Path = SHARE_PATH / "splits.json"

# --- Tiling ---

# Tile Size in Pixels
TILE_SIZE: int = 512

# Magnification of the Tiles
TILE_UM: str = "20x"

# Whether to clear the tiff buffer after processing
CLEAR_BUFFER: bool = True

# TIFF buffer size
TIFF_BUFFER_SIZE: int = 10

# --- Bags ---

# Feature Extractor to use
FEATURE_EXTRACTOR: str = "histossl"

# --- Training ---

# Learning Rate
LEARNING_RATE: float = 1e-4

# Batch Size
BATCH_SIZE: int = 32

# Number of Epochs
EPOCHS: int = 40

# --- Evaluation ---



# ----------------------- #
# --- Utility methods --- #
# ----------------------- #

# --- Misc ---

def sum_file_sizes(path: Path) -> int:
    """Sums up the file sizes in a given path.

    Args:
        path (Path): Path to folder

    Returns:
        sum (int): Sum of file sizes in bytes
    """
    return sum([os.stat(str(file)).st_size for file in path.iterdir()])


def timer(show_mins: bool = False): 
    """Decorator for timing a method

    Args:
        func (function): method to time
    """
    def wrapper(func):
        def wrap_func(*args, **kwargs): 
            t1 = time() 
            result = func(*args, **kwargs) 
            t2 = time()
            elapsed_time = t2 - t1
            min_print = f"/{(elapsed_time / 60):.2f}min" if show_mins else ""
            print(f'Function {func.__name__!r} executed in {elapsed_time:.4f}s{min_print}') 
            return result 
        return wrap_func
    return wrapper

# Generic Type for Annotations
T = TypeVar('T')

def batch_generator(input_list: list[T], batch_size: int) -> Generator[list[T], None, None]: 
    """
    Generator that yields batches of a given size from the input list.
    
    Args:
        input_list (list): List of items to be batched.
        batch_size (int):  Size of each batch.
    
    Yields:
        batch (list):      A batch of items from the input list.
    """
    for i in range(0, len(input_list), batch_size):
        yield input_list[i:i + batch_size]

# --- PNG -> TIFF Conversion ---

def convert_img_to_tiff(in_path: Path, out_path: Path) -> Union[str, bool]:
    """Converts a given image into a .tiff file

    Args:
        in_path (Path):     Input path to image (.png, .jpg, ...)
        out_path (Path):    Output path to where to save the converted .tiff to
    
    Returns:
        str: error message in case of failure or an empty string in case of success
    """
    try:
        image: pyvips.Image = pyvips.Image.new_from_file(in_path)
        image.tiffsave(out_path, tile=True, pyramid=True, bigtiff=True)
    except Exception as err:
        return str(err)
    else:
        return ""
    

def batch_conversion(file_list: list[Path], out_folder: Path, verbose: bool = True) -> list[Path]:
    """Converts a list of image files to .tiff and saves them in the given output folder.

    Args:
        file_list (list[Path]):     List of file paths to convert
        out_folder (Path):          Path to output folder in which to save the tiffs.
                                    The tiff file will inherit the original files stem (i.e /input/image.png => /output/image.tiff)
        verbose (bool, optional):   Whether to log progress messages. Defaults to False.
    
    Returns:
        file_list (list[Path]):     List of paths to .tiff files
    """
    out_list = []   # List of output files
    v = verbose     # Shortened verbose flag

    for in_path in file_list:
        # Path to output file
        out_path = out_folder / f"{in_path.stem}.tiff"

        # Skip already converted images
        if out_path.exists():
            if v: print(f"{out_path} already exists. Skipping")
            out_list.append(out_path)
            continue
        
        # Convert image to tiff
        if err := convert_img_to_tiff(in_path, out_path):
            if v: print(f"Error while converting {in_path}: {err}")
        else:
            if v: print(f"Successfully converted {in_path.name}")
            out_list.append(out_path)
        
    return out_list
