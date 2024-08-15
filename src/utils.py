import os
import pyvips
import numpy as np
from pathlib import Path
from time import time
from dataclasses import dataclass

from pandas import (
    DataFrame,
    read_csv,
    concat,
)

from typing import (
    Generator,
    TypeVar,
    Union,
)

"""
Utility Methods and Variables
"""

# --- Project Variables ---

# Random seed for reproducing results
RANDOM_SEED: int = 42

# Root Project folder
PROJECT_PATH: Path = Path("/data/waibeljo/MIL-Image-Classification")

# Path to bags folder
BAGS_PATH: Path = PROJECT_PATH / "bags"

# Path to models folder
MODEL_PATH: Path = PROJECT_PATH / "models"

# Path to dataset
DATASET_PATH: Path = Path("/share/UBC-OCEAN")

# Path to share folder
SHARE_PATH: Path = Path("/share/praktikum2024")

# Path to TFRecord folder
TFRECORD_PATH: Path = SHARE_PATH / "tfrecords"

# Generic Type for Annotations
T = TypeVar('T')

# --- Utility methods ---

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

# --- Data Splitting ---

def _equal_balance_sampler(sample_fraction: float, dataframes: list[DataFrame],
                           drop: bool = True, shuffle: bool = True,
                           random_seed: int = 42) -> tuple[DataFrame, list[DataFrame]]:
    """Takes a number of dataframes and samples from them in equal parts, with or without replacement.

    Args:
        sample_fraction     (float):                Sample fraction
        dataframes          (list[DataFrame]):      List of dataframes to sample from
        drop                (bool, optional):       Whether to drop sampled entries (no replacement). Defaults to True.
        shuffle             (bool, optional):       Whether to shuffle the resulting sample. Defaults to True.
        random_seed         (int, optional):        Random state for reproducing results

    Returns:
        tuple[DataFrame, list[DataFrame]]: Tuple of the sample and the remaining dataframes
    """
    result = DataFrame()

    # Iterate over dataframes
    for index, df in enumerate(dataframes):
        # Take sample
        sample = df.sample(frac=sample_fraction, replace=False, random_state=random_seed)
        # Drop samples from original
        if drop: dataframes[index] = df.drop(sample.index)
        # Add to result
        result = concat([result, sample])
    
    # Shuffle
    if shuffle: result = result.sample(frac=1, random_state=RANDOM_SEED)

    return result, dataframes


def _simple_sampler(sample_fraction: float, dataframe: DataFrame,
                    drop: bool = True, shuffle: bool = True,
                    random_seed: int = 42 ) -> tuple[DataFrame, DataFrame]:
    """Takes a sample from a dataframe with or without replacement.

    Args:
        sample_fraction     (float):                Sample fraction to take
        dataframe           (DataFrame):            Dataframe to sample from
        drop                (bool, optional):       Whether to drop sampled entries (no replacement). Defaults to True.
        shuffle             (bool, optional):       Whether to shuffle the resulting sample. Defaults to True.
        random_seed         (int, optional):        Random state for reproducing results

    Returns:
        tuple[DataFrame, DataFrame]: Tuple of the sample and the remaining dataframe
    """
    result = DataFrame()

    # Take sample
    result = dataframe.sample(frac=sample_fraction, replace=False, random_state=random_seed)
    # Drop samples from original
    if drop: dataframe = dataframe.drop(result.index)
    # Shuffle
    if shuffle: result = result.sample(frac=1, random_state=random_seed)

    return result, dataframe


@dataclass
class DataSplitter:

    train       :   DataFrame
    test        :   DataFrame
    validation  :   DataFrame

    def __init__(self, csv_path: Path, label_col: str, index_col: str,
                 train_frac: float = 0.6, test_frac: float = 0.2,
                 validation_frac: float = 0.2, balance_classes: bool = False,
                 random_seed: int = 42):
        """Splits a pandas Dataframe into a training, testing and validation
        subset. 

        Args:
            csv_path            (Path):                     Path to a csv file from which to construct the Dataframe
            label_col           (str):                      Name of the label column
            index_col           (str):                      Name of the index column
            train_frac          (float, optional):          Fraction of the train subset. Defaults to 0.6.
            test_frac          (float, optional):          Fraction of the test subset. Defaults to 0.2.
            validation_frac     (float, optional):          Fraction of the validation subset. Defaults to 0.2.
            balance_classes     (bool, optional):           Whether to balance the class distribution. Defaults to False.
            random_seed         (int, optional):            Random state for reproducing results
        
        Attributes:
            train               DataFrame:      Dataframe of train split
            test                DataFrame:      Dataframe of test split
            validation          Dataframe:      Dataframe of validation split

        Raises:
            FileNotFoundError:                  If the given csv_path does not exist
            ValueError:                         If the sum of the fractions is not 1 or a fraction is outside
                                                of the 0.0-1.0 range
        """
        # Error Handling
        if not csv_path.exists():
            raise FileNotFoundError(f"{csv_path.name} not found")
        fractions = [train_frac, test_frac, validation_frac]
        if sum(fractions) - 1.0 > 0.001:
            raise ValueError("Sum of split fractions is not one")
        if any([frac > 1.0 or frac < 0.0 for frac in fractions]):
            raise ValueError("Given split fraction not in 0-1 range")
        
        # Load dataset
        data = read_csv(csv_path, index_col=index_col)
        
        # Balanced approach
        if balance_classes:
            class_dfs: list[DataFrame] = []
            # Iterate over classes and split into dataframes
            for cls in np.unique(data[label_col]):
                class_dfs.append(data.loc[data[label_col] == cls])

            # Split train set
            self.train, class_dfs = _equal_balance_sampler(train_frac, class_dfs, random_seed=random_seed)
            # Adjust test fraction to size of remaining data
            test_frac = test_frac / (test_frac + validation_frac)
            # Split test set
            self.test, class_dfs = _equal_balance_sampler(test_frac, class_dfs, random_seed=random_seed)
            # Split validation set (rest of data)
            self.validation, _ = _equal_balance_sampler(1, class_dfs, random_seed=random_seed)
        
        # Simple approach
        else:
            # Split train set
            self.train, data = _simple_sampler(train_frac, data, random_seed=random_seed)
            # Adjust test fraction to size of remaining data
            test_frac = test_frac / (test_frac + validation_frac)
            self.test,  data = _simple_sampler(test_frac, data, random_seed=random_seed)
            # Split validation set (rest of data)
            self.validation, data = _simple_sampler(1, data, random_seed=random_seed)
    

    def save(self, train_path: Path, test_path: Path, validation_path: Path) -> bool:
        """Saves the tain, test, valdation split at the given paths

        Args:
            train_path (Path):      Path to save train split to
            test_path (Path):       Path to save test split to
            validation_path (Path): Path to save validation split to

        Returns:
            bool: True if successfull, False if not
        """
        try:
            self.train.to_csv(train_path)
            self.test.to_csv(test_path)
            self.validation.to_csv(validation_path)
            return True
        except:
            return False
