import numpy as np
from pandas import DataFrame, read_csv, concat
from dataclasses import dataclass
from pathlib import Path
from utils import RANDOM_SEED
from typing import Optional


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
    for df in dataframes:
        # Take sample
        sample = df.sample(frac=sample_fraction, replace=False, random_state=random_seed)
        # Drop samples from original
        if drop: df = df.drop(sample.index)
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

    train       :   Optional[DataFrame]
    test        :   Optional[DataFrame]
    validation  :   Optional[DataFrame]

    def __init__(self, csv_path: Path, label_col: str, index_col: str,
                 train_size: float = 0.6, test_size: float = 0.2,
                 validation_size: float = 0.2, balance_classes: bool = False,
                 random_seed: int = 42):
        """Splits a pandas Dataframe into a training, testing and validation
        subset. 

        Args:
            csv_path            (Path):                     Path to a csv file from which to construct the Dataframe
            label_col           (str):                      Name of the label column
            index_col           (str):                      Name of the index column
            train_size          (float, optional):          Fraction of the train subset. Defaults to 0.6.
            test_size           (float, optional):          Fraction of the test subset. Defaults to 0.2.
            validation_size     (float, optional):          Fraction of the validation subset. Defaults to 0.2.
            balance_classes     (bool, optional):           Whether to balance the class distribution. Defaults to False.
            random_seed         (int, optional):            Random state for reproducing results
        
        Attributes:
            train               (DataFrame, optional):      Optional Dataframe of train split, depending on the given train_fraction
            test                (DataFrame, optional):      Optional Dataframe of test split, depending on the given test_fraction
            validation          (Dataframe, optional):      Optional Dataframe of validation split, depending on the given validation_fraction

        Raises:
            FileNotFoundError:                  If the given csv_path does not exist
            ValueError:                         If the sum of the fractions is not 1 or a fraction is outside
                                                of the 0.0-1.0 range
        """
        # Error Handling
        if not csv_path.exists():
            raise FileNotFoundError(f"{csv_path.name} not found")
        fractions = [train_size, test_size, validation_size]
        if sum(fractions) != 1.0:
            raise ValueError("Sum of split fractions is not one")
        if any([frac > 1.0 or frac < 0.0 for frac in fractions]):
            raise ValueError("Given split fraction not in 0-1 range")
        
        # Load dataset
        data = read_csv(csv_path, index_col=index_col)
        
        # Balanced approach
        if balance_classes:
            class_dfs = []
            # Iterate over classes and split into dataframes
            for cls in np.unique(data[label_col]):
                class_dfs.append(data.loc[data[label_col] == cls])

            # Sample class dataframes into train, test, validation
            self.train, class_dfs      = _equal_balance_sampler(train_size,         class_dfs,      random_seed=random_seed) if train_size != 0.0 else None, class_dfs
            self.test, class_dfs       = _equal_balance_sampler(test_size,          class_dfs,      random_seed=random_seed) if test_size != 0.0 else None, class_dfs
            self.validation, class_dfs = _equal_balance_sampler(validation_size,    class_dfs,      random_seed=random_seed) if validation_size != 0.0 else None, class_dfs
        
        # Simple approach
        else:
            self.train, data      = _simple_sampler(train_size,         data,       random_seed=random_seed) if train_size != 0.0 else None, data
            self.test,  data      = _simple_sampler(test_size,          data,       random_seed=random_seed) if test_size != 0.0 else None, data
            self.validation, data = _simple_sampler(validation_size,    data,       random_seed=random_seed) if validation_size != 0.0 else None, data
    

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



        



