import pandas as pd
from pathlib import Path
from legacy.DataSplitter import DataSplitter

"""
Splits train.csv from UBC-OCEAN into distinct train, test and validation splits using the DataSplitter Class
"""

# Root Project folder (borrowed from utils.py)
PROJECT_PATH: Path = Path("/data/waibeljo/MIL-Image-Classification")

# Path to dataset (borrowed from utils.py)
DATASET_PATH: Path = Path("/share/UBC-OCEAN")

# Balance classes in splits
BALANCE: bool = True

if __name__ == "__main__":

    # Create train, test and validation split from source .csv
    ds = DataSplitter(csv_path=DATASET_PATH / "train.csv",
                      label_col="label",
                      index_col="image_id",
                      train_frac=0.7,
                      test_frac=0.2,
                      validation_frac=0.1,
                      balance_classes=BALANCE)

    # Displaying class distribution in the splits
    splits = {"Original": pd.read_csv(DATASET_PATH / "train.csv", index_col="image_id"), "Train": ds.train, "Test": ds.test, "Validation": ds.validation}

    for split_name, df in splits.items():
        class_counts = df["label"].value_counts()

        print(f"{split_name}:")
        if split_name != "Original": print(f"(Fraction of Original: {round(df.shape[0] / splits['Original'].shape[0], ndigits=1)})")

        for class_name, count in class_counts.items():
            print(f"\tClass: {class_name:<8} Count: {count:^8} ~ Fraction: {round(count / df.shape[0], ndigits=2):>8}")
    
    # Assure that there are no overlaps
    train_test_overlap = len(set(ds.train.index) & set(ds.test.index))
    train_val_overlap = len(set(ds.train.index) & set(ds.validation.index))
    val_test_overlap = len(set(ds.validation.index) & set(ds.test.index))

    print(f"Overlap between train and test set:         {train_test_overlap}")
    print(f"Overlap between train and validation set:   {train_val_overlap}")
    print(f"Overlap between validation and test set:    {val_test_overlap}")

    # Ensure the splits are distinct
    if all(overlap == 0 for overlap in [train_test_overlap, train_val_overlap, val_test_overlap]):
        # Save splits as .csvs
        ds.save(train_path=PROJECT_PATH / "data" / "train.csv",
                test_path=PROJECT_PATH / "data" / "test.csv",
                validation_path=PROJECT_PATH / "data" / "validation.csv")
    
