from DataSplitter import DataSplitter
from utils import (
    DATASET_PATH,
    PROJECT_PATH
)

BALANCE: bool = True

if __name__ == "__main__":

    # Create train, test and validation split from source .csv
    ds = DataSplitter(csv_path=DATASET_PATH / "train.csv",
                      label_col="label",
                      index_col="image_id",
                      balance_classes=BALANCE)
    
    # Displaying class distribution in the splits
    splits = {"Train": ds.train, "Test": ds.test, "Validation": ds.validation}

    for split_name, df in splits.items():
        class_counts = df["label"].value_counts()

        print(f"{split_name}:")

        for class_name, count in class_counts.items():
            print(f"\tClass: {class_name:<8} Count: {count:^8} ~ Fraction: {round(count / df.shape[0], ndigits=2):>8}")

    # Save splits as .csvs
    ds.save(PROJECT_PATH / "data" / "train.csv",
            PROJECT_PATH / "data" / "test.csv",
            PROJECT_PATH / "data" / "validation.csv")
    
