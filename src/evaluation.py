import os
import json
import warnings
import slideflow as sf
import numpy as np
from pathlib import Path
from slideflow.mil import eval_mil
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay, 
    classification_report, balanced_accuracy_score,
    roc_auc_score, average_precision_score
)

from utils import (
    TILE_SIZE,
    TILE_UM,
    PROJECT_PATH,
    SPLIT_PATH,
    MODEL_PATH,
    BAGS_PATH,
    EVALUATION_PATH,
    IMAGES_PATH
)

LABELS = ["EC", "CC", "HGSC", "LGSC", "MC"]

def evaluate_fold(fold, index: int, model_path: Path, project: sf.Project, ubc: sf.Dataset, bags_path: Path, outdir: Path) -> tuple[str, np.ndarray, float, float, float]:
    """
    Evaluate a single fold of the model

    Parameters:
        fold: dict
            Fold information as received from split.json.
        index: int
            Fold index.
        model_path: Path
            Path to the model.
        project: sf.Project
            Project object.
        ubc: sf.Dataset
            Dataset object.
        bags_path: Path
            Path to the bags.
        outdir: Path
            Path to the output directory.
    """

    # Load test dataset and filter slides
    test_slides = list(fold['splits']['test'].keys())
    test = ubc.filter({"slide": test_slides})

    weight_path = model_path / f"0000{index}-attention_mil-label"

    # Disable warnings (else slideflow spams stdout)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        evaluation = eval_mil(
            project=project,
            dataset=test,
            weights=str(weight_path),
            outcomes="label",
            bags=str(bags_path),
            outdir=str(outdir),
        )

    # Convert predicted probabilities to class labels
    evaluation['y_pred'] = np.argmax(evaluation[['y_pred0', 'y_pred1', 'y_pred2', 'y_pred3', 'y_pred4']].values, axis=1)

    # Map ints to correct labels
    evaluation["true_label"] = evaluation["y_true"].map(lambda i: LABELS[i])
    evaluation["pred_label"] = evaluation["y_pred"].map(lambda i: LABELS[i])

    # Evaluation
    report              = classification_report(evaluation['y_true'], evaluation['y_pred'], target_names=LABELS)
    con_matrix          = confusion_matrix(evaluation['y_true'], evaluation['y_pred'])
    balanced_acc        = balanced_accuracy_score(evaluation['y_true'], evaluation['y_pred'])
    roc_auc             = roc_auc_score(evaluation['y_true'], evaluation[['y_pred0', 'y_pred1', 'y_pred2', 'y_pred3', 'y_pred4']].values, multi_class='ovr')
    avg_precision       = average_precision_score(evaluation['y_true'], evaluation[['y_pred0', 'y_pred1', 'y_pred2', 'y_pred3', 'y_pred4']].values, average='weighted')

    return report, con_matrix, balanced_acc, roc_auc, avg_precision


def evaluate(tile_px: int = TILE_SIZE, tile_um: str = TILE_UM,
             splits_path: Path = SPLIT_PATH, model_path: Path = MODEL_PATH,
             bags_path: Path = BAGS_PATH, outdir: Path = EVALUATION_PATH, image_path: Path = IMAGES_PATH):
    """
    Pipeline step 5: Evaluate MIL model

    -> Evaluates a MIL model using the slideflow library.\n
    -> Utilizes the split information in splits.json to filter the dataset into test.\n
    -> Ignoring warnings since slideflows eval_mil function produces a lot of them. (feel free to remove)\n

    Parameters:
        tile_px: int
            Size of the thumbnail in pixels.
        tile_um: str
            Size of the thumbnail in micrometers.
        splits_path: Path
            Path to the splits.
        model_path: Path
            Path to the model.
        BAGS_PATH: Path
            Path to the bags.
        outdir: Path
            Path to the output directory.
    """
    balanced_accuracy_list = []
    roc_auc_list           = []
    avg_precision_list     = []

    # Load project
    project = sf.load_project(PROJECT_PATH)

    # Load dataset
    ubc = project.dataset(
        tile_px=tile_px,
        tile_um=tile_um,
        config=(PROJECT_PATH / "datasets.json").name,
        sources="MIL-Image-Classification"
    )

    # Load splits
    with open(splits_path, "r") as file_stream:
        splits = json.load(file_stream)
    

    for index, fold in enumerate(splits):
        print(f"--- Fold {index} ---")

        report, con_matrix, balanced_acc, roc_auc, avg_precision = evaluate_fold(fold, index, model_path, project, ubc, bags_path, outdir)

        # Print classification report and save confusion matrix
        print(report)
        ConfusionMatrixDisplay(con_matrix, display_labels=LABELS).plot().figure_.savefig(image_path / f"confusion_matrix_fold{index}.png")

        # Print metrics
        print(f"Balanced accuracy: {balanced_acc:^20}")
        print(f"ROC AUC:           {roc_auc:^20}")
        print(f"Average precision: {avg_precision:^20}")

        # Save metrics to for calculating average
        balanced_accuracy_list.append(balanced_acc)
        roc_auc_list.append(roc_auc)
        avg_precision_list.append(avg_precision)
    
    # Print average metrics
    print("\n --- Average metrics ---")
    print(f"Average balanced accuracy: {np.mean(balanced_accuracy_list):^20}")
    print(f"Average ROC AUC:           {np.mean(roc_auc_list):^20}")
    print(f"Average average precision: {np.mean(avg_precision_list):^20}")
    

if __name__ == "__main__":
    evaluate(
        tile_px=TILE_SIZE,
        tile_um=TILE_UM,
        splits_path=SPLIT_PATH,
        model_path=MODEL_PATH,
        bags_path=BAGS_PATH,
        outdir=EVALUATION_PATH
    )