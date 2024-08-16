import os
import json
import warnings
import slideflow as sf
import numpy as np
from slideflow.mil import eval_mil
from sklearn.metrics import classification_report

from utils import PROJECT_PATH, SHARE_PATH, MODEL_PATH, BAGS_PATH

if __name__ == "__main__":

    os.environ["SF_LOGGING_LEVEL"] = "10"

    # Loading project and torch dataset
    project = sf.load_project(PROJECT_PATH)
    # Load torch dataset of thumbnails from project
    test = project.dataset(
        tile_px=512,
        tile_um="20x",
        config=(PROJECT_PATH / "datasets.json").name,
        sources="MIL-Image-Classification"
    )

    with open(SHARE_PATH / "splits.json", "r") as file_stream:
        splits = json.load(file_stream)
    
    current_fold = splits[0]
    test_slides = list(current_fold['splits']['test'].keys())
    test = test.filter({"slide": test_slides})

    with warnings.catch_warnings():
        warnings.simplefilter("ignore") # Ignore warnings

        evaluation = eval_mil(
            project=project,
            dataset=test,
            weights=str(MODEL_PATH / "00000-attention_mil-label"),
            outcomes="label",
            bags=str(BAGS_PATH),
            outdir=str(PROJECT_PATH / "evaluation"),
        )
    
    evaluation['y_pred'] = np.argmax(evaluation[['y_pred0', 'y_pred1', 'y_pred2', 'y_pred3', 'y_pred4']].values, axis=1)
    report = classification_report(evaluation['y_true'], evaluation['y_pred'], target_names=[str(i) for i in range(5)])

    print(report)