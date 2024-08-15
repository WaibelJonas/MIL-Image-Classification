import json
from utils import PROJECT_PATH
import slideflow as sf
from slideflow.mil import mil_config, train_mil, eval_mil
import os

with open("/share/praktikum2024/train_test_split.json", "r") as f:
        splits = json.load(f)
project = sf.load_project(PROJECT_PATH)

entry: dict = splits[0]

for key, value in entry.items():
    print(f"{key}: {value}")

"""
train_slides = list(splits['splits']['train'].keys())
val_slides = list(splits['splits']['val'].keys())
test_slides = list(splits['splits']['test'].keys())

train_dataset = project.dataset(
            tile_px=512,
            tile_um=20,
            filters={'slide': train_slides}
        )

val_dataset = project.dataset(
            tile_px=512,
            tile_um=20,
            filters={'slide': val_slides}
        )
test_dataset = project.dataset(
            tile_px=512,
            tile_um=20,
            filters={'slide': test_slides}
        )

config = mil_config(
            model='attention_mil',
            lr=1e-4,
            fit_one_cycle=True 
        )

outdir = os.path.join(PROJECT_PATH, 'hannasSplit')
outcomes = 'label' 
bags = '/storage/emilio/slideflow_project/bags'
train_mil(
            config=config,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            outcomes=outcomes,
            bags=bags,
            outdir=outdir
        )
"""