"""An example run file which trains a dummy AutoML system on the training split of a dataset
and logs the accuracy score on the test set.

In the example data you are given access to the labels of the test split, however
in the test dataset we will provide later, you will not have access
to this and you will need to output your predictions for the images of the test set
to a file, which we will grade using github classrooms!
"""
from __future__ import annotations

from pathlib import Path
import argparse
import torch
from qtt.utils import extract_image_dataset_metadata
from qtt.factory import get_optimizer
from qtt.tuners import QuickTuner
from qtt.finetune.cv.classification import finetune_script

import logging

from src.datasets import FashionDataset, FlowersDataset, EmotionsDataset

logger = logging.getLogger(__name__)


def main(
    dataset: str,
    budget: int,
    output_path: Path,
    seed: int,
):
    match dataset:
        case "fashion":
            dataset_class = FashionDataset
        case "flowers":
            dataset_class = FlowersDataset
        case "emotions":
            dataset_class = EmotionsDataset
        case _:
            raise ValueError(f"Invalid dataset: {args.dataset}")

    logger.info("Fitting AutoML")

    dataset = dataset_class(
        root="./data",
        download=True,
    )

    opt = get_optimizer("mtlbm/micro")
    opt.metafeatures = torch.tensor(
        extract_image_dataset_metadata("data/" + dataset.dataset_name).to_numpy(), dtype=torch.float
    )
    qt = QuickTuner(opt, finetune_script)
    task_info = {
        "data_path": "data/" + dataset.dataset_name,
        "train-split": "train",
        "val-split": "val",
        "num-classes": dataset.num_classes,
    }
    qt.run(task_info=task_info, time_budget=budget)

    config, score, cost, config_id = qt.get_incumbent()
    logger.info(f"Done! Config {config_id} performed best on validation data and got score {score}.")

    # # Do the same for the test dataset
    # test_preds, test_labels = automl.predict(dataset_class)
    #
    # # Write the predictions of X_test to disk
    # # This will be used by github classrooms to get a performance
    # # on the test set.
    # logger.info("Writing predictions to disk")
    # with output_path.open("wb") as f:
    #     np.save(f, test_preds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="The name of the dataset to run on.",
        choices=["fashion", "flowers", "emotions"]
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("output/"),
        help=(
            "The path to save the predictions to."
        )
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=3600,
        help=(
            "Budget in seconds"
        )
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help=(
            "Random seed for reproducibility if you are using and randomness,"
            " i.e. torch, numpy, pandas, sklearn, etc."
        )
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Whether to log only warnings and errors."
    )

    args = parser.parse_args()

    if not args.quiet:
        logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(level=logging.WARNING)

    logger.info(
        f"Running dataset {args.dataset}"
        f"\n{args}"
    )

    main(
        dataset=args.dataset,
        budget=args.budget,
        output_path=args.output_path,
        seed=args.seed,
    )