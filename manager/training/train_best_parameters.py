import time

import pandas as pd
from manager.config import Kwargs
from manager.models.random_forest import which_rf


def best_model(
    best_params: dict,
    train_features: pd.DataFrame,
    train_labels: pd.Series,
    train_classes: pd.Series,
    train_scores: pd.Series,
    test_features: pd.DataFrame,
    test_labels: pd.Series,
    test_classes: pd.Series,
    kwargs: Kwargs,
):
    """
    train and test the-best-parameters/model
    *_features = pd.DataFrame(..., columns=[genes: List[Any], index=[cell_lines: List[int]])
    *_labels = pd.Series(..., columns=["ic_50" if regression else "labels"], index=[cell_lines: List[int]])
    *_classes = pd.Series(..., columns=["labels"], index=[cell_lines: List[int]])
    train_scores = pd.Series(..., columns=["ic_50"], index=[cell_lines: List[int]])
    return: dict of results for the final model
    """

    start = time.time()
    (
        fit_runtime,
        test_runtime,
        acc,
        sorted_features,
        num_features_overall,
        num_features,
        num_trees_features,
    ) = which_rf(
        rf_params=best_params,
        train_features=train_features,
        train_classes=train_classes,
        train_scores=train_scores,
        train_labels=train_labels,
        test_features=test_features,
        test_labels=test_labels,
        test_classes=test_classes,
        kwargs=kwargs,
        output_preds=True,
    )

    # output the selected number of features to be used for the random model
    if "subgraphilp" in kwargs.model.current_model:
        with open(kwargs.subgraphilp_num_features_output_file, "a") as out_num_features:
            out_num_features.write(f"{kwargs.data.drug_name},{num_features}\n")

    return {
        "params": best_params,
        "train_runtime": fit_runtime,
        "test_runtime": test_runtime,
        "model_runtime": time.time() - start,
        "test_scores": acc,
        "features_importance": sorted_features,
        "num_features_overall": num_features_overall,
        "num_features": num_features,
        "num_tress_features": num_trees_features,
    }
