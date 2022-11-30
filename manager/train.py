import json
import logging
import os

import pandas as pd

from manager.config import Kwargs
from manager.data.data_split import split_data
from manager.training.grid_search_cv import grid_search_cv
from manager.training.train_best_parameters import best_model
from manager.utils import NewJsonEncoder, get_thresh

logger = logging.getLogger(__name__)


def train_models(drug_name: str, drug_info: dict, kwargs: Kwargs):
    """
    drug_info = {$drug_name: {"gene_mat": $gene_mat, "meta_data": $meta_data}}
        gene_mat = pd.DataFrame(..., columns=[cell_lines: List[str]], index= [genes: List[int])
        meta_data = pd.DataFrame(..., columns=["ic_50", "labels"], index= [cell_lines: List[int]])
    return None, outputs results to a file
    """

    if kwargs.training.weight_samples:
        # retrieve the discretization threshold of the current drug
        drug_thresh = get_thresh(drug_name, kwargs.data.processed_files.thresholds)
        kwargs.data.drug_threshold = drug_thresh  # globalize
    kwargs.data.drug_name = drug_name  # globalize

    logger.info(f"*** Splitting data into train and test sets for {drug_name} ***")
    splits = split_data(drug_info["gene_mat"], drug_info["meta_data"], kwargs)

    # start training and output results to json file
    all_models = {"parameters_grid": kwargs.training.parameters_grid}
    for specified_model in kwargs.model.model_names:
        kwargs.model.current_model = specified_model

        train_features = splits["train"]["data"]
        train_classes = splits["train"]["classes"]
        train_scores = splits["train"]["scores"]
        test_features = splits["test"]["data"]
        test_classes = splits["test"]["classes"]
        test_scores = splits["test"]["scores"]

        if kwargs.training.regression:  # set labels to the ic-50 scores
            train_labels = splits["train"]["scores"]
            test_labels = splits["test"]["scores"]
        else:  # set labels to the discretized labels
            train_labels = splits["train"]["classes"]
            test_labels = splits["test"]["classes"]

        if kwargs.training.test_average:
            train_features = pd.concat([train_features, test_features])
            train_labels = pd.concat([train_labels, test_labels])
            train_classes = pd.concat([train_classes, test_classes])
            train_scores = pd.concat([train_scores, test_scores])
            test_features = test_labels = None

        if kwargs.training.grid_search:
            logger.info(
                f"=== Starting grid search cross validation for {drug_name} using {specified_model.upper()} model ==="
            )
            all_models[specified_model] = grid_search_cv(
                train_features=train_features,
                train_labels=train_labels,
                train_classes=train_classes,
                train_scores=train_scores,
                test_features=test_features,
                test_labels=test_labels,
                test_classes=test_classes,
                kwargs=kwargs,
            )
        else:
            all_models[specified_model] = {}
            for idx, rf_params in kwargs.training.parameters_grid.items():
                all_models[specified_model][idx] = best_model(
                    rf_params,
                    train_features=train_features,
                    train_labels=train_labels,
                    train_classes=train_classes,
                    train_scores=train_scores,
                    test_features=test_features,
                    test_labels=test_labels,
                    test_classes=test_classes,
                    kwargs=kwargs,
                )

    not_to_analyse = kwargs.data.not_to_analyse
    if len(not_to_analyse) > 0:
        if len(not_to_analyse) > 0:
            not_to_analyse = "\n".join(not_to_analyse)
        else:
            not_to_analyse = str(not_to_analyse)
        with open(os.path.join(kwargs.results_dir, "dummy_result.csv"), "a") as dummy:
            dummy.write(not_to_analyse)

    with open(kwargs.results_doc, "a") as convert_file:
        convert_file.write(
            json.dumps({drug_name: all_models}, indent=2, cls=NewJsonEncoder)
        )
        convert_file.write("\n")
