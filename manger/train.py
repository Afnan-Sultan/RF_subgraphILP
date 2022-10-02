import json
import logging

from manger.config import Kwargs
from manger.data.data_split import split_data
from manger.training.grid_search_cv import grid_search_cv
from manger.training.train_best_parameters import best_model
from manger.utils import NewJsonEncoder, get_thresh

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

        if kwargs.training.regression:  # set labels to the ic-50 scores
            train_labels = splits["train"]["scores"]
            test_labels = splits["test"]["scores"]
        else:  # set labels to the discretized labels
            train_labels = splits["train"]["classes"]
            test_labels = splits["test"]["classes"]

        if kwargs.training.grid_search:
            logger.info(
                f"=== Starting grid search cross validation for {drug_name} using {specified_model.upper()} model ==="
            )
            all_models[specified_model] = grid_search_cv(
                train_features=splits["train"]["data"],
                train_labels=train_labels,
                train_classes=splits["train"]["classes"],
                train_scores=splits["train"]["scores"],
                test_features=splits["test"]["data"],
                test_labels=test_labels,
                test_classes=splits["test"]["classes"],
                kwargs=kwargs,
            )
        else:
            all_models[specified_model] = {}
            for idx, rf_params in kwargs.training.parameters_grid.items():
                all_models[specified_model][idx] = best_model(
                    rf_params,
                    train_features=splits["train"]["data"],
                    train_labels=train_labels,
                    train_classes=splits["train"]["classes"],
                    train_scores=splits["train"]["scores"],
                    test_features=splits["test"]["data"],
                    test_labels=test_labels,
                    test_classes=splits["test"]["classes"],
                    kwargs=kwargs,
                )

    with open(kwargs.results_doc, "a") as convert_file:
        convert_file.write(
            json.dumps({drug_name: all_models}, indent=2, cls=NewJsonEncoder)
        )
        convert_file.write("\n")
