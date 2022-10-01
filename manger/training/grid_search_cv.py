import logging
import time

import pandas as pd
from manger.config import Kwargs
from manger.scoring_metrics.utils import rank_parameters, subsets_means
from manger.training.cross_validation import cross_validation, get_rand_cv_results
from manger.training.train_best_parameters import best_model

logger = logging.getLogger(__name__)


def grid_search_cv(
    train_features: pd.DataFrame,
    train_labels: pd.Series,
    train_classes: pd.Series,
    train_scores: pd.Series,
    test_features: pd.DataFrame,
    test_labels: pd.Series,
    test_classes: pd.Series,
    kwargs: Kwargs,
    out_log=True,
):
    """
    perform grid search cross validation.
    *_features = pd.DataFrame(..., columns=[genes: List[Any], index=[cell_lines: List[int]])
    *_labels = pd.Series(..., columns=["ic_50" if regression else "labels"], index=[cell_lines: List[int]])
    *_classes = pd.Series(..., columns=["labels"], index=[cell_lines: List[int]])
    train_scores = pd.Series(..., columns=["ic_50"], index=[cell_lines: List[int]])
    return: dict of results for each hyperparameters combination
    """
    parameters_grid = kwargs.training.parameters_grid
    model = kwargs.model.current_model

    # ensure the column names are strings to be recognized as feature names for random forest
    train_features.columns = train_features.columns.astype(str)
    test_features.columns = test_features.columns.astype(str)

    # grid search with cross validation
    start = time.time()
    gcv_results = {}
    params_mean_perf = {}
    for idx, rf_params in parameters_grid.items():
        if out_log:
            logger.info(
                f"--- {kwargs.data.drug_name} - parameters combination {idx} ---"
            )

        kwargs.training.gcv_idx = idx
        if model == "random" and kwargs.training.bias_rf:
            cv_results = get_rand_cv_results(
                rf_params=rf_params,
                train_features=train_features,
                train_labels=train_labels,
                train_classes=train_classes,
                train_scores=train_scores,
                kwargs=kwargs,
                out_log=False,
            )
        else:
            cv_results = cross_validation(
                rf_params=rf_params,
                train_features=train_features,
                train_labels=train_labels,
                train_classes=train_classes,
                train_scores=train_scores,
                kwargs=kwargs,
                out_log=out_log,
            )
        if "parameters_combo_cv_results" in gcv_results:
            gcv_results["parameters_combo_cv_results"][idx] = cv_results
        else:
            gcv_results["parameters_combo_cv_results"] = {idx: cv_results}

        # update params_mean_perf for the current fold
        subsets_means(cv_results, params_mean_perf, kwargs.training.regression, idx)

    # update gcv_results with parameters rank and retrieve the best parameters (based on sensitive cell lines scores)
    best_params = rank_parameters(
        parameters_grid, gcv_results, params_mean_perf, kwargs
    )
    gcv_results["gcv_runtime"] = time.time() - start
    logger.info(f"--- {kwargs.data.drug_name} - finished CROSS VALIDATION ---")

    # ========================== #

    # train best model with the best parameters from cross validation
    gcv_results["best_params_performance"] = best_model(
        best_params,
        train_features,
        train_labels,
        train_classes,
        train_scores,
        test_features,
        test_labels,
        test_classes,
        kwargs,
    )
    logger.info(
        f"&&& {kwargs.data.drug_name} - finished training/testing using best parameters combination "
        f"with scores for sensitive cell line: "
        f"{gcv_results['best_params_performance']['test_scores']['sensitivity']}&&&"
    )
    return gcv_results
