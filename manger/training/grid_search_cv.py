import logging
import time

import pandas as pd

from manger.config import Kwargs
from manger.scoring_metrics.utils import subsets_means
from manger.training.cross_validation import (cross_validation,
                                              get_rand_cv_results)
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

    train_features.columns = train_features.columns.astype(str)
    test_features.columns = test_features.columns.astype(str)

    start = time.time()

    parameters_grid = kwargs.training.parameters_grid
    gcv_results = {}
    params_mean_perf = {}
    for idx, rf_params in parameters_grid.items():
        if out_log:
            logger.info(
                f"--- {kwargs.data.drug_name} - parameters combination {idx} ---"
            )

        kwargs.training.gcv_idx = idx
        if kwargs.model.current_model == "random" and kwargs.training.bias_rf:
            cv_results = get_rand_cv_results(
                rf_params,
                train_features,
                train_labels,
                train_classes,
                train_scores,
                kwargs,
                out_log=False,
            )
        else:
            cv_results = cross_validation(
                rf_params,
                train_features,
                train_labels,
                train_classes,
                train_scores,
                kwargs,
                out_log,
            )

        if not cv_results:
            return None

        for model, results in cv_results.items():
            if model in gcv_results:
                gcv_results[model]["parameters_combo_cv_results"][idx] = results
            else:
                gcv_results[model] = {"parameters_combo_cv_results": {idx: results}}

        params_mean_perf = subsets_means(
            cv_results, params_mean_perf, kwargs.training.regression, idx
        )

    # rank parameters based on performance of sensitive cell lines
    gcv_results, rank_params, best_params = process_grid_search_results(
        parameters_grid, gcv_results, params_mean_perf, kwargs
    )
    for model in gcv_results.keys():
        gcv_results[model]["gcv_runtime"] = time.time() - start
    logger.info(f"--- {kwargs.data.drug_name} - finished CROSS VALIDATION ---")

    # train best model
    gcv_results = best_params_model(
        gcv_results,
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
    return gcv_results


def process_grid_search_results(parameters_grid, gcv_results, params_mean_perf, kwargs):
    if kwargs.training.regression:
        reverse = False
    else:
        reverse = True
    rank_params = {}
    for model, subset_means in params_mean_perf.items():
        rank_params[model] = {}
        for subset, means in subset_means.items():
            rank_params[model][subset] = {
                param_idx: perf
                for param_idx, perf in sorted(
                    means.items(), key=lambda item: item[1], reverse=reverse
                )
            }

    for model in rank_params.keys():
        gcv_results[model]["rank_params_scores"] = rank_params[model]

    best_params = [
        parameters_grid[list(rank.keys())[0]]
        for model, subset_rank in rank_params.items()
        for subset, rank in subset_rank.items()
        if subset == "sensitivity"
    ][0]
    return gcv_results, rank_params, best_params


def best_params_model(
    gcv_results: dict,
    best_params: dict,
    train_features: pd.DataFrame,
    train_labels: pd.Series,
    train_classes: pd.Series,
    train_scores: pd.Series,
    test_features: pd.DataFrame,
    test_labels: pd.Series,
    test_classes: pd.Series,
    kwargs,
):
    start = time.time()
    test_results = best_model(
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
    best_model_runtime = time.time() - start
    if test_results is None:
        gcv_results[kwargs.model.current_model]["best_params_performance"] = None
        logger.info(f"&&& no features found to train best parameters combination &&&")
    else:
        for model, results in test_results.items():
            gcv_results[model]["best_params_performance"] = {
                "params": best_params,
                "model_runtime": best_model_runtime,
                "train_runtime": results[0],
                "test_runtime": results[1],
                "test_scores": results[2],
                "features_importance": results[3],
                "num_features": results[4],
            }
        logger.info(
            f"&&& {kwargs.data.drug_name} - finished training/testing using best parameters combination "
            f"with scores for sensitive cell line: "
            f"{gcv_results[kwargs.model.current_model]['best_params_performance']['test_scores']['sensitivity']}&&&"
        )
    return gcv_results
