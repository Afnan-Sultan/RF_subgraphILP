import logging
import time

import pandas as pd
from manager.config import Kwargs
from manager.training.cross_validation import cross_validation, get_rand_cv_results
from manager.training.train_best_parameters import best_model

logger = logging.getLogger(__name__)


def subsets_means(
    cv_results: dict, params_mean_perf: dict, subsets: list, regression: bool, idx: int
):
    """
    calculate the results for the subsets of sensitive/resistant cell lines
    """
    for subset in subsets:
        if subset in params_mean_perf:
            if regression:
                params_mean_perf[subset][idx] = cv_results["stats"][
                    f"{subset}_mean_mse"
                ]
            else:
                params_mean_perf[subset][idx] = cv_results["stats"][f"{subset}_mean"]
        else:
            if regression:
                params_mean_perf[subset] = {
                    idx: cv_results["stats"][f"{subset}_mean_mse"]
                }
            else:
                params_mean_perf[subset] = {idx: cv_results["stats"][f"{subset}_mean"]}


def rank_parameters(parameters_grid, gcv_results, params_mean_perf, kwargs):

    # rank the mean performance for each parameter per subset (subsets = {overall, sensitive, resistant} if regression
    # or subsets = {sensitivity, specificity, f1. ...etc.} if classification)
    rank_params = {}
    if kwargs.training.regression:
        reverse = False  # the lower score, the better
    else:
        reverse = True  # the higher score, the better
    for subset, mean_scores in params_mean_perf.items():
        rank_params[subset] = {
            param_idx: score
            for param_idx, score in sorted(
                mean_scores.items(), key=lambda item: item[1], reverse=reverse
            )
        }
    gcv_results["rank_params_scores"] = rank_params

    # retrieve the best parameters based on performance on the sensitive cell lines
    # ("mse fo sensitive cell lines" if regression or "sensitivity" if classification)
    best_params = [
        parameters_grid[
            list(rank.keys())[0]
        ]  # best parameter's index is the first key in rank_params
        for subset, rank in rank_params.items()
        if subset == "sensitivity"
    ][0]
    return best_params


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
    *_features = pd.DataFrame(..., columns=[genes: List[int], index=[cell_lines: List[int]])
    *_labels = pd.Series(..., columns=["ic_50" if regression else "labels"], index=[cell_lines: List[int]])
    *_classes = pd.Series(..., columns=["labels"], index=[cell_lines: List[int]])
    train_scores = pd.Series(..., columns=["ic_50"], index=[cell_lines: List[int]])
    return: dict of results for each hyperparameter combination
    """
    parameters_grid = kwargs.training.parameters_grid
    model = kwargs.model.current_model

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
        subsets_means(
            cv_results,
            params_mean_perf,
            kwargs.data.acc_subset,
            kwargs.training.regression,
            idx,
        )
    kwargs.training.gcv_idx = None

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
