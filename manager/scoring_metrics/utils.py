import statistics

import numpy as np
import pandas as pd


def get_sensitivity(
    test_classes: pd.Series, label: int, true_labels: pd.Series, prediction: np.array
):
    """
    split the true and predicted results into sensitive/resistant
    """

    # convert to numerical indices for proper sub-setting of `predictions' as it doesn't contain cell lines IDs
    temp = test_classes.reset_index(drop=True)

    # select the samples of the specified class in both true and predicted labels
    idxs = temp[temp.values == label].index
    true = true_labels.iloc[idxs]
    pred = prediction[idxs]
    return true, pred


def splits_stats(model_results: dict, subsets: list, regression: bool):
    """
    calculate the statistics of the cross validation folds results
    """
    ACCs_stats = {}
    for subset in subsets:
        subset_ACCs = model_results[f"ACCs_{subset}"]
        if regression:
            ACCs_stats[f"{subset}_median_mse"] = statistics.median(subset_ACCs)
            ACCs_stats[f"{subset}_mean_mse"] = statistics.mean(subset_ACCs)
            ACCs_stats[f"{subset}_std_mse"] = statistics.stdev(subset_ACCs)
        else:
            scores_df = pd.DataFrame(subset_ACCs)
            scores_median = scores_df.median().to_dict()
            scores_means = scores_df.mean().to_dict()
            scores_std = scores_df.std().to_dict()
            for score_key, score_val in scores_means.items():
                ACCs_stats[f"{subset}_median"] = scores_median[score_key]
                ACCs_stats[f"{subset}_mean"] = score_val
                ACCs_stats[f"{subset}_std"] = scores_std[score_key]

    train_runtime = model_results["train_runtime"]
    ACCs_stats["train_time_median"] = statistics.median(train_runtime)
    ACCs_stats["train_time_mean"] = statistics.mean(train_runtime)
    ACCs_stats["train_time_std"] = statistics.stdev(train_runtime)

    test_runtime = model_results["test_runtime"]
    ACCs_stats["test_time_median"] = statistics.median(test_runtime)
    ACCs_stats["test_time_mean"] = statistics.mean(test_runtime)
    ACCs_stats["test_time_std"] = statistics.stdev(test_runtime)
    return ACCs_stats


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
