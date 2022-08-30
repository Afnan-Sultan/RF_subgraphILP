import statistics
from typing import List

import numpy as np
import pandas as pd


def get_sensitivity(
    test_classes: pd.Series, label: int, true_labels: pd.Series, prediction: np.array
):
    """
    split the true and predicted results into sensitive/resistant
    """
    temp = (
        test_classes.reset_index()
    )  # to get numerical indices for sub-setting `predictions' instead of cell_lines
    idxs = temp[temp.values == label].index
    true = true_labels.iloc[idxs]
    pred = prediction[idxs]
    return true, pred


def splits_stats(model_results: dict, regression: bool):
    """
    calculate the statistics of the cross validation folds results
    """
    ACCs_stats = {}
    subsets = [
        "_".join(metric.split("_")[1:])
        for metric in model_results.keys()
        if metric.startswith("ACCs")
    ]
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


def subsets_means(cv_results: dict, params_mean_perf: dict, regression: bool, idx: int):
    """
    calculate the results for the subsets of sensitive/resistant cell lines
    """
    for model, results in cv_results.items():
        subsets = [
            "_".join(metric.split("_")[1:])
            for metric in results.keys()
            if metric.startswith("ACCs")
        ]
        if model not in params_mean_perf:
            params_mean_perf[model] = {}
        for subset in subsets:
            if subset in params_mean_perf[model]:
                if regression:
                    params_mean_perf[model][subset][idx] = cv_results[model]["stats"][
                        f"{subset}_mean_mse"
                    ]
                else:
                    params_mean_perf[model][subset][idx] = cv_results[model]["stats"][
                        f"{subset}_mean"
                    ]
            else:
                if regression:
                    params_mean_perf[model][subset] = {
                        idx: cv_results[model]["stats"][f"{subset}_mean_mse"]
                    }
                else:
                    params_mean_perf[model][subset] = {
                        idx: cv_results[model]["stats"][f"{subset}_mean"]
                    }
    return params_mean_perf
