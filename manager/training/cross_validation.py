import logging
import statistics

import pandas as pd
from manager.config import Kwargs
from manager.training.random_forest import which_rf
from sklearn.model_selection import StratifiedKFold

logger = logging.getLogger(__name__)


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


def cross_validation(
    rf_params: dict,
    train_features: pd.DataFrame,
    train_labels: pd.Series,
    train_classes: pd.Series,
    train_scores: pd.Series,
    kwargs: Kwargs,
    out_log=True,
):
    """
    rf_params: dict of parameters for sklearn.ensemble.RandomForestRegressor/Classifier
    train_features = pd.DataFrame(..., columns=[genes: List[Any], index=[cell_lines: List[int]])
    labels = pd.Series(..., columns=["ic_50" if regression else "labels"], index=[cell_lines: List[int]])
    classes = pd.Series(..., columns=["labels"], index=[cell_lines: List[int]])
    scores = pd.Series(..., columns=["ic_50"], index=[cell_lines: List[int]])
    return: dict of results for each cross fold
    """

    # cross validation splits accounts for class imbalance
    cv = StratifiedKFold(
        n_splits=kwargs.training.num_kfold,
        shuffle=True,
        random_state=kwargs.training.random_state,
    )

    cv_results = {}
    k = 0  # cross validation counter
    for train_index, test_index in cv.split(train_features, train_classes):
        kwargs.training.cv_idx = k  # globalize as it will be used for naming

        # identify labels for classification and ic50 for regression (ic50 will be used to calc corr w classification)
        cv_train_labels = train_labels.iloc[train_index]
        cv_train_classes = train_classes.iloc[train_index]
        cv_train_scores = train_scores.iloc[train_index]
        cv_test_labels = train_labels.iloc[test_index]
        cv_test_classes = train_classes.iloc[test_index]

        # get train and test splits
        cv_train_features = train_features.iloc[train_index]
        cv_test_features = train_features.iloc[test_index]

        (fit_runtime, test_runtime, acc, sorted_features, _, _, _,) = which_rf(
            rf_params=rf_params,
            train_features=cv_train_features,
            train_classes=cv_train_classes,
            train_scores=cv_train_scores,
            train_labels=cv_train_labels,
            test_features=cv_test_features,
            test_labels=cv_test_labels,
            test_classes=cv_test_classes,
            kwargs=kwargs,
        )

        if len(cv_results) > 1:
            if cv_results["important_features"] is not None:
                cv_results["important_features"].append(sorted_features[:10])
            cv_results["train_runtime"].append(fit_runtime)
            cv_results["test_runtime"].append(test_runtime)
            for subset, acc_score in acc.items():
                cv_results[f"ACCs_{subset}"].append(acc_score)
        else:
            cv_results = {
                "train_runtime": [fit_runtime],
                "test_runtime": [test_runtime],
                "important_features": None  # features not reported for the random model
                if sorted_features is None
                else [sorted_features[:10]],
            }
            for subset, acc_score in acc.items():
                cv_results[f"ACCs_{subset}"] = [acc_score]
        if out_log:
            logger.info(
                f"{kwargs.data.drug_name} - finished CV {k} with scores for sensitive cell lines: "
                f"{cv_results['ACCs_sensitivity'][k]}"
            )
        k += 1
    kwargs.training.cv_idx = None

    cv_results["stats"] = splits_stats(
        cv_results, kwargs.data.acc_subset, kwargs.training.regression
    )
    return cv_results


def get_rand_cv_results(
    rf_params: dict,
    train_features: pd.DataFrame,
    train_labels: pd.Series,
    train_classes: pd.Series,
    train_scores: pd.Series,
    kwargs: Kwargs,
    out_log: bool,
):
    """
    independent function for the random model, to average the performance and return single values as the other models
    invoked when bias_rf is selected, to train n rf models with each tree in a model limited to the average number
    of features selected from the subgraphilp model.

    rf_params = dict of parameters to be passed to sklearn.RandomForestRegressor/Classifier
    train_features = pd.DataFrame(..., columns=[genes: List[Any], index=[cell_lines: List[int]])
    train_labels = pd.Series(..., columns=["ic_50" if regression else "labels"], index=[cell_lines: List[int]])
    train_classes = pd.Series(..., columns=["labels"], index=[cell_lines: List[int]])
    train_scores = pd.Series(..., columns=["ic_50"], index=[cell_lines: List[int]])
    return dict of cross validation performance scores
    """
    rand_cv_results = {}
    for ctr in range(kwargs.training.num_random_samples):
        cv_results = cross_validation(
            rf_params,
            train_features,
            train_labels,
            train_classes,
            train_scores,
            kwargs,
            out_log,
        )
        subsets = kwargs.data.acc_subset
        if len(rand_cv_results) > 1:
            rand_cv_results["train_runtime"].append(cv_results["train_runtime"])
            rand_cv_results["test_runtime"].append(cv_results["test_runtime"])
            rand_cv_results["stats"].append(cv_results["stats"])
            for subset in subsets:
                rand_cv_results[f"ACCs_{subset}"].append(cv_results[f"ACCs_{subset}"])
        else:
            rand_cv_results = {
                "train_runtime": [cv_results["train_runtime"]],
                "test_runtime": [cv_results["test_runtime"]],
                "stats": [cv_results["stats"]],
            }
            for subset in subsets:
                rand_cv_results[f"ACCs_{subset}"] = [cv_results[f"ACCs_{subset}"]]

    for metric, values in rand_cv_results.items():
        if metric == "stats":
            rand_cv_results[metric] = pd.DataFrame(values).mean().to_dict()
        else:
            rand_cv_results[metric] = pd.DataFrame(values).mean().to_list()

    return rand_cv_results