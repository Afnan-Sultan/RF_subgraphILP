import logging

import pandas as pd
from manger.config import Kwargs
from manger.models.random_forest import which_rf
from manger.scoring_metrics.utils import splits_stats
from sklearn.model_selection import StratifiedKFold

logger = logging.getLogger(__name__)


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

        fit_runtime, test_runtime, acc, sorted_features, _, _ = which_rf(
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
            if (
                cv_results["important_features"] is not None
            ):  # features not reported for the random model
                cv_results["important_features"].append(sorted_features[:10])
            cv_results["train_runtime"].append(fit_runtime)
            cv_results["test_runtime"].append(test_runtime)
            for subset, acc_score in acc.items():
                cv_results[f"ACCs_{subset}"].append(acc_score)
        else:
            cv_results = {
                "train_runtime": [fit_runtime],
                "test_runtime": [test_runtime],
                # features not reported for the random model
                "important_features": None
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

    cv_results["stats"] = splits_stats(cv_results, kwargs.training.regression)
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
        if len(rand_cv_results) > 1:
            rand_cv_results["train_runtime"].append(cv_results["train_runtime"])
            rand_cv_results["test_runtime"].append(cv_results["test_runtime"])
            subsets = [
                key.split("_")[1] for key in cv_results if key.startswith("ACCs")
            ]
            for subset in subsets:
                rand_cv_results[f"ACCs_{subset}"].append(cv_results[f"ACCs_{subset}"])
        else:
            rand_cv_results = {
                "train_runtime": [cv_results["train_runtime"]],
                "test_runtime": [cv_results["test_runtime"]],
            }
            subsets = [
                key.split("_")[1] for key in cv_results if key.startswith("ACCs")
            ]
            for subset in subsets:
                rand_cv_results[f"ACCs_{subset}"] = [cv_results[f"ACCs_{subset}"]]

    for metric, values in rand_cv_results.items():
        rand_cv_results[metric] = pd.DataFrame(values).mean()

    return rand_cv_results
