import logging

import numpy as np
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
    model = kwargs.model.current_model
    # train_labels = np.array(train_labels)
    k = 0
    cv_results = {}

    cv = StratifiedKFold(
        n_splits=kwargs.training.num_kfold,
        shuffle=True,
        random_state=kwargs.training.random_state,
    )
    for train_index, test_index in cv.split(train_features, train_classes):
        kwargs.training.cv_idx = k

        # identify labels for classification and scores for regression
        cv_train_labels = train_labels.iloc[train_index]
        cv_train_classes = train_classes.iloc[train_index]
        cv_train_scores = train_scores.iloc[
            train_index
        ]  # to pass for correlation models

        cv_test_labels = train_labels.iloc[test_index]
        cv_test_classes = train_classes.iloc[test_index]  # .reset_index(drop=True)

        # get train and test splits
        cv_train_features = train_features.iloc[train_index]
        cv_test_features = train_features.iloc[test_index]

        fit_runtime, test_runtime, acc, sorted_features, _ = which_rf(
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

        if (
            acc is None
        ):  # for cases like thresholded correlation, when no features available
            logger.info(
                f"{kwargs.data.drug_name} - finished CV {k} with scores for sensitive cell lines: not available"
            )
            continue

        if model in cv_results:
            if (
                cv_results[model]["important_features"] is not None
            ):  # features not reported for the random model
                cv_results[model]["important_features"].append(sorted_features[:10])
            cv_results[model]["train_runtime"].append(fit_runtime)
            cv_results[model]["test_runtime"].append(test_runtime)
            for subset, acc_score in acc.items():
                cv_results[model][f"ACCs_{subset}"].append(acc_score)
        else:
            cv_results[model] = {
                "train_runtime": [fit_runtime],
                "test_runtime": [test_runtime],
                # features not reported for the random model
                "important_features": None
                if sorted_features is None
                else [sorted_features[:10]],
            }
            for subset, acc_score in acc.items():
                cv_results[model][f"ACCs_{subset}"] = [acc_score]
        if out_log:
            logger.info(
                f"{kwargs.data.drug_name} - finished CV {k} with scores for sensitive cell lines: "
                f"{cv_results[model]['ACCs_sensitivity'][k]}"
            )
        k += 1
    kwargs.training.cv_idx = None
    if model in cv_results:
        cv_results[model]["stats"] = splits_stats(
            cv_results[model], kwargs.training.regression
        )
        return cv_results
    else:
        return None


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
    rf_params = dict of parameters to be passed to sklearn.RandomForestRegressor/Classifier
    train_features = pd.DataFrame(..., columns=[genes: List[Any], index=[cell_lines: List[int]])
    train_labels = pd.Series(..., columns=["ic_50" if regression else "labels"], index=[cell_lines: List[int]])
    train_classes = pd.Series(..., columns=["labels"], index=[cell_lines: List[int]])
    train_scores = pd.Series(..., columns=["ic_50"], index=[cell_lines: List[int]])
    return dict of cross validation performance scores
    """
    model = kwargs.model.current_model
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
        if model in rand_cv_results:
            rand_cv_results[model]["train_runtime"].append(
                cv_results[model]["train_runtime"]
            )
            rand_cv_results[model]["test_runtime"].append(
                cv_results[model]["test_runtime"]
            )
            subsets = [
                key.split("_")[1] for key in cv_results[model] if key.startswith("ACCs")
            ]
            for subset in subsets:
                rand_cv_results[model][f"ACCs_{subset}"].append(
                    cv_results[model][f"ACCs_{subset}"]
                )
        else:
            rand_cv_results[model] = {
                "train_runtime": [cv_results[model]["train_runtime"]],
                "test_runtime": [cv_results[model]["test_runtime"]],
            }
            subsets = [
                key.split("_")[1] for key in cv_results[model] if key.startswith("ACCs")
            ]
            for subset in subsets:
                rand_cv_results[model][f"ACCs_{subset}"] = [
                    cv_results[model][f"ACCs_{subset}"]
                ]

    for metric, values in rand_cv_results[model].items():
        rand_cv_results[model][metric] = pd.DataFrame(values).mean()

    return rand_cv_results
