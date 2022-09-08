import os
import statistics
import time
from typing import List, Union

import pandas as pd
from manger.config import Kwargs
from manger.models.utils.biased_random_forest import (
    BiasedRandomForestClassifier,
    BiasedRandomForestRegressor,
)
from manger.scoring_metrics.scoring import calc_accuracy
from manger.training.feature_selection import feature_selection
from manger.training.weighting_samples import (
    calculate_linear_weights,
    calculate_simple_weights,
)

# from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


def which_rf(
    rf_params: dict,
    train_features: pd.DataFrame,
    train_classes: pd.Series,
    train_scores: pd.Series,
    train_labels: pd.Series,
    test_features: pd.DataFrame,
    test_labels: pd.Series,
    test_classes: pd.Series,
    kwargs: Kwargs,
    output_preds: bool = False,
):
    """
    rf_params: dict of parameters for sklearn.ensemble.RandomForestRegressor/Classifier
    *_features = pd.DataFrame(..., columns=[genes: List[Any], index=[cell_lines: List[int]])
    *_labels = pd.Series(..., columns=["ic_50" if regression else "labels"], index=[cell_lines: List[int]])
    *_classes = pd.Series(..., columns=["labels"], index=[cell_lines: List[int]])
    train_scores = pd.Series(..., columns=["ic_50"], index=[cell_lines: List[int]])
    """

    model = kwargs.model.current_model
    if kwargs.training.bias_rf:
        features = train_features.columns.to_list()
        (
            fit_runtime,
            test_runtime,
            acc,
            sorted_features,
            num_features,
            num_trees_features,
        ) = do_rf(
            rf_params,
            features,
            train_features,
            train_labels,
            train_scores,
            train_classes,
            test_features,
            test_labels,
            test_classes,
            kwargs,
            output_preds,
        )
    else:
        to_rf = feature_selection(
            train_features, train_classes, train_scores, kwargs, test_features
        )
        if to_rf is None:
            return None, None, None, None, None, None

        if model == "random":
            fit_runtime, test_runtime, acc = get_rand_rf_results(
                rf_params,
                to_rf["features"],
                train_features,
                test_features,
                train_labels,
                test_labels,
                train_scores,
                train_classes,
                test_classes,
                kwargs,
            )
            sorted_features = num_features = num_trees_features = None
        else:
            (
                fit_runtime,
                test_runtime,
                acc,
                sorted_features,
                num_features,
                num_trees_features,
            ) = do_rf(
                rf_params,
                to_rf["features"],
                to_rf["train_features"],
                train_labels,
                train_scores,
                train_classes,
                to_rf["test_features"],
                test_labels,
                test_classes,
                kwargs,
                output_preds,
            )
    return (
        fit_runtime,
        test_runtime,
        acc,
        sorted_features,
        num_features,
        num_trees_features,
    )


def do_rf(
    rf_params: dict,
    features_names: Union[List, None],
    train_features: pd.DataFrame,
    train_labels: pd.Series,
    train_scores: pd.Series,
    train_classes: Union[pd.Series, None],
    test_features: pd.DataFrame,
    test_labels: pd.Series,
    test_classes: pd.Series,
    kwrags: Kwargs,
    output_preds=False,
):
    """
    rf_params: dict of parameters for sklearn.ensemble.RandomForestRegressor/Classifier
    *_features = pd.DataFrame(..., columns=[genes: List[Any], index=[cell_lines: List[int]])
    *_labels = pd.Series(..., columns=["ic_50" if regression else "labels"], index=[cell_lines: List[int]])
    *_classes = pd.Series(..., columns=["labels"], index=[cell_lines: List[int]])
    train_scores = pd.Series(..., columns=["ic_50"], index=[cell_lines: List[int]])
    return:
        fit_runtime: float
        test_runtime: float
        acc: dict of model performance (keys = [mse] if regression else [sensitivity, specificity, f1, youden_J, mcc]
        sorted_features: List
        num_features: Union[int, None]
    """

    if kwrags.training.weight_samples:
        if kwrags.training.simple_weight:
            weights = calculate_simple_weights(kwrags.data.drug_threshold, train_scores)
        else:
            weights = calculate_linear_weights(kwrags.data.drug_threshold, train_scores)
    else:
        weights = None

    if kwrags.training.regression:
        # rf = RandomForestRegressor(**rf_params)
        rf = BiasedRandomForestRegressor(
            kwrags, train_classes, train_scores, **rf_params
        )

    else:
        # rf = RandomForestClassifier(**rf_params)
        rf = BiasedRandomForestClassifier(
            kwrags, train_classes, train_scores, **rf_params
        )

    start = time.time()
    rf.fit(train_features, train_labels, sample_weight=weights)
    fit_runtime = time.time() - start

    # Use the forest's predict method on the test data
    start = time.time()
    predictions = rf.predict(test_features)
    test_runtime = time.time() - start

    if output_preds:
        output_file = os.path.join(kwrags.intermediate_output, kwrags.data.drug_name)
        os.makedirs(output_file, exist_ok=True)
        output_file = os.path.join(
            output_file, f"{kwrags.model.current_model}_preds.json"
        )
    else:
        output_file = None
    acc = calc_accuracy(
        test_labels, predictions, test_classes, kwrags.training.regression, output_file
    )

    if features_names is not None:  # not used/reported for the random model
        col_name = "feature_importance"
        features_importance = pd.DataFrame(
            {"genes": features_names, col_name: rf.biased_feature_importance}
        )  # biased_feature_importance will return the normal features in case of no bias
        if kwrags.training.bias_rf:
            num_trees_features = [
                len(tree.feature_names_in_) for tree in rf.estimators_
            ]
            num_features = statistics.mean(num_trees_features)
        else:
            num_features = len(features_importance)
            num_trees_features = num_features
        sorted_features = features_importance.sort_values(
            by=[col_name], ascending=False
        )[:100]
        sorted_features = (
            sorted_features.astype(str)
            .merge(
                kwrags.data.processed_files.entrez_symbols.astype(str),
                left_on="genes",
                right_on="GeneID",
                how="left",
            )
            .drop(["genes", "GeneID"], axis=1)
        )
    else:
        sorted_features = num_features = num_trees_features = "not_reported"
    return (
        fit_runtime,
        test_runtime,
        acc,
        sorted_features,
        num_features,
        num_trees_features,
    )


def get_rand_rf_results(
    rf_params: dict,
    features_list: list,
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
    train_labels: pd.Series,
    test_labels: pd.Series,
    train_scores: pd.Series,
    train_classes: pd.Series,
    test_classes: pd.Series,
    kwargs: Kwargs,
):
    """
    rf_params: dict of parameters for sklearn.ensemble.RandomForestRegressor/Classifier
    features_list: List[List[genes: Any], ...]
    *_features = pd.DataFrame(..., columns=[genes: List[Any], index=[cell_lines: List[int]])
    *_labels = pd.Series(..., columns=["ic_50" if regression else "labels"], index=[cell_lines: List[int]])
    test_classes = pd.Series(..., columns=["labels"], index=[cell_lines: List[int]])
    train_scores = pd.Series(..., columns=["ic_50"], index=[cell_lines: List[int]])
    return:
        avg_fit_runtime: float
        avg_test_runtime: float
        acc: dict of model performance (keys = [mse] if regression else [sensitivity, specificity, f1, youden_J, mcc]
    """
    fit_runtimes = []
    test_runtimes = []
    accs = []
    for features in features_list:
        rand_train_features = train_features.loc[:, features]
        rand_test_features = test_features.loc[:, features]
        fit_runtime, test_runtime, acc, _, _, _ = do_rf(
            rf_params,
            features,
            rand_train_features,
            train_labels,
            train_scores,
            train_classes,
            rand_test_features,
            test_labels,
            test_classes,
            kwargs,
        )
        fit_runtimes.append(fit_runtime)
        test_runtimes.append(test_runtime)
        accs.append(acc)
    avg_fit_runtime = statistics.mean(fit_runtimes)
    avg_test_runtime = statistics.mean(test_runtimes)
    avg_acc = pd.DataFrame(accs).mean().to_dict()
    return avg_fit_runtime, avg_test_runtime, avg_acc
