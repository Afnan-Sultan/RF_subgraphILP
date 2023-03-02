import os
import statistics
import time
from math import ceil
from typing import List, Union

import pandas as pd
from manager.config import Kwargs
from manager.models.biased_random_forest import (
    BiasedRandomForestClassifier,
    BiasedRandomForestRegressor,
)
from manager.training.feature_selection import feature_selection
from manager.training.scoring import calc_accuracy
from manager.training.weighting_samples import get_weights


def get_feature_importance(rf, features_names, kwargs):
    col_name = "feature_importance"
    features_importance = pd.DataFrame(
        {
            "genes": features_names,
            col_name: rf.biased_feature_importance,
        }  # biased_feature_importance will return all features in case of no bias
    )
    num_features_overall = num_features = num_trees_features = len(features_importance)

    # filter and sort non-zero feature_importance
    features_importance = features_importance[features_importance[col_name] > 0]
    sorted_features = features_importance.sort_values(by=[col_name], ascending=False)

    # convert gene entrez-ids to gene-symbols
    sorted_features = (
        sorted_features.astype(str)
        .merge(
            kwargs.data.processed_files.entrez_symbols.astype(str),
            left_on="genes",
            right_on="GeneID",
            how="left",
        )
        .drop(["genes", "GeneID"], axis=1)
    )

    if kwargs.training.bias_rf:
        trees_features = {
            feature for tree in rf.estimators_ for feature in tree.feature_names_in_
        }
        num_features_overall = len(trees_features)

        # calculate number of features used for each tree
        num_trees_features = [len(tree.feature_names_in_) for tree in rf.estimators_]

        # report average number of features per tree as the final number of features
        num_features = ceil(statistics.mean(num_trees_features))
    return (
        sorted_features,
        num_features_overall,
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
    kwargs: Kwargs,
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

    # instantiate a model
    if kwargs.training.regression:
        # rf = RandomForestRegressor(**rf_params)
        rf = BiasedRandomForestRegressor(
            kwargs, train_classes, train_scores, **rf_params
        )
    else:
        # rf = RandomForestClassifier(**rf_params)
        rf = BiasedRandomForestClassifier(
            kwargs,
            train_classes,
            train_scores,
            **rf_params,
        )

    # ensure the column names are strings to be recognized as feature names for random forest
    train_features.columns = train_features.columns.astype(str)
    test_features.columns = test_features.columns.astype(str)

    # Fit
    start = time.time()
    rf.fit(
        train_features, train_labels, sample_weight=get_weights(train_scores, kwargs)
    )
    fit_runtime = time.time() - start

    # Predict
    start = time.time()
    predictions = rf.predict(test_features)
    test_runtime = time.time() - start

    # evaluation metrics
    if output_preds:
        # output predictions in case later analysis are needed. Only used for best model in case of grid search.
        output_file = os.path.join(kwargs.intermediate_output, kwargs.data.drug_name)
        os.makedirs(output_file, exist_ok=True)
        output_file = os.path.join(
            output_file, f"{kwargs.model.current_model}_preds.json"
        )
    else:
        output_file = None

    acc = calc_accuracy(
        test_labels, predictions, test_classes, kwargs.training.regression, output_file
    )
    kwargs.data.acc_subset = list(acc.keys())

    if features_names is not None:
        # retrieve features with importance > 0
        (
            sorted_features,
            num_features_overall,
            num_features,
            num_trees_features,
        ) = get_feature_importance(rf, features_names, kwargs)
    else:
        # not used/reported for the random model
        sorted_features = (
            num_features_overall
        ) = num_features = num_trees_features = "not_reported"

    return (
        fit_runtime,
        test_runtime,
        acc,
        sorted_features,
        num_features_overall,
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
    independent function for the random model, to average the performance and return single values as the other models
    invoked when bias_rf is NOT selected, to train n rf models with the initial number of features passed to the model
    limited to the same number of features selected from the subgraphilp model.

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
        fit_runtime, test_runtime, acc, _, _, _, _ = do_rf(
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
    Select the requested approach for RF

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
            num_features_overall,
            num_features,
            num_trees_features,
        ) = do_rf(
            rf_params=rf_params,
            features_names=features,
            train_features=train_features,
            train_labels=train_labels,
            train_scores=train_scores,
            train_classes=train_classes,
            test_features=test_features,
            test_labels=test_labels,
            test_classes=test_classes,
            kwargs=kwargs,
            output_preds=output_preds,
        )
    else:
        to_rf = feature_selection(
            train_features, train_classes, train_scores, kwargs, test_features
        )
        if kwargs.training.get_features_only:
            return (
                None,
                None,
                None,
                to_rf["features"],
                len(to_rf["features"]),
                None,
                None,
            )
        if model == "random":
            fit_runtime, test_runtime, acc = get_rand_rf_results(
                rf_params=rf_params,
                features_list=to_rf["features"],
                train_features=train_features,
                test_features=test_features,
                train_labels=train_labels,
                test_labels=test_labels,
                train_scores=train_scores,
                train_classes=train_classes,
                test_classes=test_classes,
                kwargs=kwargs,
            )
            num_features_overall = num_features = num_trees_features = len(
                to_rf["features"][0]
            )
            sorted_features = None
        else:
            (
                fit_runtime,
                test_runtime,
                acc,
                sorted_features,
                num_features_overall,
                num_features,
                num_trees_features,
            ) = do_rf(
                rf_params=rf_params,
                features_names=to_rf["features"],
                train_features=to_rf["train_features"],
                test_features=to_rf["test_features"],
                train_labels=train_labels,
                test_labels=test_labels,
                train_scores=train_scores,
                train_classes=train_classes,
                test_classes=test_classes,
                kwargs=kwargs,
                output_preds=output_preds,
            )
    return (
        fit_runtime,
        test_runtime,
        acc,
        sorted_features,
        num_features_overall,
        num_features,
        num_trees_features,
    )
