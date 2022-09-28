import pandas as pd
from manger.config import Kwargs
from manger.models.random_forest import which_rf


def best_model(
    best_params: dict,
    train_features: pd.DataFrame,
    train_labels: pd.Series,
    train_classes: pd.Series,
    train_scores: pd.Series,
    test_features: pd.DataFrame,
    test_labels: pd.Series,
    test_classes: pd.Series,
    kwargs: Kwargs,
):
    """
    train and test the-best-parameters/model
    *_features = pd.DataFrame(..., columns=[genes: List[Any], index=[cell_lines: List[int]])
    *_labels = pd.Series(..., columns=["ic_50" if regression else "labels"], index=[cell_lines: List[int]])
    *_classes = pd.Series(..., columns=["labels"], index=[cell_lines: List[int]])
    train_scores = pd.Series(..., columns=["ic_50"], index=[cell_lines: List[int]])
    return: dict of results for the final model
    """
    train_features.columns = train_features.columns.astype(str)
    test_features.columns = train_features.columns.astype(str)

    if "subgraphilp" in kwargs.model.current_model:
        kwargs.data.output_num_feature = True
    else:
        kwargs.data.output_num_feature = False

    rf_results = {}
    (
        fit_runtime,
        test_runtime,
        acc,
        sorted_features,
        num_features,
        num_trees_features,
    ) = which_rf(
        best_params,
        train_features,
        train_classes,
        train_scores,
        train_labels,
        test_features,
        test_labels,
        test_classes,
        kwargs,
        output_preds=True,
    )

    rf_results[kwargs.model.current_model] = [
        fit_runtime,
        test_runtime,
        acc,
        sorted_features,
        num_features,
        num_trees_features,
    ]
    return rf_results
