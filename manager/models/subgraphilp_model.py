import logging
import os.path
from typing import Union

import numpy as np
import pandas as pd
from manager.config import Kwargs
from manager.models.utils.subgraphilp import (
    differential_expression,
    run_subgraphilp_executables,
    subgraphilp_features,
)

logger = logging.getLogger(__name__)


def subgraphilp_model(
    train_features: pd.DataFrame,
    train_classes: pd.Series,
    kwargs: Kwargs,
    tree_idx: Union[int, None] = None,
):
    """
    perform subgraphILP feature selection
    train_features = pd.DataFrame(..., columns=[genes: List[Any], index=[cell_lines: List[int]])
    train_classes = pd.Series(..., columns=["labels"], index=[cell_lines: List[int]])
    k = the index of the current cross validation
    tree_idx = the index of the current tree in the RF
    return:
        selected_features List of selected genes
    """
    if kwargs.training.cv_idx is not None:  # performing cross validation
        drug_output_dir = os.path.join(
            kwargs.intermediate_output,
            kwargs.data.drug_name,
            f"params_comb_{kwargs.training.gcv_idx}",
            f"subgraphilp_cv_{kwargs.training.cv_idx}",
        )
    else:
        drug_output_dir = os.path.join(
            kwargs.intermediate_output,
            kwargs.data.drug_name,
            f"subgraphilp_best_model",
        )
    if tree_idx is not None:
        drug_output_dir = os.path.join(drug_output_dir, f"tree_{tree_idx}")
    os.makedirs(drug_output_dir, exist_ok=True)

    de_file = differential_expression(
        train_features,
        train_classes,
        drug_output_dir,
        de_method=kwargs.training.de_method,
    )

    run_subgraphilp_executables(de_file, drug_output_dir, kwargs)

    selected_features = subgraphilp_features(
        drug_output_dir, train_features.columns.to_list(), kwargs
    )
    return selected_features
