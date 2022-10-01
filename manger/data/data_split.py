import json
import logging
import os

import pandas as pd
from manger.config import Kwargs
from manger.utils import NewJsonEncoder

logger = logging.getLogger(__name__)


def split_data(gene_mat: pd.DataFrame, meta_data: pd.DataFrame, kwargs: Kwargs):
    """
    split data into train and test sets, stratified with respect to sensitive/resistant assignments.
    gene_mat = pd.DataFrame(..., columns=[cell_lines: List[int]], index= [genes: List[Any])
    meta_data = pd.DataFrame(..., columns=["ic_50", "labels"], index= [cell_lines: List[int]])
    return: dict of the train and test splits with info for both classification and regression
    """
    gene_mat = gene_mat.transpose()  # so that genes are the features

    # make sure index type is the same to enable sampling
    gene_mat.index = gene_mat.index.map(int)
    meta_data.index = meta_data.index.map(int)

    scores, classes = meta_data["ic_50"], meta_data["Labels"]

    # randomly select 20% of the data to be test-set, and account for class imbalance.
    stratified_sample = (
        classes.groupby(classes)
        .apply(lambda x: x.sample(frac=0.2, random_state=kwargs.training.random_state))
        .droplevel(0)
    )

    # sample the training and test sets according to the above sampling
    test_features = gene_mat.loc[stratified_sample.index]
    test_classes = classes.loc[test_features.index]
    test_scores = scores.loc[test_features.index]

    train_features = gene_mat.drop(test_features.index)
    train_classes = classes.loc[train_features.index]
    train_scores = scores.loc[train_features.index]

    train_info = {
        "data": train_features,
        "scores": train_scores,
        "classes": train_classes,
    }
    test_info = {
        "data": test_features,
        "scores": test_scores,
        "classes": test_classes,
    }
    splits = {"train": train_info, "test": test_info}

    file_dir = os.path.join(
        kwargs.matrices_output_dir, kwargs.data.drug_name, "train_test_splits.json"
    )
    with open(file_dir, "w") as split_output:
        split_output.write(json.dumps(splits, indent=2, cls=NewJsonEncoder))
    return splits
