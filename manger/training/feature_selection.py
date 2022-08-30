import logging
from typing import Union

import pandas as pd

from manger.config import Kwargs
from manger.models.subgraphilp_model import subgraphilp_model
from manger.training.utils import calc_corr, get_num_features, random_samples

logger = logging.getLogger(__name__)


def feature_selection(
    train_features: pd.DataFrame,
    train_classes: pd.Series,
    train_scores: pd.Series,
    kwargs: Kwargs,
    test_features: Union[pd.DataFrame, None] = None,
    tree_idx: Union[int, None] = None,
):
    """
    perform the feature selection method specified in model
    train_features = pd.DataFrame(..., columns=[genes: List[Any], index=[cell_lines: List[int]])
    train_classes = pd.Series(..., columns=["labels"], index=[cell_lines: List[int]])
    train_scores = pd.Series(..., columns=["ic_50"], index=[cell_lines: List[int]])
    return: np.array for train_features with only selected features
    """
    model = kwargs.model.current_model

    # identify important features
    if model == "subgraphilp":  # using network ILP
        model_features = subgraphilp_model(
            train_features, train_classes, kwargs, tree_idx
        )
    else:
        if model in ["random", "corr_num"]:
            if kwargs.data.num_features_file is None:
                infile = kwargs.subgraphilp_num_features_output_file
            else:
                infile = kwargs.data.num_features_file
            num_features = get_num_features(infile, kwargs.data.drug_name)
        else:
            num_features = None

        if model.startswith("corr"):
            corr_sorted_scores = calc_corr(train_features.copy(deep=True), train_scores)
            if (
                model == "corr_num"
            ):  # using the top-k number of correlated features to ic50. k_corr = k_ILP
                model_features = corr_sorted_scores.index.to_list()[:num_features]
            else:  # using the features that have correlation with the ic_50 scores higher than or equal a threshold
                model_features = corr_sorted_scores[
                    corr_sorted_scores.values >= kwargs.training.corr_thresh
                ].index.to_list()

        elif model == "random":  # using k randomly selected features. k_random = k_ILP
            if kwargs.training.bias_rf:
                rand_features = random_samples(
                    train_features.columns.to_list(), num_features, n_rand=1
                )
                model_features = rand_features[0]
            else:
                rand_features = random_samples(
                    train_features.columns.to_list(),
                    num_features,
                    n_rand=kwargs.training.num_random_samples,
                )
                model_features = rand_features
        else:  # performing the usual random forest
            model_features = train_features.columns.to_list()

    # select the new train and test features
    if len(model_features) > 0:
        if model == "random" and not kwargs.training.bias_rf:
            model_train_features = (
                None  # it will be assigned later for each randomly selected subset
            )
        else:
            model_train_features = train_features.loc[:, model_features]

        if test_features is None:  # subset train features only
            return {"features": model_features, "train_features": model_train_features}
        else:  # subset both train and test features because we will be doing feature selection only.
            # should be invoked only when NOT biasing random forest
            assert not kwargs.training.bias_rf

            if model == "random":
                model_test_features = (
                    None  # it will be assigned later for each randomly selected subset
                )
            else:
                model_test_features = test_features.loc[:, model_features]

            return {
                "features": model_features,
                "train_features": model_train_features,
                "test_features": model_test_features,
            }
    else:
        return None
