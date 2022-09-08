import json
import logging

from manger.config import Kwargs
from manger.data.feature_split import feature_split
from manger.training.grid_search_cv import grid_search_cv
from manger.utils import NewJsonEncoder, get_thresh

logger = logging.getLogger(__name__)


def train_gs_cv(drug_name: str, drug_info: dict, kwargs: Kwargs):
    """
    drugs_info = {$drug_name: {"gene_mat": $gene_mat, "meta_data": $meta_data}}
        gene_mat = pd.DataFrame(..., columns=[cell_lines: List[int]], index= [genes: List[Any])
        meta_data = pd.DataFrame(..., columns=["ic_50", "labels"], index= [cell_lines: List[int]])
    return None, outputs results to a file
    """
    kwargs.data.drug_name = drug_name

    drug_thresh = get_thresh(
        kwargs.data.drug_name, kwargs.data.processed_files.thresholds
    )
    kwargs.data.drug_threshold = drug_thresh

    # split to train and test
    logger.info(f"*** Splitting data into train and test sets for {drug_name} ***")
    splits = feature_split(drug_info["gene_mat"], drug_info["meta_data"], kwargs)

    all_models = {"parameters_grid": kwargs.training.parameters_grid}
    for specified_model in kwargs.model.model_names:
        kwargs.model.current_model = specified_model
        if kwargs.training.regression:  # set labels to the ic-50 scores
            train_labels = splits["train"][2]
            test_labels = splits["test"][2]
        else:  # set labels to the sensitivity discretized labels
            train_labels = splits["train"][1]
            test_labels = splits["test"][1]

        logger.info(
            f"=== Starting grid search cross validation for {drug_name} using {specified_model.upper()} model ==="
        )
        grid_results = grid_search_cv(
            train_features=splits["train"][0],
            train_labels=train_labels,
            train_classes=splits["train"][4],
            train_scores=splits["train"][3],
            test_features=splits["test"][0],
            test_labels=test_labels,
            test_classes=splits["test"][3],
            kwargs=kwargs,
        )
        if (
            grid_results is None
        ):  # corr_thresh models sometime report no correlated features > corr_thresh
            all_models[specified_model] = None
        else:
            for sub_model, results in grid_results.items():
                all_models[sub_model] = results

    with open(kwargs.results_doc, "a") as convert_file:
        convert_file.write(
            json.dumps({drug_name: all_models}, indent=4, cls=NewJsonEncoder)
        )
        convert_file.write("\n")
