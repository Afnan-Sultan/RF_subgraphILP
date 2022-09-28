import json
import logging
import os.path

from tqdm import tqdm

from manger.config import Kwargs
from manger.data.feature_split import feature_split
from manger.training.train_best_parameters import best_model
from manger.utils import NewJsonEncoder, get_thresh

logger = logging.getLogger(__name__)


def train_final(drugs_info: dict, kwargs: Kwargs):
    """
    drugs_info = {$drug_name: {"gene_mat": $gene_mat, "meta_data": $meta_data}}
        gene_mat = pd.DataFrame(..., columns=[cell_lines: List[int]], index= [genes: List[Any])
        meta_data = pd.DataFrame(..., columns=["ic_50", "labels"], index= [cell_lines: List[int]])
    return None, outputs results to a file
    """

    test_results_file = os.path.join(kwargs.results_dir, "test_results.jsonl")
    if not os.path.isfile(test_results_file):
        with open(test_results_file, "w") as _:
            pass

    for drug_name, info_dict in tqdm(
        drugs_info.items(), desc="training models per drug..."
    ):
        kwargs.data.drug_name = drug_name

        drug_thresh = get_thresh(
            kwargs.data.drug_name, kwargs.data.processed_files.thresholds
        )
        kwargs.data.drug_threshold = drug_thresh

        # split to train and test
        splits = feature_split(info_dict["gene_mat"], info_dict["meta_data"], kwargs)

        all_models = {"parameters_grid": kwargs.training.parameters_grid}
        results = {}
        for specified_model in kwargs.model.model_names:
            results[specified_model] = {}
            kwargs.model.current_model = specified_model
            if kwargs.training.regression:  # set labels to the ic-50 scores
                train_labels = splits["train"][2]
                test_labels = splits["test"][2]
            else:  # set labels to the sensitivity discretized labels
                train_labels = splits["train"][1]
                test_labels = splits["test"][1]
            for idx, rf_params in kwargs.training.parameters_grid.items():
                test_results = best_model(
                    rf_params,
                    splits["train"][0],  # train features
                    train_labels,  # train labels
                    splits["train"][4],  # train classes
                    splits["train"][3],  # train scores
                    splits["test"][0],  # test features
                    test_labels,  # test labels
                    splits["test"][3],  # test classes
                    kwargs,
                )

                for model, output in test_results.items():
                    results[model][idx] = {
                        "params": rf_params,
                        "train_runtime": output[0],
                        "test_runtime": output[1],
                        "test_scores": output[2],
                        "features_importance": output[3],
                        "num_features": output[4],
                        "num_tress_features": output[5],
                    }
            all_models[specified_model] = results[specified_model]

        with open(test_results_file, "a") as convert_file:
            convert_file.write(
                json.dumps({drug_name: all_models}, indent=4, cls=NewJsonEncoder)
            )
            convert_file.write("\n")
