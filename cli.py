import json
import logging
import os
import warnings

import typer as typer
from joblib import Parallel, delayed
from tqdm import tqdm

from manager.config import Kwargs
from manager.data.filter_drugs import filter_drugs
from manager.train import train_models

app = typer.Typer()


@app.command()
def train(
    kwargs_file: str = typer.Argument(
        ...,
        help="Path to the config file",
    )
):
    """
    train a parameters list/grid using either grid search and cross validation or direct training.
    In case of grid search, the best parameters list wrt cv is used for training the best model.
    """
    # parse the configuration file
    kwargs = Kwargs.parse_file(kwargs_file)

    # output the configuration in the results' folder for reference
    with open(os.path.join(kwargs.results_dir, "config.json"), "w") as config_file:
        config_file.write(json.dumps(json.loads(open(kwargs_file).read()), indent=4))

    if os.path.isfile(kwargs.results_doc):
        if kwargs.overwrite_results:
            with open(kwargs.results_doc, "w") as _:
                pass
        else:
            warnings.warn(
                "Results file already exists, new results will be appended to it. "
                "To avoid this behaviour, set kwargs.data.overwrite_results to True"
            )

    logging.basicConfig(
        filename=os.path.join(kwargs.results_dir, "logger.txt"),
        format="%(asctime)s - %(message)s",
        level=logging.INFO,
    )

    selected_drugs_info = filter_drugs(kwargs)

    if kwargs.data.drug_subset is not None:
        if isinstance(kwargs.data.drug_subset, list):
            if not kwargs.data.include_subset:
                selected_drugs_info = {
                    k: selected_drugs_info[k]
                    for k in selected_drugs_info.keys()
                    if k not in kwargs.data.drug_subset
                }
            else:
                selected_drugs_info = {
                    k: selected_drugs_info[k] for k in kwargs.data.drug_subset
                }
        else:
            selected_drugs_info = {
                k: selected_drugs_info[k]
                for k in list(selected_drugs_info.keys())[: kwargs.data.drug_subset]
            }

    if kwargs.training.target_root_node:
        drugs_info = {}
        for drug, info in selected_drugs_info.items():
            if drug.split("___")[0] in kwargs.data.processed_files.drugs_targets.keys():
                drugs_info[drug] = info
    else:
        drugs_info = selected_drugs_info

    if kwargs.training.drugs_n_jobs > 1:
        Parallel(n_jobs=kwargs.training.drugs_n_jobs, verbose=10)(
            delayed(train_models)(drug_name, data_split, kwargs.copy(deep=True))
            for drug_name, data_split in drugs_info.items()
        )
    else:
        for drug_name, data_split in tqdm(
            drugs_info.items(), desc="training models per drug..."
        ):
            train_models(drug_name, data_split, kwargs)


if __name__ == "__main__":
    train("config.json")
    app()
